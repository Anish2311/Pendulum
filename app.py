import numpy as np
from p5 import *
import pandas as pd


# Download latest version


#-----------------------------------
#          VISUALISATION
#-----------------------------------

class Pendulum:
    def __init__(self,x,y,l,a):
        self.pos = [x,y]
        self.vel = 0
        self.acc = 0
        self.ang = a
        self.angVel = 0
        self.angAcc = 0
        self.l = l
        self.m = 100
    
    def show(self):
        x2 = self.pos[0] + self.l * cos(self.ang)
        y2 = self.pos[1] + self.l * sin(self.ang)
        strokeWeight(4)
        stroke(255 - ((evalu+1.1)/1.1)*255,((evalu+1.1)/1.1)*255,20)
        line(self.pos[0],self.pos[1],x2,y2)
        noStroke()
        fill(255 - ((evalu+1.1)/1.1)*255,((evalu+1.1)/1.1)*255,20)
        circle(x2,y2,self.l/4)
        strokeWeight(2)
        stroke(100 + (self.vel/25)*150)
        noFill()
        circle(self.pos[0],self.pos[1],self.l/7)
        circle(self.pos[0],self.pos[1],5)
    
    def update(self):
        self.acc = f + (self.l/10)*self.angVel*self.angVel*(abs(cos(self.ang))/cos(self.ang))*abs(cos(self.ang))
        self.vel += self.acc
        self.vel = min(25,self.vel)
        self.pos[0] += self.vel
        self.angAcc = g*cos(self.ang) + (f/self.m)*sin(self.ang)
        self.angVel += self.angAcc
        self.angVel += damp*self.angVel
        self.angVel = min(2,self.angVel)
        self.ang += self.angVel


g = 0.005
damp = -0.005
f = 0
evalu = 0
height = 640
width = 1200
pendulum = Pendulum(width/2,height/1.75,200,HALF_PI)
keyIsDown = False
started = False
fcounter = 0

def setup():
    size(width,height)

def draw():
    global f
    global evalu
    global started
    global fcounter 
    background(0)
    strokeWeight(1)
    stroke(0,100,150)
    line(0,height/1.75,width,height/1.75)
    evalu = evaluate()
    
    fill(255)
    text(str(evalu),1000,100)
    pendulum.show()

    f = 0
    if fcounter == 60 or pendulum.pos[0] < 0 or pendulum.pos[0] > width:
        fit()
        reset()
        fcounter = 0
    elif frame_count%5 == 0:
        goOn()
        fcounter += 1


    # if(keyIsDown and key == RIGHT): # type: ignore
    #     f = 0.5
    
    # elif(keyIsDown and key == LEFT): # type: ignore
    #     f = -0.5
    # else:
    #     f = 0

    pendulum.update()

def reset():
    global cache
    global episodes

    cache.clear()

    pendulum.ang = PI
    pendulum.angVel = 0
    pendulum.angAcc = 0

    pendulum.pos = [width/2,height/1.75]
    pendulum.vel = 0
    pendulum.acc = 0

    episodes += 1

    print(episodes)

def key_pressed():
    global keyIsDown
    keyIsDown = True

def key_released():
    global keyIsDown
    keyIsDown = False
    

def evaluate():

    w_theta = 1.0       # Upright angle penalty
    w_theta_dot = 0.1   # Angular velocity penalty (prevents spinning)
    w_x = 0.5          # Position penalty (keeps cart centered)

    ang = (pendulum.ang)%(2*PI)
    if ang < 0:
        ang += 2*PI

    theta_err = 3*HALF_PI - ang
    if theta_err > PI:
        theta_err = 2*PI - theta_err

    offset_err = abs(pendulum.pos[0] - width/2)/(width/2)

    off_t = -1
    off_w = -0.1

    reward = off_t * theta_err/PI + off_w * offset_err
    
    return reward


#----------------------------------------
#         NEURAL NETWORK PART
#----------------------------------------

import numpy as np

class Network():
    def __init__(self, n, larr, o, opf):
        self.opFunc = opf

        self.weights = []
        self.biases = []

        self.errorFunctionDeriv = None
        self.activationFunctionDeriv = []
        self.opForLayers = []

        self.totalLayers = 1 + len(larr)

        layer_dims = [n] + larr + [o]
        for i in range(len(layer_dims) - 1):
            fan_in, fan_out = layer_dims[i], layer_dims[i+1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(np.random.normal(0, std, (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

    def activationFunction(self, a):
        sgnm = 1 / (1 + np.exp(-np.clip(a, -500, 500)))
        self.activationFunctionDeriv.append(sgnm * (1 - sgnm))
        return sgnm

    def outputFunction(self, a):
        exps = np.exp(a - np.max(a, axis=-1, keepdims=True))
        soft = exps / np.sum(exps, axis=-1, keepdims=True)

        if self.opFunc:
            return soft
        return a

    def traverse(self, arr):
        self.activationFunctionDeriv = []
        self.opForLayers = []
        
        ip = np.array(arr, dtype=np.float64)
        if ip.ndim == 1:
            ip = ip.reshape(1, -1)

        for i in range(self.totalLayers):
            self.opForLayers.append(ip)
            op = ip @ self.weights[i] + self.biases[i]
            if i < self.totalLayers - 1:
                op = self.activationFunction(op)
            else:
                op = self.outputFunction(op)
            ip = op

        return op

    def errorFunction(self, op, data):
        data = np.array(data, dtype=np.float64)
        op = np.array(op, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        self.errorFunctionDeriv = (op - data)
        
        return np.mean((op - data) ** 2)

    def backPropogate(self,delt=0):
        gradient_w = []
        gradient_b = []

        delta = self.errorFunctionDeriv
        if type(delt) != int:
            delta = delt

        for i in range(self.totalLayers - 1, -1, -1):
            dW = self.opForLayers[i].T @ delta
            dB = np.sum(delta, axis=0, keepdims=True)
            
            gradient_w.append(dW)
            gradient_b.append(dB)

            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activationFunctionDeriv[i - 1]

        return gradient_w[::-1], gradient_b[::-1]



actor = Network(4,[32,32],3,1)
critic = Network(4,[32,32],1,0)

discount_factor = 0.98
learning_rate = 0.01
cache = []
episodes = 0

moment_grad_w_critic = 0
moment_grad_b_critic = 0
moment_grad_w_actor = 0
moment_grad_b_actor = 0

def goOn():
    global f
    global cache
    global episodes

    state = [(pendulum.ang%(2*PI))/(2*PI),pendulum.angVel,pendulum.pos[0]/width,pendulum.vel/25]
    output = actor.traverse(state)[0]
    immReward = evaluate()

    opt = [-0.5,0,0.5]

    expl = 0

    # print(output)
    j = np.random.random()
    density = 0
    ind = 0
    for k in range(len(output)):
        density += output[k]
        if j < density:
            ind = k
            break

    j = np.random.random()
    if j < expl:
        newInd = np.random.randint(0,3)
        ind = newInd

    cache.append([state,ind,immReward])

    f = opt[ind]

def clip_grads(gw, gb, max_norm=1.0):
    norm = np.sqrt(sum(np.sum(g**2) for g in gw + gb))
    s = min(1.0, max_norm / (norm + 1e-8))
    return [g * s for g in gw], [g * s for g in gb]

# initialise once, outside fit(), so you don't need the `first` flag
mw_c = [np.zeros_like(w) for w in critic.weights]
mb_c = [np.zeros_like(b) for b in critic.biases]
mw_a = [np.zeros_like(w) for w in actor.weights]
mb_a = [np.zeros_like(b) for b in actor.biases]

def fit():
    n = len(cache) - 1
    aw_c = [np.zeros_like(w) for w in critic.weights]
    ab_c = [np.zeros_like(b) for b in critic.biases]
    aw_a = [np.zeros_like(w) for w in actor.weights]
    ab_a = [np.zeros_like(b) for b in actor.biases]
    avgR = 0
    avgH = 0

    for q in range(n):
        st, a = cache[q][0], cache[q][1]
        st_1, r = cache[q + 1][0], cache[q + 1][2]
        avgR += r

        v_next = critic.traverse(st_1)[0][0]      # constant target
        target = r + discount_factor * v_next      # use just r if st_1 is terminal
        v = critic.traverse(st)[0][0]              # last traverse = st, so backprop is correct
        advantage = target - v

        critic.errorFunction([[v]], [[target]])
        gw, gb = critic.backPropogate()
        for i in range(len(aw_c)):
            aw_c[i] += gw[i]; ab_c[i] += gb[i]

        p = actor.traverse(st)[0]
        logp = np.log(p + 1e-8)
        H = -np.sum(p * logp)
        avgH += H

        delta = advantage * p                  # A * p
        delta[a] -= advantage                  # A * (p - onehot)
        delta += 0.001 * p * (logp + H)         # entropy bonus
        gw, gb = actor.backPropogate(delta.reshape(1, -1))
        for i in range(len(aw_a)):
            aw_a[i] += gw[i]; ab_a[i] += gb[i]

    print(avgR/n,"  -> Reward")
    print(avgH/n,"  -> Entropy")

    # average over the episode, clip, then momentum
    aw_c, ab_c = clip_grads([g / n for g in aw_c], [g / n for g in ab_c])
    aw_a, ab_a = clip_grads([g / n for g in aw_a], [g / n for g in ab_a])

    for i in range(len(critic.weights)):
        mw_c[i] = 0.9 * mw_c[i] + aw_c[i]
        mb_c[i] = 0.9 * mb_c[i] + ab_c[i]
        critic.weights[i] -= learning_rate * mw_c[i]
        critic.biases[i]  -= learning_rate * mb_c[i]

    for i in range(len(actor.weights)):
        mw_a[i] = 0.9 * mw_a[i] + aw_a[i]
        mb_a[i] = 0.9 * mb_a[i] + ab_a[i]
        actor.weights[i] -= 0.001 * mw_a[i]
        actor.biases[i]  -= 0.001 * mb_a[i]

# def train():

#     steps = 500

#     for k in range(steps):



if __name__ == '__main__':
  run()