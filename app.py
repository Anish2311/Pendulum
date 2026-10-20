import json
import numpy as np
from p5 import *
from datetime import datetime

class Pendulum:
    def __init__(self,x,y,l,a):
        self.pos = [x,y]
        self.vel = 0
        self.acc = 0
        self.ang = a
        self.angVel = 0
        self.angAcc = 0
        self.l = l
        self.m = 200
    
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
        self.acc = f + (self.l / 10) * self.angVel**2 * cos(self.ang)
        self.vel += self.acc
        self.vel = max(-25,min(25,self.vel))
        self.pos[0] += self.vel
        self.angAcc = g*cos(self.ang) + (f/self.m)*sin(self.ang)
        self.angVel += self.angAcc
        self.angVel += damp*self.angVel
        self.angVel = max(-1,min(1,self.angVel))
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
    global episodes

    background(0)
    strokeWeight(1)
    stroke(0,100,150)
    line(0,height/1.75,width,height/1.75)
    evalu = evaluate()
    
    fill(255)
    text(str(evalu),1000,100)
    pendulum.show()

    f = 0
    # if fcounter == 180:
    #     fcounter = 0
    #     reset()
    # else:
    action()
    fcounter += 1


    # if(keyIsDown and key == RIGHT): # type: ignore
    #     f = 0.5
    
    # elif(keyIsDown and key == LEFT): # type: ignore
    #     f = -0.5
    # else:
    #     f = 0

    pendulum.update()

def get_state():
    return [sin(pendulum.ang), cos(pendulum.ang),
            pendulum.angVel * 10,
            (pendulum.pos[0] - width/2) / (width/2),
            pendulum.vel / 10]

def reset():
    cache.clear()
    pendulum.ang = 3*HALF_PI + np.random.uniform(-0.3, 0.3)
    pendulum.angVel = 0; pendulum.angAcc = 0
    pendulum.pos = [width/2, height/1.75]
    pendulum.vel = 0; pendulum.acc = 0

def key_pressed():
    global keyIsDown
    keyIsDown = True

def key_released():
    global keyIsDown
    keyIsDown = False
    

def evaluate():
    target = 3 * HALF_PI

    # Wrapped angular difference in [-pi, pi]
    theta_err = (pendulum.ang - target + PI) % (2 * PI) - PI

    x_err = (pendulum.pos[0] - width / 2) / (width / 2)
    angular_velocity = pendulum.angVel

    reward = (
        -1.0 * theta_err**2
        -0.1 * (angular_velocity*10)**2
        -1.0 * x_err**2
    )

    return reward


#----------------------------------------
#         NEURAL NETWORK PART
#----------------------------------------

import numpy as np


class Network:
    def __init__(self, n, larr, o, opf, hidden_activation="tanh"):
        self.opFunc = opf
        self.hidden_activation = hidden_activation

        self.weights = []
        self.biases = []

        self.errorFunctionDeriv = None
        self.activationFunctionDeriv = []
        self.opForLayers = []

        layer_dims = [n] + larr + [o]
        self.totalLayers = len(layer_dims) - 1

        for i in range(self.totalLayers):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]

            is_output = (i == self.totalLayers - 1)

            if is_output and self.opFunc:
                # Small initial logits -> near-uniform policy
                std = 0.01
            elif hidden_activation == "relu":
                std = np.sqrt(2.0 / fan_in)
            else:
                # Xavier normal: sigmoid/tanh
                std = np.sqrt(2.0 / (fan_in + fan_out))

            W = np.random.randn(fan_in, fan_out) * std
            b = np.zeros((1, fan_out))

            self.weights.append(W)
            self.biases.append(b)

    def activationFunction(self, a):
        if self.hidden_activation == "tanh":
            out = np.tanh(a)
            deriv = 1.0 - out**2
        elif self.hidden_activation == "relu":
            out = np.maximum(0.0, a)
            deriv = (a > 0).astype(np.float64)
        else:
            out = 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500)))
            deriv = out * (1.0 - out)

        self.activationFunctionDeriv.append(deriv)
        return out

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


actor = Network(5, [32,32,32], 3, 1, hidden_activation="tanh")
critic = Network(5, [32,32,32], 1, 0, hidden_activation="tanh")

discount_factor = 0.995
learning_rate_actor, learning_rate_critic = 0.01, 0.1
cache = []
episodes = 0

moment_grad_w_critic = 0
moment_grad_b_critic = 0
moment_grad_w_actor = 0
moment_grad_b_actor = 0

def action():
    global f

    state = get_state()
    output = actor.traverse(state)[0]

    opt = [-0.6,0,0.6]

    # print(output)
    j = np.where(output == np.max(output))[0][0]



    f = opt[j]

def goOn():
    global cache
    global f

    state = get_state()
    output = actor.traverse(state)[0]
    immReward = evaluate()

    # print(output)
    j = np.random.random()
    density = 0
    ind = 0
    for k in range(len(output)):
        density += output[k]
        if j < density:
            ind = k
            break

    opt = [-0.6,0,0.6]
    f = opt[ind]

    cache.append([state,ind,immReward])

def clip_grads(gw, gb, max_norm=1.0):
    norm = np.sqrt(sum(np.sum(g**2) for g in gw + gb))
    s = min(1.0, max_norm / (norm + 1e-8))
    return [g * s for g in gw], [g * s for g in gb]

# initialise once, outside fit(), so you don't need the `first` flag
mw_c = [np.zeros_like(w) for w in critic.weights]
mb_c = [np.zeros_like(b) for b in critic.biases]
mw_a = [np.zeros_like(w) for w in actor.weights]
mb_a = [np.zeros_like(b) for b in actor.biases]

def fit(num_episode):
    global episodes
    n = len(cache) - 2
    aw_c = [np.zeros_like(w) for w in critic.weights]
    ab_c = [np.zeros_like(b) for b in critic.biases]
    aw_a = [np.zeros_like(w) for w in actor.weights]
    ab_a = [np.zeros_like(b) for b in actor.biases]
    avgR = 0
    avgH = 0
    beta = 0.001 - (0.001*(episodes/num_episode))

    for q in range(n):
        st, a = cache[q][0], cache[q][1]
        st_1, r = cache[q + 1][0], cache[q + 1][2]
        avgR += r

        v_next = critic.traverse(st_1)[0][0]       
        target = r + discount_factor * v_next      
        
        v = critic.traverse(st)[0][0]              
        advantage = (target - v)

        critic.errorFunction([[v]], [[target]])
        gw, gb = critic.backPropogate()
        for i in range(len(aw_c)):
            aw_c[i] += gw[i]; ab_c[i] += gb[i]

        p = actor.traverse(st)[0]
        # print(
        #     f"A={advantage:.5f}, "
        #     f"V={float(v):.5f}, "
        #     f"target={float(target):.5f}"
        # )
        logp = np.log(p + 1e-8)
        H = -np.sum(p * logp)
        H = H
        avgH += H

        delta = advantage * p                  # A * p
        delta[a] -= advantage                  # A * (p - onehot)
        delta += beta * p * (logp + H)         # entropy bonus
        gw, gb = actor.backPropogate(delta.reshape(1, -1))
        for i in range(len(aw_a)):
            aw_a[i] += gw[i]; ab_a[i] += gb[i]

    # print(avgR/n,"  -> Reward")
    # print(avgH/n,"  -> Entropy")

    #average over the episode, clip, then momentum
    aw_c, ab_c = clip_grads([g / n for g in aw_c], [g / n for g in ab_c])
    aw_a, ab_a = clip_grads([g / n for g in aw_a], [g / n for g in ab_a])

    # aw_c = [a/n for a in aw_c]
    # ab_c = [a/n for a in ab_c]
    # aw_a = [a/n for a in aw_a]
    # ab_a = [a/n for a in ab_a]

    for i in range(len(critic.weights)):
        # mw_c[i] = 0.9 * mw_c[i] + aw_c[i]
        # mb_c[i] = 0.9 * mb_c[i] + ab_c[i]
        critic.weights[i] -= learning_rate_critic * aw_c[i]
        critic.biases[i]  -= learning_rate_critic * ab_c[i]

    for i in range(len(actor.weights)):
        # aw_a[i] = 0.9 * aw_a[i] + aw_a[i]
        # ab_a[i] = 0.9 * ab_a[i] + ab_a[i]
        actor.weights[i] -= learning_rate_actor * aw_a[i]
        actor.biases[i]  -= learning_rate_actor * ab_a[i]

    return (avgR/n,avgH/n)

def train(num_episode, num_frames):
    global episodes
    for ep in range(num_episode):
        reset()
        for _ in range(num_frames):
            goOn()
            pendulum.update()
        episodes += 1
        res = fit(num_episode)
        if ep%100 == 0:
            print(res)

def ser(wts) :
    return [w.tolist() for w in wts]

if __name__ == '__main__':
    now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    train(2000, 180)

    with open(f"pendulum-{now}.json", "w") as f :
        json.dump({
            "actor_w" : ser(actor.weights),
            "actor_b" : ser(actor.biases),
            "critic_w" : ser(critic.weights),
            "critic_b" : ser(critic.biases),
        }, f)
    reset()
    run()