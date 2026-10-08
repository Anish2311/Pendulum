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
        self.m = 200
    
    def show(self):
        x2 = self.pos[0] + self.l * cos(self.ang)
        y2 = self.pos[1] + self.l * sin(self.ang)
        strokeWeight(4)
        stroke(255 - ((evalu+50)/50)*255,((evalu+50)/50)*255,20)
        line(self.pos[0],self.pos[1],x2,y2)
        noStroke()
        fill(255 - ((evalu+50)/50)*255,((evalu+50)/50)*255,20)
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

    if fcounter == 300 or pendulum.pos[0] < 0 or pendulum.pos[0] > width:
        fir = False
        if frame_count <= 301:
            fir = True
        fit(fir)
        reset()
        fcounter = 0
    else:
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

    pendulum.ang = HALF_PI
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
    w_x = 1           # Position penalty (keeps cart centered)

    ang = pendulum.ang
    # if ang < 0:
    #     ang += 2*PI

    theta_err = (1 + np.sin(ang))/2

    reward = -(w_theta * (theta_err ** 2) + 
            w_theta_dot * ((pendulum.angVel/2) ** 2) + 
            w_x * (((pendulum.pos[0]-(width/2))/(width/2)) ** 2))
    
    return reward*100


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

discount_factor = 0.999
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

    state = [(abs(pendulum.ang)%(2*PI))/(2*PI),pendulum.angVel/2,pendulum.pos[0]/width,pendulum.vel/25]
    output = actor.traverse(state)[0]
    immReward = evaluate()

    opt = [-0.5,0,0.5]

    expl = max(0,(1000-episodes)/1111)

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

    cache.append([state,ind,immReward,output])

    f = opt[ind]

def fit(first):
    global cache
    global moment_grad_w_critic
    global moment_grad_b_critic
    global moment_grad_w_actor
    global moment_grad_b_actor

    accum_grad_w_critic = 0
    accum_grad_b_critic = 0
    accum_grad_w_actor = 0
    accum_grad_b_actor = 0

        

    for q in range(len(cache) - 1):
        t_iter = cache[q]
        t_1iter = cache[q+1]

        st = t_iter[0]
        selt = t_iter[1]
        opt = t_iter[3]

        st_1 = t_1iter[0]
        rt_1 = t_1iter[2]


        curValue = critic.traverse(st_1)[0][0]
        actualValue = rt_1 + discount_factor*curValue

        stateValue = critic.traverse(st)[0][0]

        advantage = actualValue - stateValue

        print(advantage)

        critic.errorFunction([[stateValue]],[[actualValue]])
        grad_w_critic, grad_b_critic = critic.backPropogate()
        if q == 0:
            accum_grad_w_critic = grad_w_critic
            accum_grad_b_critic = grad_b_critic
        else:
            for jind in range(len(accum_grad_w_critic)):
                accum_grad_w_critic[jind] = grad_w_critic[jind] + accum_grad_w_critic[jind]
                accum_grad_b_critic[jind] = grad_b_critic[jind] + accum_grad_b_critic[jind]

        otpt = actor.traverse(st)[0]
        j = selt
        lossFunc = [0,0,0]
        lossFunc[j] = 1
        lossFunc = np.array([lossFunc])
        lossFunc = (opt - lossFunc)*advantage

        grad_w_actor, grad_b_actor = actor.backPropogate(lossFunc)
        if q == 0:
            accum_grad_w_actor = grad_w_actor
            accum_grad_b_actor = grad_b_actor
        else:
            for jind in range(len(accum_grad_w_actor)):
                accum_grad_w_actor[jind] = grad_w_actor[jind] + accum_grad_w_actor[jind]
                accum_grad_b_actor[jind] = grad_b_actor[jind] + accum_grad_b_actor[jind]

    if first:
        moment_grad_w_critic = accum_grad_w_critic
        moment_grad_b_critic = accum_grad_b_critic
        moment_grad_w_actor = accum_grad_w_actor
        moment_grad_b_actor = accum_grad_b_actor

    else:
        for k in range(len(moment_grad_b_actor)):
            moment_grad_w_critic[k] = 0.1*accum_grad_w_critic[k] + moment_grad_w_critic[k] * 0.9
            moment_grad_b_critic[k] = 0.1*accum_grad_b_critic[k] + moment_grad_b_critic[k] * 0.9
            moment_grad_w_actor[k] = 0.1*accum_grad_w_actor[k] + moment_grad_w_actor[k] * 0.9
            moment_grad_b_actor[k] = 0.1*accum_grad_b_actor[k] + moment_grad_b_actor[k] * 0.9
        
    for i in range(len(accum_grad_w_critic)):
        critic.weights[i] -= moment_grad_w_critic[i]*learning_rate
        critic.biases[i] -= moment_grad_b_critic[i]*learning_rate

    for i in range(len(grad_w_actor)):
        actor.weights[i] -= moment_grad_w_actor[i]*(learning_rate)
        actor.biases[i] -= moment_grad_b_actor[i]*(learning_rate)

# def train():

#     steps = 500

#     for k in range(steps):



if __name__ == '__main__':
  run()