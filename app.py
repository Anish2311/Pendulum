import numpy as np
from p5 import *

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
        stroke(255 - ((evalu+10)/20)*255,((evalu+10)/20)*255,20)
        line(self.pos[0],self.pos[1],x2,y2)
        noStroke()
        fill(255 - ((evalu+10)/20)*255,((evalu+10)/20)*255,20)
        circle(x2,y2,self.l/4)
        strokeWeight(2)
        stroke(100 + (self.vel/25)*150)
        noFill()
        circle(self.pos[0],self.pos[1],self.l/7)
        circle(self.pos[0],self.pos[1],5)
    
    def update(self):
        self.acc = f + (self.l/10)*self.angVel*self.angVel*(abs(cos(self.ang))/cos(self.ang))*abs(cos(self.ang))
        self.vel += self.acc
        self.pos[0] += self.vel
        self.angAcc = g*cos(self.ang) + (f/self.m)*sin(self.ang)
        self.angVel += self.angAcc
        self.angVel += damp*self.angVel
        self.ang += self.angVel


g = 0.005
damp = -0.005
f = 0
evalu = 0
height = 640
width = 1200
pendulum = Pendulum(width/2,height/1.75,200,PI + HALF_PI)
keyIsDown = False

def setup():
    size(width,height)

def draw():
    global f
    global evalu
    background(0)
    strokeWeight(1)
    stroke(0,100,150)
    line(0,height/1.75,width,height/1.75)
    evalu = evaluate()
    fill(255)
    pendulum.show()
    if(keyIsDown and key == RIGHT): # type: ignore
        f = 0.5
    
    elif(keyIsDown and key == LEFT): # type: ignore
        f = -0.5
    else:
        f = 0
    pendulum.update()

def key_pressed():
    global keyIsDown
    keyIsDown = True

def key_released():
    global keyIsDown
    keyIsDown = False
    

def evaluate():
    score = 0
    scaling = 10
    score += scaling*-1*sin(pendulum.ang)
    offset = -1 * scaling * abs(pendulum.pos[0] - width/2)/(width/2)
    score += offset
    score -= abs(pendulum.angVel)*scaling*scaling*scaling/6
    score -= (abs(pendulum.vel)/50)*scaling 
    return score


#----------------------------------------
#         NEURAL NETWORK PART
#----------------------------------------

class Network():
    def __init__(self,n,larr,o):

        self.weights = []
        self.biases = []

        self.errorFunctionDeriv = 0
        self.outputFunctionDeriv = 0
        self.activationFunctionDeriv = []
        self.opForLayers = []

        self.totalLayers = 1+len(larr)

        a = n
        for i in range(self.totalLayers):
            b = o
            if i < len(larr):
                b = larr[i]
            layerWeights = np.abs(np.random.normal(0,0.1,a*b))
            layerWeights = np.reshape(layerWeights,(a,b))
            self.weights.append(layerWeights)
            self.biases.append(np.zeros(b))
            a = b

    def activationFunction(self,a):
        sgnm = 1 / (1 + np.exp(-a))
        app = sgnm*(1-sgnm)
        app = app.T
        self.activationFunctionDeriv.append(app)
        return sgnm

    def outputFunction(self,a):
        m = np.exp(a)
        self.outputFunctionDeriv = m*(m.sum() - m.ndim)/(m*m).sum()
        self.outputFunctionDeriv = self.outputFunctionDeriv.T
        return m/m.sum()

    def traverse(self,arr):
        self.activationFunctionDeriv = []
        self.opForLayers = []
        ip = np.array([arr])

        for i in range(self.totalLayers):
            self.opForLayers.append(ip)
            op = ip @ self.weights[i]
            op = op + self.biases[i]
            if i < self.totalLayers-1:
                op = self.activationFunction(op)
            else:
                op = self.outputFunction(op)
            ip = op

        return op

    def errorFunction(self,op,data):
        sq = op - data
        l = sq.ndim
        self.errorFunctionDeriv = (-2/l)*np.abs(sq)
        self.errorFunctionDeriv = self.errorFunctionDeriv.T
        sq = sq * sq
        msq = sq.sum()/l
        return msq


    def backPropogate(self):

        gradient = []

        initDeriv = self.errorFunctionDeriv * self.outputFunctionDeriv

        for i in range(self.totalLayers-1,-1,-1):
            if i != self.totalLayers-1:
                initDeriv *= self.activationFunctionDeriv[i]
            appDeriv = initDeriv @ self.opForLayers[i]
            appDeriv = appDeriv.T
            gradient.append(appDeriv)
            initDeriv = self.weights[i] @ initDeriv

        gradient = gradient[::-1]
        return gradient

    





agent = Network(4,[16,16],3)
# critic = Network(4,[16,16],1)

epochs = 100
learningRate = 0.1

for epoch in range(epochs):
    output = agent.traverse([100,100,100,100])
    error = agent.errorFunction(output,[[0,1,0]])
    gradient = agent.backPropogate()

    for i in range(len(gradient)):
        agent.weights[i] -= gradient[i]*learningRate

output = agent.traverse([100,100,100,100])

print(output)

# if __name__ == '__main__':
#   run()