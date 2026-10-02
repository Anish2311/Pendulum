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
#         NUERAL NETWOR PART
#----------------------------------------



inputLayer = np.array([[100,200,300,400]])

layerOneWeights = np.abs(np.random.normal(0,0.1,64))
layerOneWeights = np.reshape(layerOneWeights,(4,16))
layerOneBias = np.zeros(16)

layerTwoWeights = np.abs(np.random.normal(0,0.1,16*16))
layerTwoWeights = np.reshape(layerTwoWeights,(16,16))
layerTwoBias = np.zeros(16)

layerThreeWeights = np.abs(np.random.normal(0,0.1,16*3))
layerThreeWeights = np.reshape(layerThreeWeights,(16,3))
layerThreeBias = np.zeros(3)

def activationFunction(a):
    return np.sign(a)

def outputFunction(a):
    return a/a.sum()

def traverse():
    oLayer1 = inputLayer @ layerOneWeights
    oLayer1 = oLayer1 + layerOneBias
    oLayer1 = activationFunction(oLayer1)

    olayer2 = oLayer1 @ layerTwoWeights
    olayer2 = olayer2 + layerTwoBias
    olayer2 = activationFunction(olayer2)


    outputLayer = olayer2 @ layerThreeWeights
    outputLayer = outputLayer + layerThreeBias
    outputLayer = outputFunction(outputLayer)

    return outputLayer

output = traverse()

if __name__ == '__main__':
  run()