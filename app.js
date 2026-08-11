let pendulum;
let g = 0.005;
let damp = -0.001;
let f = 0;
let eval = 0

function setup(){
    createCanvas(window.innerWidth,window.innerHeight)
    pendulum = new Pendulum(width/2,height/1.75,200,0)
}

function draw(){
    background(0)
    strokeWeight(1)
    stroke(0,100,150)
    line(0,height/1.75,width,height/1.75)
    eval = evaluate()
    fill(255)
    textSize(22)
    textFont('Courier')
    text(eval.toFixed(3),width - 200,100)
    pendulum.show()
    if(keyIsDown(RIGHT_ARROW)){
        f = 1
    }
    else if( keyIsDown(LEFT_ARROW)){
        f = -1
    }
    else{
        f = 0;
    }
    pendulum.update()
}

class Pendulum{
    constructor(x,y,l,a){
        this.pos = createVector(x,y)
        this.vel = 0
        this.acc = 0
        this.ang = a
        this.angVel = 0
        this.angAcc = 0
        this.l = l
        this.m = 500;
    }
    show(){
        let x2 = this.pos.x + this.l * cos(this.ang)
        let y2 = this.pos.y + this.l * sin(this.ang)
        strokeWeight(4)
        stroke(map(eval,-10,10,255,0),map(eval,-10,10,0,255),20)
        line(this.pos.x,this.pos.y,x2,y2)
        noStroke()
        fill(map(eval,-10,10,255,0),map(eval,-10,10,0,255),20)
        circle(x2,y2,this.l/4)
        strokeWeight(2)
        stroke(map(abs(this.vel),0,25,100,250))
        noFill()
        circle(this.pos.x,this.pos.y,this.l/7)
        circle(this.pos.x,this.pos.y,5)
        
    }
    update(){
        this.acc = f + g*abs(sin(this.ang))*(abs(cos(this.ang))/cos(this.ang))
        this.vel += this.acc
        this.pos.x += this.vel
        this.angAcc = g*cos(this.ang) + (f/this.m)*sin(this.ang)
        this.angVel += this.angAcc
        this.angVel += damp*this.angVel
        this.ang += this.angVel
    }
}

function evaluate(){
    let score = 0;
    let scaling = 10
    score += scaling*-1*sin(pendulum.ang)
    let offset = -1 * scaling * abs(pendulum.pos.x - width/2)/(width/2)
    score += offset
    score -= abs(pendulum.angVel)*scaling*scaling*scaling/6
    score -= (abs(pendulum.vel)/50)*scaling 
    return score
}