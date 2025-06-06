#!/usr/bin/python3
import math as m
import random
from decimal import Decimal
import copy
import time
import json

def activ(x):
    return float(1/(Decimal(1)+Decimal(m.e)**(Decimal(-x))))
def activPrime(x):
    return activ(x)*(1-activ(x))
def MSE(y,yExp):
    o=0
    for i in range(len(y)):
        o+=(y[i]-yExp[i])**2
    return o/len(y)
def vectorMatrixProd(vec,mat):
    o=[]
    for i in mat:
        sum=0
        if len(vec)!=len(i):
            raise Exception("Vector size mismatch")
        for j in range(len(i)):
            sum+=vec[j]*i[j]
        o.append(sum)
    return o
def vectorAdd(vec1,vec2):
        if len(vec1)!=len(vec2):
            raise Exception("Vector size mismatch")
        return [vec1[i]+vec2[i] for i in range(len(vec1))]
def vectorizeInt(x):
    o=[0,0,0,0,0,0,0,0,0,0]
    o[x]=1
    return o
def deserializeFile(name):
    o=[]
    with open(name) as f:
        content=f.read()
        f.close()
    for i in content.split("\n"):
        try:
            t=i.split(',')
            label=t[0]
            data=t[1:]
            o.append({'label':vectorizeInt(int(label)),'data':[int(j)/255 for j in data]})
        except:
            pass
    return o


def forwardPass(input,w1,b1,w2,b2,w3,b3):
    #layer 1
    a1=vectorAdd(vectorMatrixProd(input,w1),b1)
    a1old=copy.deepcopy(a1)
    #skibidi sigma rizzler frfr
    for i in range(len(a1)):
        a1[i]=activ(a1[i])
    #layer 2
    a2=vectorAdd(vectorMatrixProd(a1,w2),b2)
    a2old=copy.deepcopy(a2)
    #skibidi sigma rizzler frfr 2: electric boogaloo
    for i in range(len(a2)):
        a2[i]=activ(a2[i])
    a3=vectorAdd(vectorMatrixProd(a2,w3),b3)
    a3old=copy.deepcopy(a3)
    #skibidi sigma rizzler frfr 3: electric plane uwu
    for i in range(len(a3)):
        a3[i]=activ(a3[i])

    return [a3,a1old,a2old,a3old]
def stupidGrad(data,label,w1,b1,w2,b2,h=0.01):
    dx=0.0001
    w1copy=copy.deepcopy(w1)
    for i in range(len(w1)):
        for j in range(len(w1[i])):
            w1copy[i][j]+=dx
            w1[i][j]-=((MSE(forwardPass(data,w1copy,b1,w2,b2)[0],label)-MSE(forwardPass(data,w1,b1,w2,b2)[0],label))/dx)*h
            w1copy[i][j]-=dx
    print('w1 done')
    b1copy=copy.deepcopy(b1)
    for i in range(len(b1)):
        b1copy[i]+=dx
        b1[i]-=((MSE(forwardPass(data,w1,b1copy,w2,b2)[0],label)-MSE(forwardPass(data,w1,b1,w2,b2)[0],label))/dx)*h
        b1copy[i]-=dx
    
    w2copy=copy.deepcopy(w2)
    for i in range(len(w2)):
        for j in range(len(w2[i])):
            w2copy[i][j]+=dx
            w2[i][j]-=((MSE(forwardPass(data,w1,b1,w2copy,b2)[0],label)-MSE(forwardPass(data,w1,b1,w2,b2)[0],label))/dx)*h
            w2copy[i][j]-=dx

    b2copy=copy.deepcopy(b2)
    for i in range(len(b2)):
        b2copy[i]+=dx
        b2[i]-=((MSE(forwardPass(data,w1,b1,w2,b2copy)[0],label)-MSE(forwardPass(data,w1,b1,w2,b2)[0],label))/dx)*h
        b2copy[i]-=dx

    return [w1,b1,w2,b2]
def backProp(data,label,w1,b1,w2,b2,w3,b3,h=0.1):
    fw=forwardPass(data,w1,b1,w2,b2,w3,b3)
    out=fw[0]
    z1=fw[1]
    z2=fw[2]
    z3=fw[3]
    local3=[]
    for i in range(len(w3)):
        localGrad=(out[i]-label[i])*activPrime(z3[i])#*out[i]*(1-out[i])
        local3.append(localGrad)
        for j in range(len(w3[i])):
            delta=localGrad*activ(z2[j])
            w3[i][j]-=delta*h
    
    for i in range(len(b3)):
        b3[i]-=local3[i]*h

    local2=[]
    for i in range(len(w2)):
        sum=0
        for j in range(len(local3)):
            sum+=local3[j]*w3[j][i]
        localGrad=sum*activPrime(z2[i])
        local2.append(localGrad)
        for j in range(len(w2[i])):
            delta=localGrad*z1[j]
            w2[i][j]-=delta*h

    for i in range(len(b2)):
        sum=0
        for j in range(len(local3)):
            sum+=local3[j]*w3[j][i]
        delta=sum*activPrime(z2[i])
        b2[i]-=delta*h
    
    for i in range(len(w1)):
        sum=0
        for j in range(len(local2)):
            sum+=local2[j]*w2[j][i]
        localGrad=sum*activPrime(z1[i])
        for j in range(len(w1[i])):
            delta=localGrad*data[j]
            w1[i][j]-=delta*h
    
    for i in range(len(b1)):
        sum=0
        for j in range(len(local2)):
            sum+=local2[j]*w2[j][i]
        localGrad=sum*activPrime(z1[i])
        b1[i]-=delta*h

    return [w1,b1,w2,b2,w3,b3]

#inicialize values
w1=[]
for i in range(16):
    t=[]
    for j in range(784):
        t.append(random.uniform(-0.5,0.5))
    w1.append(t)
b1=[0 for i in range(16)]

w2=[]
for i in range(16):
    t=[]
    for j in range(16):
        t.append(random.uniform(-0.5,0.5))
    w2.append(t)
b2=[0 for i in range(16)]

w3=[]
for i in range(10):
    t=[]
    for j in range(16):
        t.append(random.uniform(-0.5,0.5))
    w3.append(t)
b3=[0 for i in range(10)]

count=0

t1=time.time()
for i in deserializeFile('MNIST_train.txt'):
    if count%100==0:
        print(count, MSE(forwardPass(i['data'],w1,b1,w2,b2,w3,b3)[0],i['label']),time.time()-t1)
        t1=time.time()

    o=backProp(i['data'],i['label'],w1,b1,w2,b2,w3,b3,0.05)
    w1=o[0]
    b1=o[1]
    w2=o[2]
    b2=o[3]
    count+=1

    #print()
correct=0
print('-----')
test=deserializeFile('MNIST_test.txt')
for i in test:
    dw=forwardPass(i['data'],w1,b1,w2,b2,w3,b3)[0]
    if i['label'].index(max(i['label']))==dw.index(max(dw)):
        correct+=1
print('correct:',correct,'; test size: ',len(test),'; success rate: ',correct/len(test))
params=json.dumps({'w1':w1,'b1':b1,'w2':w2,'b2':b2,'w3':w3,'b3':b3})
with open("parameters.json","w") as f:
    f.write(params)
    f.close()
'''
w1=[[random.uniform(-0.5,0.5),random.uniform(-0.5,0.5),random.uniform(-0.5,0.5)],[random.uniform(-0.5,0.5),random.uniform(-0.5,0.5),random.uniform(-0.5,0.5)],[random.uniform(-0.5,0.5),random.uniform(-0.5,0.5),random.uniform(-0.5,0.5)]]
w2=[[random.uniform(-0.5,0.5),random.uniform(-0.5,0.5),random.uniform(-0.5,0.5)],[random.uniform(-0.5,0.5),random.uniform(-0.5,0.5),random.uniform(-0.5,0.5)]]
w3=[[random.uniform(-0.5,0.5),random.uniform(-0.5,0.5)]]
b1=[0,0,0]
b2=[0,0]
b3=[0]

w1old=copy.deepcopy(w1)
w2old=copy.deepcopy(w2)

def xor(a,b,c):
    if a+b+c==1:
        return 1
    else:
        return 0

possible=[
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,1,0],
    [1,1,1]
]

for i in range((30000)):
    t1=time.time()
    data=random.choice(possible)
    #if random.randint(0,80)==2:
    #    data=[0,0,0,0]
    #print(i, MSE(forwardPass(data,w1,b1,w2,b2)[0],[xor(data[0],data[1])]))

    #o=stupidGrad(data,[xor(data[0],data[1])],w1,b1,w2,b2,0.3)
    o=backProp(data,[xor(data[0],data[1],data[2])],w1,b1,w2,b2,w3,b3,0.2)

    w1=o[0]
    b1=o[1]
    w2=o[2]
    b2=o[3]
    #print(time.time()-t1)
    if i%100==0:
        print(i)
correct=0
print('\n\n')

print([0,0,0],round(forwardPass([0,0,0],w1,b1,w2,b2,w3,b3)[0][0]),forwardPass([0,0,0],w1,b1,w2,b2,w3,b3)[0])
print([0,0,1],round(forwardPass([0,0,1],w1,b1,w2,b2,w3,b3)[0][0]),forwardPass([0,0,1],w1,b1,w2,b2,w3,b3)[0])
print([0,1,0],round(forwardPass([0,1,0],w1,b1,w2,b2,w3,b3)[0][0]),forwardPass([0,1,0],w1,b1,w2,b2,w3,b3)[0])
print([0,1,1],round(forwardPass([0,1,1],w1,b1,w2,b2,w3,b3)[0][0]),forwardPass([0,1,1],w1,b1,w2,b2,w3,b3)[0])
print([1,0,0],round(forwardPass([1,0,0],w1,b1,w2,b2,w3,b3)[0][0]),forwardPass([1,0,0],w1,b1,w2,b2,w3,b3)[0])
print([1,0,1],round(forwardPass([1,0,1],w1,b1,w2,b2,w3,b3)[0][0]),forwardPass([1,0,1],w1,b1,w2,b2,w3,b3)[0])
print([1,1,0],round(forwardPass([1,1,0],w1,b1,w2,b2,w3,b3)[0][0]),forwardPass([1,1,0],w1,b1,w2,b2,w3,b3)[0])
print([1,1,1],round(forwardPass([1,1,1],w1,b1,w2,b2,w3,b3)[0][0]),forwardPass([1,1,1],w1,b1,w2,b2,w3,b3)[0])
print(possible)
'''
'''
correct=0
for i in range(100):
    data=[random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1)]
    o=forwardPass(data,w1,b1,w2,b2)[0]
    #print(data, round(o[0]))
    if xor(data[0],data[1])==round(o[0]):
        correct+=1
print(correct)
'''
