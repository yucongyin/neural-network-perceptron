import numpy
import pandas
import random

data = pandas.read_excel("data1.xls", header = None)

X = pandas.DataFrame.to_numpy(data)

Xc = numpy.zeros((1000,1))
Xc = numpy.concatenate((Xc,numpy.ones((1000,1))),axis = 0)

random.seed()
w = numpy.transpose(numpy.array([random.random()-0.5, random.random()-0.5, random.random()-0.5]))

X = numpy.concatenate((numpy.ones((2000,1)),X),axis = 1)
x1 = X[0:1000,:]
x2 = X[1000:2000,:]

totalErrorPrev = 0
for epoch in range(1000):
    totalError = 0
    for i in range(2000):
        z = (numpy.sign(numpy.dot(X[i,:],w)) + 1)/2   
        error  = Xc[i] - z
        totalError = totalError + abs(error)
        w = w + 0.1*error*X[i,:]
        
    #if abs(totalError - totalErrorPrev) < 1 :
    #        break
    #totalErrorPrev = totalError
        
print("Slope is ", -w[1] / w[2], "\n")
print("Y Intercept is", -w[0] / w[2])

print("done")    
