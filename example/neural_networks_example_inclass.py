import numpy
import math
import random
import matplotlib.pyplot

# create data from a multivariate normal distribution
mean1 = [-3,0]
mean2 = [3,0]

cov = [[2,0],[0,3]]

x1 = numpy.random.multivariate_normal(mean1, cov, 1000)
matplotlib.pyplot.scatter(x1[:,0], x1[:,1], c = 'b', marker = '.')
x2 = numpy.random.multivariate_normal(mean2, cov, 1000)
matplotlib.pyplot.scatter(x2[:,0], x2[:,1], c = 'r', marker = '.')
matplotlib.pyplot.ion()
matplotlib.pyplot.show()

X = numpy.concatenate((x1,x2))
X = numpy.concatenate((numpy.ones((2000,1)),X), axis = 1)

#first half of the data from class 0, second half from class 1
Xc = numpy.zeros(1000)
Xc = numpy.concatenate((Xc, numpy.ones(1000)))


#randomly initialize the weights
random.seed()
w1 = numpy.transpose(numpy.array([random.random()-0.5, random.random()-0.5, random.random()-0.5]))
w2 = numpy.transpose(numpy.array([random.random()-0.5, random.random()-0.5, random.random()-0.5]))
w3 = numpy.transpose(numpy.array([random.random()-0.5, random.random()-0.5, random.random()-0.5]))

alpha = 0.25
outputError = numpy.empty((2000,1))
for epoch in range(2000):
    totalError = 0
    for i in range(2000):
        #forward calculation
        z1 = 1/(1 + math.exp(-numpy.dot(X[i,:],w1)))
        z2 = 1/(1 + math.exp(-numpy.dot(X[i,:],w2)))
        xhidden = [1, z1, z2]
        z3 = 1/(1 + math.exp(-numpy.dot(xhidden,w3)))

        # prediction
        prediction = round(z3,0) #one option...
        outputError[i] = abs(Xc[i] - prediction)
        totalError = totalError + numpy.asscalar(outputError[i])

        #backward propagation
        #update the error
        error3 = z3*(1-z3)*(Xc[i] - z3) #output node 
        error1 = z1*(1-z1)*(error3*w3[1]) #hidden layer node
        error2 = z2*(1-z2)*(error3*w3[2]) #hidden layer

        w3 = w3 + [alpha*error3, alpha*error3*z1, alpha*error3*z2 ]
        w1 = w1 + [alpha*error1, alpha*error1*X[i,1], alpha*error1*X[i,2]]
        w2 = w2 + [alpha*error2, alpha*error2*X[i,1], alpha*error2*X[i,2]]

    # for every epoch
    print("Iteration... ", epoch + 1, "Error = ", totalError, "            ", end = '\r', flush = True)
    



print("done")


