import numpy
import matplotlib.pyplot as plt
import random
import pandas 

#setup constants
C = 0.001
COVARIANCE = 1
CLASSONE_VAR_1 = 3
CLASSONE_VAR_2 = 1
CLASSTWO_VAR_1 = 1
CLASSTWO_VAR_2 = 2
MEAN1 = [-2,0]
MEAN2 = [2,0]
NUM_DATAPOINTS = 1000
#create two classes of data


#cov is 1, variance is 3,1
cov1 = [[CLASSONE_VAR_1,COVARIANCE],[COVARIANCE,CLASSONE_VAR_2]]
#cov is 1, variance is 1,2
cov2 = [[CLASSTWO_VAR_1,COVARIANCE],[COVARIANCE,CLASSTWO_VAR_2]]


#each should have 1000 data points
x1 = numpy.random.multivariate_normal(MEAN1,cov1,NUM_DATAPOINTS)
x2 = numpy.random.multivariate_normal(MEAN2,cov2,NUM_DATAPOINTS)


#plotting to see the data for fun
# plt.scatter(x1[:,0],x1[:,1],c='b', marker='.')
# plt.scatter(x2[:,0],x2[:,1],c='r', marker='.')
# plt.show()

#combining the two data clouds
X = numpy.concatenate((x1,x2),axis=0)
#adding ones as bias
X = numpy.concatenate((numpy.ones((2000,1)),X),axis=1)

#trying to understand how numpy.concatenate works
# df = pandas.DataFrame(X)
# df.to_excel("example/data2.xlsx")

#adding labels
Xc = numpy.zeros((1000,1))
Xc = numpy.concatenate((Xc,numpy.ones((1000,1))),axis=0)


#initialize weight
random.seed()
w = numpy.transpose(numpy.array([random.random()-0.5,random.random()-0.5,random.random()-0.5]))
errorList = []

#implementing the algorithm
#perform a maximum of 100 iterations
for epoch in range(100):
    totalError = 0
    
    for i in range(2000):
        #using this calculation to retrieve an output of 1 or 0
        z = (numpy.sign(numpy.dot(X[i,:],w))+1)/2
        #error = t-z 
        error = Xc[i] - z
        #calculating delta z, which is c * error * Xi
        deltaZ = C * error * X[i,:]
        w += deltaZ

        #calculate total errors
        totalError = totalError + abs(error)
    #print(totalError)    
    errorList.append(totalError)



plt.plot(errorList,'k')
plt.xlabel("iterations(epoch)")
plt.ylabel("total errors")
plt.show()






