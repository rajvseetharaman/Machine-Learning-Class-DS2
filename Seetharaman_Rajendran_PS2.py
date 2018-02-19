

#import numpy and pandas libraries
import numpy as np
import pandas as pd


#initialize alpha
alpha=0.85
#initialize epsilon
epsilon=0.00001
#initialize the sample journal citation file
z=np.array([[1,0,2,0,4,3],[3,0,1,1,0,0],[2,0,4,0,1,0],[0,0,1,0,0,1],[8,0,3,0,5,2],[0,0,0,0,0,0]])
#initialize sample article file
article=[3,2,5,1,2,1]


#function to compute influence vector pi*
def compute_pi(z,alpha,epsilon,art):
    #get the shape of input matrix
    x,y = np.shape(z)
    z1=z
    #set matrix diagonal to zero
    for i in range(x):
        for j in range(y):
            if i==j:
                z1[i][j]=0
    #find sum of the individual columns of matrix            
    colsum=z1.sum(axis=0)
    #normalize the matrix
    z2=z1/z1.sum(axis=0) 
    # set the results of 0/0 computations to 0
    h=np.matrix(np.nan_to_num(z2))
    #create vector of dangling nodes
    d=[int(i==0) for i in colsum]
    d=np.matrix(d)
    article=art
    #normalize and transpose the article vector
    a=article/np.sum(article)
    a=np.matrix(a).transpose()
    #compute initial influence vector
    pi=([1/x]*x)
    pi=np.matrix(pi).transpose()
    #set variable to count number of loop iterations
    countiter=0
    while True:
        #set old value of pi
        pi_old=pi
        #compute new pi value
        pi=alpha*np.dot(h,pi) + float((alpha*np.dot(d,pi))+(1-alpha))*a
        #find the residual value
        residual=pi-pi_old
        res_absolute_sum=np.sum(np.absolute(residual))
        #count numbe rof iterations
        countiter+=1
        #break loop if residual value less than epsilon
        if res_absolute_sum<epsilon:
            break
    print(countiter,' iterations')
    #return H and pi*
    return (h,pi)



#function to compute eigenfactor values using H and pi*
def compute_eigenfactor(h,pi):
    #compute eigenfactors using H and pi*
    ef1=np.dot(h,pi)
    #normalize values and convert to percentage
    ef2=(ef1/np.sum(ef1))*100
    return ef2



#compute eigenfactors for the test set
h,pi=compute_pi(z,alpha,epsilon,article)
ef2=compute_eigenfactor(h,pi)
#print eigenfactor values
print('Eigenfactors for test data:')
print(ef2)



#Read the journal citation file into a dataframe
citations= pd.read_csv("links.txt",header=None,names=['node1','node2','citation_count'], )
#create lists pertaining to node1,node2,citation count values
n1 = citations.node1.values
n2 = citations.node2.values
c_val = citations.citation_count.values

#create blank adjacency matrix of 0's for journal citations data
adjMat = np.zeros([10747,10747],dtype=np.int16)
#initialize adjacency matrix with citation values for each combination of journal nodes in the text file
for i in range(len(n1)):
    adjMat[n1[i]-1,n2[i]-1] = c_val[i]

#create the article matrix corresponding to our data
article=np.matrix([1]*10747)

import time
start_time=time.time()
#compute eigenfactor values for data
h1,pi1=compute_pi(adjMat,alpha,epsilon,article)
ef3=compute_eigenfactor(h1,pi1)
end_time=time.time()
print(end_time-start_time,' seconds taken for code to run on network')


#compute eigenfactor scores and print the number of iterations taken to run algorithm
list1, list2 = (list(t) for t in zip(*sorted(zip(ef3,range(1,len(ef3)+1)),reverse=True)))
#ef4=sorted(ef3,reverse=True)
print('Top 20 Journals and their corresponding scores')
for x,y in zip(list2[:20],list1[:20]):
    print('Node: ',x, 'Score: ',float(y))

'''
a) 
Top 20 Journals and their corresponding scores:
Node:  8930 Score:  1.1084058959847904
Node:  725 Score:  0.24740366221656335
Node:  239 Score:  0.24382597005945472
Node:  6523 Score:  0.2351794932102711
Node:  6569 Score:  0.22609342705372548
Node:  6697 Score:  0.22526193056645713
Node:  6667 Score:  0.21670235911629118
Node:  4408 Score:  0.20647553543821487
Node:  1994 Score:  0.2014407503239001
Node:  2992 Score:  0.1850407499240807
Node:  5966 Score:  0.1827520446254698
Node:  6179 Score:  0.1807563007016855
Node:  1922 Score:  0.17507725859917256
Node:  7580 Score:  0.17044831864568102
Node:  900 Score:  0.17020871319564979
Node:  1559 Score:  0.1680011184242936
Node:  1383 Score:  0.16356508602043948
Node:  1223 Score:  0.1507420890364194
Node:  422 Score:  0.14937222240246908
Node:  5002 Score:  0.1490070500884894

b) The time taken to run this code on the real network was 10.41 seconds.

c) It took 34 iterations to get the answer.
'''

