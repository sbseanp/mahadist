'''
Created on May 26, 2014

@author: Sean
'''

from sys import argv
import math
import numpy as np

'''find covariance matrix and output as array'''        
def covariance(input, centroid):
    dim = len(centroid)
    centroid_array = centroid
    mat_array = np.zeros(shape=(dim,dim))
    '''used formula SUM(K*K^T)/N'''
    for i in range(len(input)):
        input_array = np.asarray(input[i])
        diff_array = np.subtract(input_array, centroid_array)
        cov = np.outer(diff_array, diff_array.T)
        mat_array = np.add(mat_array, cov)
    mat_array = mat_array/len(input)
    return mat_array  

'''returns covariance matrix'''
def train_classifier(filename):
    train_file = open(filename)
    row, col = map(int, train_file.readline().split())
    '''get centroid and use it to find covariance matrix'''
    centroid = [0 for _ in range(col)]
    L = []
    for line in train_file:
        x = list(map(float, line.split()))
        centroid = np.add(centroid, x)
        L.append(x)
    train_file.close()
    centroid = np.asarray(centroid)
    centroid = centroid/row
    matrix = covariance(L, centroid)
    '''print some stuff with sample formatting''' 
    print('Centroid: ',end=" ")
    print(centroid)
    print('Covariance Matrix:')
    print(matrix)
    print('Distances:')   
    return matrix

'''dedicated function for finding centroid'''
def find_centroid(filename):
    train_file = open(filename)
    row, col = map(int, train_file.readline().split())
    centroid = [0 for _ in range(col)]
    for line in train_file:
        x = list(map(float, line.split()))
        centroid = np.add(centroid, x)    
    train_file.close()
    centroid = np.asarray(centroid)
    return centroid/row

def test_classifier(filename, covar):
    test_file = open(filename)
    row, col = map(int, test_file.readline().split())
    centroid = find_centroid(argv[1])
    count = 1
    '''go line by line and calculate Mahalanobis distance'''
    for line in test_file:
        x = list(map(float, line.split()))
        print(count, end='')
        count = count + 1
        print('.',end=' ')
        print(np.asarray(x),end=' ')
        print('-- ',end='')
        x = np.asarray(x)
        '''get difference of input and centroid'''
        diff = x - centroid
        '''invert the covariance matrix'''
        inverse = np.linalg.inv(covar)
        '''take the first dot product'''
        first = np.dot(diff.T, inverse)
        '''take the second dot product'''
        second = np.dot(first, diff)
        '''take square root for result'''
        print(math.sqrt(second))
    test_file.close()
          
if __name__ == '__main__':
    if len(argv) == 3:
        temp = train_classifier(argv[1])
        test_classifier(argv[2], temp)
    else: 
        print('Please input the training file and the test file')
