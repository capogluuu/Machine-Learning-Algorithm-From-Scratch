import numpy as np
import pandas as pd
import random


class LinearRegression():
    print("LinearRegression Class")
    def __init__(self, learning_rate, epoch):
        self.learning_rate = learning_rate
        self.epoch         = epoch
        self.weight        = 0
        self.bias          = 0
    def training(self, X , y):
        lost_skor = []
        for _ in range(self.epoch):
            y_hat = np.dot(X.T,self.weight)+self.bias
            print(f" weight= {self.weight} bias= {self.bias} mse= {np.mean((y_hat-y)**2)}")
            # şekli a= [[1,2,2,3,4,5]] gibi
            MSE =  np.mean((y_hat-y)**2)
            lost_skor.append(MSE)
            for __ in range(10):
                weight_derive =  np.mean(np.dot(X,((np.dot(X.T,self.weight)+self.bias)-y).T))
                bias_derive   =  np.mean((np.dot(X.T,self.weight)+self.bias)-y)
                self.weight   =  self.weight - self.learning_rate*weight_derive
                self.bias     =  self.bias   - self.learning_rate*bias_derive
            
        #print(lost_skor)
        
    def predict(self, X):
        resultat = np.dot(X,self.weight)+self.bias
        print(type(resultat))
        return resultat

def main():
    X = np.array([[1,2,3,4,5,6,7,8,9,10]])
    y = np.array([11,21,31,41,51,61,71,81,91,100])
    model = LinearRegression(learning_rate = 0.00001, epoch =80)
    model.training(X,y)
    X1 = np.array([8,9,10]),
    X2 = np.array([81,91,101])
    a = model.predict(X1)
    #print(X1.shape, a.shape)
    out_arr = a-X2
    print("a = ", out_arr)
    
if __name__ == "__main__" : 
    main()