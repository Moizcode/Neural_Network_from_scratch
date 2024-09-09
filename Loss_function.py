import numpy as np
class Loss:
    @staticmethod
    def MSE_Loss(predicted,actual):
        return np.mean((actual-predicted)**2)
    
    @staticmethod
    def MSE_derivative(y_pred,y_true):
        return 2*((y_pred-y_true)/y_true.size)