import numpy as np
class DenseLayer:
    def __init__(self,inputs,units,activation_function = 'NoActivation') -> None:
        self.units = units
        self.activation_function = activation_function
        # for now only uniform-glorot is using which is default in keras
        self.glorot_x = np.sqrt(6/(inputs+units))
        self.weigth_matrix = np.random.uniform(low=-self.glorot_x,high=self.glorot_x,size=(inputs,units))
        self.bias = np.zeros((1,units))

    def forward_pass(self,input):
        self.input = input
        self.z = np.dot(self.input,self.weigth_matrix)+self.bias
        if self.activation_function == 'relu':
            self.a = np.maximum(0,self.z) 
            return self.a
        if self.activation_function == 'tanh':
            self.a = np.tanh(self.z)
            return self.a
        if self.activation_function == 'sigmoid':
            self.a = 1/(1+np.exp(-self.z))
            return self.a
        # if self.activation_function == 'softmax':
        #     self.a = np.exp(self.z)/np.sum(np.exp(self.z))
        #     print(np.sum(self.a))
        return self.z
    
    def backward_pass(self,dl_da,learning_rate):
        # derivation of activation loss
        dl_dz = self.activation_loss(dl_da)


        # deravative of weigth and bais loss
        dl_dw = np.dot(self.input.T,dl_dz) # Shape: (input_size, output_size)
        dl_db = np.sum(dl_dz,axis=0) # Shape: (output_size,)

        # derivative for the nxt layer
        dl_dx = np.dot(dl_dz, self.weigth_matrix.T) #Shape: (batch_size,input_size)

        #weight and bais updates
        self.weigth_matrix -= learning_rate * dl_dw
        self.bias -= learning_rate * dl_db

        return dl_dx
    
    def activation_loss(self,dl_da):
        if self.activation_function == 'relu':
            return dl_da * np.where(self.z>0,1,0)
        
        if self.activation_function == 'tanh':
            return dl_da * (1- np.square(self.a))
        
        if self.activation_function == 'sigmoid':
            return dl_da* self.a *(1-self.a)
        
        # if self.activation_function == 'softmax':
        #     return dl_da - self.a
        return dl_da
    