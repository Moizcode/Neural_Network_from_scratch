import numpy as np
class DenseLayer:
    def __init__(self,inputs,units,activation_function = 'relu') -> None:
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
        return dl_da
    
def calculate_mseloss(predicted,actual):
    return np.mean((predicted-actual)**2)

def mseloss_derivative(y_pred,y_true):
    return 2*((y_pred-y_true)/y_true.size)

def forward_pass_calling(layers_list,x):
    for layers in layers_list:
        x = layers.forward_pass(x)
    return x

def backward_pass_calling(layers_list,loss_derivative,learning_rate = 0.2):
    for layers in layers_list[::-1]:
        loss_derivative = layers.backward_pass(loss_derivative,learning_rate)



# dummy data
input_size= 150
input_variables = 3
x1 = np.random.rand(input_size)
x2 = np.random.rand(input_size)
x3 = np.random.rand(input_size)

y_true = 5*x1 + 2.4*x2 + 7.897*x3 + np.random.rand(input_size)
y_true = y_true.reshape(input_size,1)

print("shape of y_true",np.shape(y_true))
input_X = np.dstack((x1,x2,x3))
input_X = input_X.reshape(input_size,input_variables)

# create layers
layer1 = DenseLayer(3,6,'relu')
layer2 = DenseLayer(6,4,'relu')
layer3 = DenseLayer(4,2,'relu')
layer4 = DenseLayer(2,1,'n')

layers_list = [layer1,layer2,layer3,layer4]

def train_function(epochs,layers_list,input_X,y_true):
    for i in range(epochs):
        y_pred = forward_pass_calling(layers_list,input_X)
        loss = calculate_mseloss(y_pred,y_true)
        print("loss at epoch",i,"is",loss)
        e = mseloss_derivative(y_pred,y_true)
        backward_pass_calling(layers_list,e,0.2)



train_function(10,layers_list,input_X,y_true)

# ## forward_propagation
# x = layer1.forward_pass(input_X)
# x = layer2.forward_pass(x)
# x = layer3.forward_pass(x)
# y_pred = layer4.forward_pass(x)


# mse_loss = calculate_mseloss(y_pred,y_true)
# dl_dmse = mseloss_derivative(y_pred,y_true)

# print("\n now backward pass start \n")
# e = layer4.backward_pass(dl_dmse,0.2)
# e = layer3.backward_pass(e,0.2)
# e = layer2.backward_pass(e,0.2)
# e = layer1.backward_pass(e,0.2)

# # print(e)
# print(np.shape(e))
# print(np.shape(x))