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
        self.output = np.dot(input,self.weigth_matrix)+self.bias
        if self.activation_function == 'relu':
            self.output = np.maximum(0,self.output)
        return self.output

# layer1 = DenseLayer(2,3,'relu')
# layer2 = DenseLayer(3,4,'relu')
# layer3 = DenseLayer(4,2,'relu')
# layer4 = DenseLayer(2,1,'relu')

# ## forward_propagation
# input = np.array([[1,2],[2,4],[4,1],[1,3]])
# x = layer1.forward_pass(input)
# x = layer2.forward_pass(x)
# x = layer3.forward_pass(x)
# x = layer4.forward_pass(x)

# print(x)
# print(np.shape(x))