import numpy as np

inp =np.ones(shape=(5,1,2))  #(batch_size,lastneuro_size,currentunits)
glorot_x1 = np.sqrt(6/(2+3))
weight = np.random.uniform(low=-glorot_x1,high=glorot_x1,size=(2,3))
bais = np.ones(shape=(1,3))*0.5
outp = np.dot(inp,weight) + bais
outp = np.maximum(0,outp)

print(inp)
print(weight)
print(outp)
print(np.shape(outp))


print("\n next layer \n")
unit2 = 4
neuron_in_lst = np.shape(outp)[-1]
glorot_x2 = np.sqrt(6/(neuron_in_lst+unit2))
weight = np.random.uniform(low=-glorot_x2,high=glorot_x2,size=(neuron_in_lst,unit2))
weight2 = np.ones(shape=(neuron_in_lst,unit2))*10
output2 = np.dot(outp,weight2)
output2 = np.maximum(0,output2)
print(output2)
print(np.shape(output2))