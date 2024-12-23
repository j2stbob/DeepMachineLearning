import numpy as np
import pandas as pd
# im really sorry if the comments are bad, I dont speak English well

class Neuron():
    def __init__(self, k, b):
        self.k = k
        self.b = b

    def output(self, x):
        out = 0
        for i in x:
            out = out + i * self.k + self.b
        return out


class Layer():                                  # Neurons layers
    def __init__(self, neuron_count):
        self.neurons = []
        self.neuron_count = neuron_count
        for i in range(neuron_count):
            self.neurons.append(Neuron(0, 0))

    def forward(self, inputs):                  # neuron for next layer
        outputs = []
        for i in range(self.neuron_count):
            out = self.neurons[i].output(inputs)
            outputs.append(out)
        return outputs


class InputLayer(Layer):                    # (support), Distribution of the first layer of neurons
    def forward(self, inputs):
        outputs = []
        for i in range(self.neuron_count):
            out = self.neurons[i].output([inputs[0][i]])
            outputs.append(out)
        return outputs


class Model():          # example: Model([Layer(1), Layer(2)])
    def __init__(self, layers):
        self.layers = layers


    def sigmoid(self, x):                  # function to convert ordinary values to similar probability
        return 1 / (1 + np.e ** -x[0])

    def predict(self, x):           # try to guess
        x = [x]
        for i in range(0, len(self.layers) - 1):
            if type(self.layers[i]) == InputLayer:
                x = self.layers[i].forward(x)
        
        return self.sigmoid(self.layers[-1].forward(x))

    def error(self):                # predict error
        sum_e = 0
        for i in range(0, len(x)):
            pred = self.predict(x[i])
            sum_e += abs(pred - y[i])
        return sum_e / len(x)

    def training(self, epochs, step):       # training model
        for i in range(epochs):
            for j in self.layers:
                all_neurons = j.neurons
                for neuron in all_neurons:
                    old_error = self.error()
                    neuron.k += step
                    if self.error() > old_error:                        # I just compared the old error and the new one,                                                               
                        neuron.k -= step * 2                                # than the new one, i just took *2 steps back                       

            print('Epoch:', i, 'Error:', old_error)



#just data
data = pd.read_csv('titanic.csv')
x1 = data['Fare'].fillna(data['Fare'].mean())
x2 = data['Pclass'].fillna(data['Pclass'].mean())                     # you can change the the values ere can be up to n x, the main thing is to add these values to the array here ---
x3 = data["Sex"] = data["Sex"].replace(
    {"female": 0,
     "male": 1}
)
x4 = data['Age'].fillna(data['Age'].mean())

y = data['Survived']

model = Model([
    InputLayer(4),              # idk how to write this, weeeeeell example: Inputlayer there should be as many first neurons as you indicated in X's 
    Layer(12),                                                                # Layer(4)   -----> 4 neurons in second layer
    Layer(1)                                                                       # Layer(1)   -----> end layer of neurons consisting of 1 neuron
])

x = []
for i in range(len(x1)):                            # distribution neurons after InputLayer
    x.append([x1[i], x2[i], x3[i], x4[i]])                                                                                           # <----- here
