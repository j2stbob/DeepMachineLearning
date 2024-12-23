import numpy as np
import pandas as pd
#im really sorry if the comments are bad, I dont speak English well



class Neuron():
    def __init__(self, k, b):
        self.k = k
        self.b = b

    def output(self, x):
        out = 0
        for i in x:
            out = out + i * self.k + self.b
        return out


class Layer():                                  #Neurons layers
    def __init__(self, neuron_count):
        self.neurons = []
        self.neuron_count = neuron_count
        for i in range(neuron_count):
            self.neurons.append(Neuron(0, 0))

    def forward(self, inputs):                  #neuron for next layer
        outputs = []
        for i in range(self.neuron_count):
            out = self.neurons[i].output(inputs)
            outputs.append(out)
        return outputs


class InputLayer(Layer):                    #(support), Distribution of the first layer of neurons
    def forward(self, inputs):
        outputs = []
        for i in range(self.neuron_count):
            out = self.neurons[i].output([inputs[0][i]])
            outputs.append(out)
        return outputs


class Model():          #example: Model([Layer(1), Layer(2)])
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
                    if self.error() > old_error:                        # I just compared the old error and the new one,                                                                       #and if the old one is bigger
                        neuron.k -= step * 2                                #than the new one, i just took *2 steps back                       neuron.w -= step * 2

            print('Epoch:', i, 'Error:', old_error)



#just data
data = pd.read_csv('train_data.csv')
x1 = data['Fare'].fillna(data['Fare'].mean())*1000 # you can remove this I did this for your understanding in predict(model will couting faster if you femove it)
x2 = data['Pclass_1']
x3 = data['Pclass_2']
x4 = data['Pclass_3']
x5 = data["Sex"]
x6 = data['Age'].fillna(data['Age'].mean())*100
x7 = data['Family_size']

y = data['Survived']


# you can add more layers
model = Model([
    InputLayer(7),              # idk how to write this, weeeeeell example: Layer(2) ----> 2 neurons in first layer
    Layer(14),                                                                #Layer(4)   -----> 4 neurons in second layer
    Layer(1)                                                                       #Layer(1)   -----> end layer of neurons consisting of 1 neuron
])

x = []
for i in range(len(x1)):                            #distribution neurons after InputLayer
    x.append([x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i]])


#-----------------------------------------------------------------------------------------------------------------------

def main():
    model.training(50, 0.01)
    print((model.predict([150, 0, 0, 1, 1, 27.5, 0.1])))
                    #Fare, Pclass_1, Pclass_2, Pclass_3, Sex, Age, Family size
                            #Select one Pclass you need and put 1 in the rest put 0



if __name__ == "__main__":
    main()

