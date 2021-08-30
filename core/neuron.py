# class for create neuron
# libs
import numpy as np

class Neuron:
    # constructor
    def __init__(self, learn_rate, *weigths):
        # learn rate
        self.learn_rate = learn_rate

        # weights
        self.weights = np.array(weigths)

        # backpropagation error size array
        self.back_propagation_error_sizes = np.zeros(len(weigths))
        
    # Get sigmoid number
    def Sigmoid(self, summator):
        return 1 / (1 + np.exp(-summator))

    # Method for get delta number from sigmoid
    def Delta_From_Sigmoid(self, sigmoid):
        return sigmoid * (1 - sigmoid)

    # Create prediction
    def Predict(self, inputs_data_array):
        # inputed data
        self.inputs_data_array = inputs_data_array

        # summator number
        summator = np.dot(inputs_data_array, self.weights)

        # neuron results
        self.result = self.Sigmoid(summator)
        return self.result

    # Method return 0 (less 0.5) or 1 (0.5 and more) 
    def Activator(self, data_array):
        return 1 if self.Predict(data_array) >= 0.5 else 0

    # Method for calibrate neurons weights.
    def Calibrate_Weights(self, correct_result):
        # error size
        error_size = self.result - correct_result

        # size for change weights
        weigth_change_size = error_size * self.Delta_From_Sigmoid(self.result)

        # change sizes for weight and add information for backpropagation method
        for number in range(len(self.weights)):
            self.weights[number] = self.weights[number] - self.inputs_data_array[number] * weigth_change_size * self.learn_rate
            self.back_propagation_error_sizes[number] = self.weights[number] * weigth_change_size

    # Method for calibrate weight usig backpropagation method
    def Back_Propagation_Calibrate_Weigth(self, backpropagation_error_size):
        
        # size for change weight
        weigth_change_size = backpropagation_error_size * self.Delta_From_Sigmoid(self.result)

        # change sizes for weight and add information for backpropagation method
        for number in range(len(self.weights)):
            self.weights[number] = self.weights[number] - self.inputs_data_array[number] * weigth_change_size * self.learn_rate
            self.back_propagation_error_sizes[number] = self.weights[number] * weigth_change_size
        

