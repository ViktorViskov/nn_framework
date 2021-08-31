# libs
import numpy as np
from core.neuron import Neuron

# class for neuron network
class NN:
    # constructor4
    def __init__(self, learn_rate):

        # Creating neurons

        # layer 1
        self.hidden1 = Neuron(learn_rate, np.random.random(),np.random.random(),np.random.random())
        self.hidden2 = Neuron(learn_rate, np.random.random(),np.random.random(),np.random.random())

        # output
        self.output1 = Neuron(learn_rate,np.random.random(),np.random.random())
        self.output2 = Neuron(learn_rate,np.random.random(),np.random.random())

    # Method for prediction
    def Predict(self, data):

        # layer 1
        r_l1_h1 = self.hidden1.Predict(data)
        r_l1_h2 = self.hidden2.Predict(data)

        # output
        r_o1 = self.output1.Predict(np.array([r_l1_h1, r_l1_h2]))
        r_o2 = self.output2.Predict(np.array([r_l1_h1, r_l1_h2]))

        # output
        return np.array([r_o1,r_o2])
    
    # Method for one time training
    def Train(self, data, answer, epoch):
        # prediction
        output_result = self.Predict(np.array(data))

        # calibration

        # output
        self.output1.Calibrate_Weights(answer[0])
        self.output2.Calibrate_Weights(answer[1])

        self.hidden1.Back_Propagation_Calibrate_Weigth((self.output1.back_propagation_error_sizes[0] + self.output2.back_propagation_error_sizes[0]) / 2)
        self.hidden2.Back_Propagation_Calibrate_Weigth((self.output1.back_propagation_error_sizes[1] + self.output2.back_propagation_error_sizes[1]) / 2)

        # show error size
        if epoch % 100 == 0:
            print(epoch)
            print("1 Error size %f" % ( np.mean((output_result[0] - answer[0]) ** 2)))
            print("2 Error size %f" % ( np.mean((output_result[1] - answer[1]) ** 2)))


    # Method for Learn network
    def Learn(self, train_array, epoch = 5000):

        # # training 
        for i in range(epoch):
            for data in train_array:

                # preprocess data
                train_data = np.array([data[0], data[1], data[2]])
                answer = np.array([data[3],data[4]])

                # learn
                self.Train(train_data, answer, i)