# libs
import numpy as np
from core.neuron import Neuron

# class for neuron network
class NN:
    # constructor
    def __init__(self):

        # learn rate
        learn_rate = 0.1

        # Creating neurons

        # layer 1
        self.laeyr1_hidden1 = Neuron(learn_rate, np.random.random())
        self.laeyr1_hidden2 = Neuron(learn_rate, np.random.random())
        self.laeyr1_hidden3 = Neuron(learn_rate, np.random.random())
        self.laeyr1_hidden4 = Neuron(learn_rate, np.random.random())

        # layer 2
        self.laeyr2_hidden1 = Neuron(learn_rate, np.random.random(),np.random.random())
        self.laeyr2_hidden2 = Neuron(learn_rate, np.random.random(),np.random.random(),np.random.random())
        self.laeyr2_hidden3 = Neuron(learn_rate, np.random.random(),np.random.random(),np.random.random())
        self.laeyr2_hidden4 = Neuron(learn_rate, np.random.random(),np.random.random())

        # output
        self.output1 = Neuron(learn_rate,np.random.random(),np.random.random())
        self.output2 = Neuron(learn_rate,np.random.random(),np.random.random(),np.random.random())
        self.output3 = Neuron(learn_rate,np.random.random(),np.random.random(),np.random.random())
        self.output4 = Neuron(learn_rate,np.random.random(),np.random.random())

    # Method for prediction
    def Predict(self, data):

        # layer 1
        r_l1_h1 = self.laeyr1_hidden1.Predict(data)
        r_l1_h2 = self.laeyr1_hidden2.Predict(data)
        r_l1_h3 = self.laeyr1_hidden3.Predict(data)
        r_l1_h4 = self.laeyr1_hidden4.Predict(data)

        # layer 2
        r_l2_h1 = self.laeyr2_hidden1.Predict(np.array([r_l1_h1,r_l1_h2]))
        r_l2_h2 = self.laeyr2_hidden2.Predict(np.array([r_l1_h1,r_l1_h2,r_l1_h3]))
        r_l2_h3 = self.laeyr2_hidden3.Predict(np.array([r_l1_h2,r_l1_h3,r_l1_h4]))
        r_l2_h4 = self.laeyr2_hidden4.Predict(np.array([r_l1_h3,r_l1_h4]))

        # output
        r_o1 = self.output1.Predict(np.array([r_l2_h1, r_l2_h2]))
        r_o2 = self.output2.Predict(np.array([r_l2_h1, r_l2_h2, r_l1_h3]))
        r_o3 = self.output3.Predict(np.array([r_l2_h2, r_l2_h3, r_l2_h4]))
        r_o4 = self.output4.Predict(np.array([r_l2_h3, r_l2_h4]))

        # output
        return np.array([r_o1,r_o2,r_o3,r_o4])
    
    # Method for one time training
    def Train(self, data, answer, epoch):
        # prediction
        output_result = self.Predict(np.array(data))

        # calibration

        # output
        self.output1.Calibrate_Weights(answer[0])
        self.output2.Calibrate_Weights(answer[1])
        self.output3.Calibrate_Weights(answer[2])
        self.output4.Calibrate_Weights(answer[3])

        # layer 2
        self.laeyr2_hidden1.Back_Propagation_Calibrate_Weigth((self.output1.back_propagation_error_sizes[0] + self.output2.back_propagation_error_sizes[0]) / 2)
        self.laeyr2_hidden2.Back_Propagation_Calibrate_Weigth((self.output1.back_propagation_error_sizes[1] + self.output2.back_propagation_error_sizes[1] + self.output3.back_propagation_error_sizes[0]) / 3)
        self.laeyr2_hidden3.Back_Propagation_Calibrate_Weigth((self.output2.back_propagation_error_sizes[2] + self.output3.back_propagation_error_sizes[1] + self.output4.back_propagation_error_sizes[0]) / 3)
        self.laeyr2_hidden4.Back_Propagation_Calibrate_Weigth((self.output3.back_propagation_error_sizes[2] + self.output4.back_propagation_error_sizes[1]) / 2)

        # layer 2
        self.laeyr1_hidden1.Back_Propagation_Calibrate_Weigth((self.laeyr2_hidden1.back_propagation_error_sizes[0] + self.laeyr2_hidden2.back_propagation_error_sizes[0]) / 2)
        self.laeyr1_hidden2.Back_Propagation_Calibrate_Weigth((self.laeyr2_hidden1.back_propagation_error_sizes[1] + self.laeyr2_hidden2.back_propagation_error_sizes[1] + self.laeyr2_hidden3.back_propagation_error_sizes[0]) / 3)
        self.laeyr1_hidden3.Back_Propagation_Calibrate_Weigth((self.laeyr2_hidden2.back_propagation_error_sizes[2] + self.laeyr2_hidden3.back_propagation_error_sizes[1] + self.laeyr2_hidden4.back_propagation_error_sizes[0]) / 3)
        self.laeyr1_hidden4.Back_Propagation_Calibrate_Weigth((self.laeyr2_hidden3.back_propagation_error_sizes[2] + self.laeyr2_hidden4.back_propagation_error_sizes[1]) / 2)

        # show error size
        print(epoch)
        print("1 Error size %f" % ( np.mean((output_result[0] - answer[0]) ** 2)))
        print("2 Error size %f" % ( np.mean((output_result[1] - answer[1]) ** 2)))
        print("3 Error size %f" % ( np.mean((output_result[2] - answer[2]) ** 2)))
        print("4 Error size %f" % ( np.mean((output_result[3] - answer[3]) ** 2)))

    # Method for Learn network
    def Learn(self, train_array, epoch = 5000):

        # # training 
        for i in range(epoch):
            for data in train_array:

                # preprocess data
                train_data = np.array([data[0]])
                answer = np.array([data[1],data[2],data[3],data[4]])

                # learn
                self.Train(train_data, answer, i)