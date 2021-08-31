# libs
import numpy as np
from core.network import NN

# data for check
train_array = np.array([
    [-10,-9,10,1,0],
    [-10,-6,10,1,0],
    [-10,-2,10,1,0],
    [-10,2,10,0,1],
    [-10,6,10,0,1],
    [-10,9,10,0,1],
    [0,8,20,1,0],
    [0,6,20,1,0],
    [0,1,20,1,0],
    [0,12,20,0,1],
    [0,15,20,0,1],
    [0,18,20,0,1],
    [-20,-18,0,1,0],
    [-20,-16,0,1,0],
    [-20,-12,0,1,0],
    [-20,-2,0,0,1],
    [-20,-5,0,0,1],
    [-20,-8,0,0,1],
    ])

brain = NN(learn_rate=0.2)

brain.Learn(train_array, 5000)


while True:
    value1 = int(input("Value 1: "))
    value2 = int(input("Value 2: "))
    value3 = int(input("Value 3: "))

    # data for check
    data = np.array([value1, value2, value3])

    
    result = brain.Predict(data)

    print(result)

