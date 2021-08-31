# libs
import numpy as np
from core.network import NN

# data for check
train_array = np.array([
    [1,8,20,1,0],
    [1,5,20,1,0],
    [1,1,20,1,0],
    [1,20,20,0,1],
    [1,16,20,0,1],
    [1,18,20,0,1],
    [-20,-15,0,1,0],
    ])

brain = NN(learn_rate=0.3)

brain.Learn(train_array, 5000)


while True:
    value1 = int(input("Value 1: "))
    value2 = int(input("Value 2: "))
    value3 = int(input("Value 3: "))

    # data for check
    data = np.array([value1, value2, value3])
    result = brain.Predict(data)

    print(result)

