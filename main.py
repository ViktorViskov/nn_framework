# libs
import numpy as np
from core.network import NN

# data for check
train_array = np.array([
    [1,1,0,0,0],
    [5,0,1,0,0],
    [10,0,0,1,0],
    [15,0,0,0,1],
    ])

brain = NN()

brain.Learn(train_array, 50000)


while True:
    value = int(input("Value 1: "))

    # data for check
    data = np.array([value])
    result = brain.Predict(data)

    print(result)

