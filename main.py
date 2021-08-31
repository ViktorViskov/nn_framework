# libs
import numpy as np
from core.network import NN

# data for check
train_array = np.array([
    [-35,1,0],
    [-25,1,0],
    [-15,1,0],
    [0,1,0],
    [5,1,0],
    [35,0,1],
    [45,0,1],
    [50,0,1],
    ])

brain = NN()

brain.Learn(train_array, 5000)


while True:
    value = int(input("Value 1: "))

    # data for check
    data = np.array([value])
    result = brain.Predict(data)

    print(result)

