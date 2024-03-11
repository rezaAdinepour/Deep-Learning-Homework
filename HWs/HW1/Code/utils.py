import numpy as np




def to_categorical(input, num_classes):
    labels = []
    for el in input:
        label = [0 for _ in range(num_classes)]
        label[el] = 1
        labels.append(label)

    return labels



def predict(inputs, weights):
    summation = np.dot(inputs, weights[1:]) + weights[0]
    activation = 1.0 if (summation > 0.0) else 0.0

    return activation