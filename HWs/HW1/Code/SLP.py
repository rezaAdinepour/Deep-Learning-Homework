import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(2, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.fc(x))






# class single_layer_perceptron():
#     def __init__(self, input_neurons=2, epoch=100, learning_rate=0.01):
#         self.w = np.random.rand(input_neurons + 1) - 0.5
#         self.learning_rate = learning_rate

#     def predict(self, inputs):
#         summation = np.dot(inputs, self.w[1:]) + self.w[0]
#         activation = 1.0 if (summation > 0.0) else 0.0

#         return activation
    
    
#     def train(self, X, y, epochs=100):
#         for EPOCH in range(epochs):
#             fail_count = 0
#             i = 0
#             train_loss = 0
#             loss_history = []
#             accuracy_history = []

#             for inputs, label in zip(X, y):
#                 i = i + 1
#                 prediction = self.predict(inputs)

#                 if (label != prediction):
#                     self.w[1:] += self.learning_rate * (label - prediction) * inputs.reshape(inputs.shape[0])
#                     self.w[0] += self.learning_rate * (label - prediction)
#                     fail_count += 1

#                 # Calculate loss
#                 loss = (label - prediction) ** 2
#                 train_loss += loss

#                 loss_history.append(np.mean(train_loss))
                

#                 plt.cla()
#                 plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet', marker='.')
#                 line_x = np.arange(-1, 1, 0.1)
#                 line_y = (-self.w[0] - self.w[1] * line_x) / self.w[2]
#                 plt.plot(line_x, line_y)
#                 plt.xlim(-0.1, 1.1)
#                 plt.ylim(-0.1, 1.1)
#                 plt.text(-0.1, 1.11, 'epoch|iter = {:2d}|{:2d}'.format(EPOCH, i), fontdict={'size': 14, 'color':  'black'})
#                 plt.pause(0.01)

#             # Calculate accuracy
#             accuracy = (i - fail_count) / i
#             accuracy_history.append(np.mean(accuracy))

#             print(f"Epoch: {EPOCH+1}, Loss: {np.mean(train_loss):.4f}, Accuracy: {np.mean(accuracy):.4f}")

#             if (fail_count == 0):
#                 plt.show()
#                 break
#         loss_history = np.array(loss_history)
#         accuracy_history = np.array(accuracy_history)

#         print(loss_history)
#         print(accuracy_history)

#         return loss_history, accuracy_history
    






class single_layer_perceptron():
    def __init__(self, input_neurons=2, epoch=100, learning_rate=0.01):
        self.w = np.random.rand(input_neurons + 1) - 0.5
        self.learning_rate = learning_rate
        self.loss_history = []
        self.accuracy_history = []

    def predict(self, inputs):
        summation = np.dot(inputs, self.w[1:]) + self.w[0]
        activation = 1.0 if (summation > 0.0) else 0.0
        return activation
    
    def train(self, X, y, epochs=100):
        for EPOCH in range(epochs):
            fail_count = 0
            i = 0
            train_loss = 0

            for inputs, label in zip(X, y):
                i = i + 1
                prediction = self.predict(inputs)

                if (label != prediction):
                    self.w[1:] += self.learning_rate * (label - prediction) * inputs.reshape(inputs.shape[0])
                    self.w[0] += self.learning_rate * (label - prediction)
                    fail_count += 1

                # Calculate loss
                loss = (label - prediction) ** 2
                train_loss += loss

                plt.cla()
                plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet', marker='.')
                line_x = np.arange(-2, 2, 0.1)
                line_y = (-self.w[0] - self.w[1] * line_x) / self.w[2]
                plt.plot(line_x, line_y)
                plt.xlim(-0.1, 1.1)
                plt.ylim(-0.1, 1.1)
                plt.text(-0.1, 1.1, 'epoch|iter = {:2d}|{:2d}'.format(EPOCH, i), fontdict={'size': 14, 'color':  'black'})
                plt.pause(0.01)

            # Calculate accuracy
            accuracy = (i - fail_count) / i

            self.loss_history.append(np.mean(train_loss))
            self.accuracy_history.append(np.mean(accuracy))

            print(f"Epoch: {EPOCH+1}, Loss: {np.mean(train_loss):.4f}, Accuracy: {np.mean(accuracy):.4f}")


            

            if (fail_count == 0):
                plt.show()
                break

        fig = plt.figure(figsize=(9, 7))
        x = np.arange(EPOCH + 1)
        plt.plot(x, self.loss_history, marker='o')
        plt.title("Loss history")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        fig = plt.figure(figsize=(9, 7))
        plt.plot(x, self.accuracy_history, marker='o')
        plt.title("Accuracy history")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

        loss_history = np.array(self.loss_history)
        accuracy_history = np.array(self.accuracy_history)

        return loss_history, accuracy_history
    

    # def plot_loss(self, loss_history, epochs=100):
    #     x = np.arange(epochs)
    #     fig = plt.figure(figsize=(9, 7))
    #     plt.plot(x, self.loss_history, marker='o')
    #     plt.title("Loss history")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Loss")

    # def plot_accuracy(self, accuracy_history, epochs=100):
    #     x = np.arange(epochs)
    #     fig = plt.figure(figsize=(9, 7))
    #     plt.plot(x, self.accuracy_history, marker='o')
    #     plt.title("Accuracy history")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Accuracy")