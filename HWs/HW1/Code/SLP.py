import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import torch.nn.functional as F


# Define the single layer perceptron network
class single_layer_perceptron(nn.Module):
    def __init__(self, input_size):
        super(single_layer_perceptron, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))





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
    






class Single_Layer_Perceptron():
    def __init__(self, input_neurons=2, epoch=100, learning_rate=0.01):
        self.w = np.random.rand(input_neurons + 1) - 0.5
        self.learning_rate = learning_rate
        self.loss_history = []
        self.accuracy_history = []
        

    def predict(self, inputs):
        summation = np.dot(inputs, self.w[1:]) + self.w[0]
        # activation = 1.0 if (summation > 0.0) else 0.0
        activation = np.where(summation > 0.0, 1.0, 0.0)
        return activation
    
    def train(self, X, y, epochs=100):
        for EPOCH in range(epochs):
            fail_count = 0
            i = 0
            train_loss = 0
            predicted_labels = []

            for inputs, label in zip(X, y):
                i = i + 1
                prediction = self.predict(inputs)
                predicted_labels.append(prediction)

                if (label != prediction):
                    self.w[1:] += self.learning_rate * (label - prediction) * inputs.reshape(inputs.shape[0])
                    self.w[0] += self.learning_rate * (label - prediction)
                    fail_count += 1

                # Calculate loss = MSE = 1/N * (w'x - y)**2
                loss = ((prediction - label) ** 2).mean() / 100
                train_loss += loss

            plt.cla()
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet', marker='.')
            # line_x = np.arange(-2, 2, 0.1)
            # line_y = (-self.w[0] - self.w[1] * line_x) / self.w[2]
            # plt.plot(line_x, line_y)
            # plt.xlim(-0.1, 1.1)
            # plt.ylim(-0.1, 1.1)
            # plt.text(-0.1, 1.1, 'epoch|iter = {:2d}|{:2d}'.format(EPOCH, i), fontdict={'size': 14, 'color':  'black'})
            # plt.pause(0.01)

            line_x = np.arange(-50, 100, 0.1)
            line_y = (-self.w[0] - self.w[1] * line_x) / self.w[2]
            plt.plot(line_x, line_y)
            plt.xlim(-5, 100)
            plt.ylim(-5, 100)
            plt.text(-5, 100, 'epoch = {:2d}'.format(EPOCH), fontdict={'size': 14, 'color':  'black'})
            plt.pause(0.01)

            # Calculate accuracy
            accuracy = (i - fail_count) / i

            self.loss_history.append(np.mean(train_loss))
            self.accuracy_history.append(np.mean(accuracy))

            # Calculate final predictions
            final_predictions = [self.predict(inputs) for inputs in X]
            
            # Create confusion matrix
            cm = confusion_matrix(y, final_predictions)

            print(f"Epoch: {EPOCH+1}, Loss: {np.mean(train_loss):.4f}, Accuracy: {np.mean(accuracy):.4f}")

            if (fail_count == 0):
                plt.show()
                break

        fig = plt.figure(figsize=(9, 7))
        x = np.arange(EPOCH + 1)
        plt.plot(x, self.loss_history, color='red', label='loss', alpha=0.3, marker='o')
        plt.title("Loss history")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        fig = plt.figure(figsize=(9, 7))
        plt.plot(x, self.accuracy_history, color='blue', label='accuracy', alpha=0.3, marker='o')
        plt.title("Accuracy history")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        loss_history = np.array(self.loss_history)
        accuracy_history = np.array(self.accuracy_history)

        avg_loss = loss_history.mean()
        avg_accuracy = accuracy_history.mean()

        f1 = f1_score(y, predicted_labels)

        plt.show()

        return self.w, f1, avg_loss, avg_accuracy
    

    
    def test(self, X, y):
        # Initialize variables
        total_loss = 0
        total_correct = 0
        predicted_labels = []

        # Iterate over all samples
        for inputs, label in zip(X, y):
            # Make a prediction
            prediction = self.predict(inputs)
            predicted_labels.append(prediction)

            # Calculate loss (MSE)
            loss = ((prediction - label) ** 2).mean()
            total_loss += loss

            # Check if prediction is correct
            if prediction == label:
                total_correct += 1

        # Calculate average loss and accuracy
        avg_loss = total_loss / len(X)
        accuracy = total_correct / len(X)

        # Calculate F1 score
        f1 = f1_score(y, predicted_labels)

        return np.mean(avg_loss), np.mean(accuracy), f1









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