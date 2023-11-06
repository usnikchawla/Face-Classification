'''
Build a Convolutional Neural Network. Train it on MNIST training set and test it
on testing set. 
'''

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils import read_mnist, save_confusion_matrix

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(4*4*64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()


'''
Define a Loss function and optimizer
'''

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


'''
Load MNIST data
'''

train_images, train_labels, test_images, test_labels = read_mnist()

# make dataloaders for training and testing
train_data = TensorDataset(torch.from_numpy(train_images), torch.from_numpy(train_labels))
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

test_data = TensorDataset(torch.from_numpy(test_images), torch.from_numpy(test_labels))
test_loader = DataLoader(test_data, batch_size=4, shuffle=True)

'''
Train the network
'''
x=[]
y=[]
rt=0
for epoch in range(4):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels.argmax(axis=1))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            if len(x)==0:
                x.append(2000)
            else:
                x.append(x[-1]+2000)
            y.append(running_loss)
            running_loss = 0.0
       
            

print('Finished Training')

plt.plot(x,y,color='blue')
plt.title('Training error')
plt.show()

correct = 0
total = 0
y_true = test_labels.argmax(axis=1)
with torch.no_grad():
    outputs = net(torch.from_numpy(test_images))
    predicted = outputs.argmax(axis=1).numpy()
    total += test_labels.shape[0]
    correct += (predicted == y_true).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))





