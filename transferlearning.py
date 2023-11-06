import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models
import torch.nn as nn
from torchvision import datasets, transforms




# how many samples per batch to load
batch_size = 8

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# choose the training and test datasets
train_data = datasets.ImageFolder('training', transform=transform)
test_data = datasets.ImageFolder('validation', transform=transform)

# make data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# specify the image classes
classes = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9']

# define the CNN architecture with resCNN18
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # load the pretrained model from pytorch
        self.resCNN18 = models.resnet18(pretrained=True)
        # freeze the parameters
        for param in self.resCNN18.parameters():
            param.requires_grad = False
        # get the number of features of the last layer
        num_ftrs = self.resCNN18.fc.in_features
        # replace the last layer with a new one
        self.resCNN18.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        # forward pass through the CNNwork
        x = self.resCNN18(x)
        return x


# initialize the NN
model = CNN()
print(model)

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# number of epochs to train the model
n_epochs = 20

valid_loss_min = np.Inf # track change in validation loss
#print training loss and testing loss every epoch
# save traing loss and testing loss every epoch
train_loss_list = []
valid_loss_list = []

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for data, target in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    with torch.no_grad():
        for data, target in test_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)

    # calculate average losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(test_loader.sampler)

    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        valid_loss_min = valid_loss


# plot the training loss and testing loss
plt.plot(train_loss_list, label='train loss')
plt.plot(valid_loss_list, label='test loss')
plt.legend()
plt.show()

test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

y_true = []
y_pred = []
model.eval()
for i, data in enumerate(test_loader):
    inputs, target = data
    output = model(inputs)
    pred = output.argmax(dim=1, keepdim=True)
    y_true.append(target.item())
    y_pred.append(pred.item())

