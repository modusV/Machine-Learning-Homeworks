import torch
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import time
from functools import wraps


n_classes = 100


def watcher(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f" ===> took {end-start} seconds")
        return result
    return wrapper


# function to define an old style fully connected network (multilayer perceptrons)
class old_nn(nn.Module):
    def __init__(self):
        super(old_nn, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, n_classes)  # last FC for classification

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


# function to define the convolutional network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # conv2d first parameter is the number of kernels at input (you get it from the output value of the previous layer)
        # conv2d second parameter is the number of kernels you wanna have in your convolution, so it will be the n. of kernels at output.
        # conv2d third, fourth and fifth parameters are, as you can read, kernel_size, stride and zero padding :)
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5, stride=2, padding=0)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_final = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(64 * 4 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, n_classes)  # last FC for classification

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.pool(self.conv_final(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        # hint: dropout goes here!
        x = self.fc2(x)
        return x


# function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_kernel(model):
    model_weights = model.state_dict()
    fig = plt.figure()
    plt.figure(figsize=(10, 10))
    for idx, filt in enumerate(model_weights['conv1.weight']):
        # print(filt[0, :, :])
        if idx >= 32: continue
        plt.subplot(4, 8, idx + 1)
        plt.imshow(filt[0, :, :], cmap="gray")
        plt.axis('off')

    plt.show()


def plot_kernel_output(model, images):
    fig1 = plt.figure()
    plt.figure(figsize=(1, 1))

    img_normalized = (images[0] - images[0].min()) / (images[0].max() - images[0].min())
    plt.imshow(img_normalized.numpy().transpose(1, 2, 0))
    plt.show()
    output = model.conv1(images)
    layer_1 = output[0, :, :, :]
    layer_1 = layer_1.data

    fig = plt.figure()
    plt.figure(figsize=(10, 10))
    for idx, filt in enumerate(layer_1):
        if idx >= 32: continue
        plt.subplot(4, 8, idx + 1)
        plt.imshow(filt, cmap="gray")
        plt.axis('off')
    plt.show()


def test_accuracy(net, dataloader):
    ########TESTING PHASE###########

    # check accuracy on whole test set
    correct = 0
    total = 0
    net.eval()  # important for deactivating dropout and correctly use batchnorm accumulated statistics
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test set: %d %%' % (
        accuracy))
    return accuracy


def show_dataset(dataiter):
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))


def plot_values(accuracy_values, loss_values):

    fig = plt.figure(figsize=(10, 20))

    ax = fig.add_subplot(211)
    ax.plot(accuracy_values, '-bo', label='accuracy')
    ax.set_title("Accuracy ")
    ax.set_xlabel("Epochs")
    ax.legend()

    ax1 = fig.add_subplot(212)
    ax1.plot(loss_values, '-ro', label='loss')
    ax1.set_title("Loss over epochs")
    ax1.set_xlabel("Epochs")
    ax1.legend()

    fig.show()

@watcher
def train(net, trainloader, testloader, criterion, optimizer, nepochs):

    ########TRAINING PHASE###########
    n_loss_print = len(trainloader)  # print every epoch, use smaller numbers if you wanna print loss more often!
    n_epochs = nepochs
    accuracy_values = []
    loss_values = []
    print("Starting Training")
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        net.train()  # important for activating dropout and correctly train batchnorm
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs and cast them into cuda wrapper
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % n_loss_print == (n_loss_print - 1):
                loss_values.append(running_loss / n_loss_print)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / n_loss_print))
                running_loss = 0.0
        accuracy_values.append(test_accuracy(net, testloader))

    print('Finished Training')
    plot_values(accuracy_values, loss_values)


if __name__ == '__main__':
    # transform are heavily used to do simple and complex transformation and data augmentation
    transform_train = transforms.Compose(
        [
            # transforms.Resize((40, 40)),
            # transforms.RandomCrop(size=[32, 32], padding=0),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    transform_test = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                             download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                              shuffle=True, num_workers=4, drop_last=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=4, drop_last=True)

    print("Dataset loaded")
    dataiter = iter(trainloader)

    # show images just to understand what is inside the dataset ;)
    # show_dataset(dataiter)

    print("NN instantiated")
    # net = old_nn()

    net = CNN()

    ####
    # for Residual Network:
    # net = models.resnet18(pretrained=True)
    # net.fc = nn.Linear(512, n_classes) #changing the fully connected layer of the already allocated network
    ####

    ###OPTIONAL:
    # print("####plotting kernels of conv1 layer:####")
    # plot_kernel(net)
    ####
    net = net.cuda()

    criterion = nn.CrossEntropyLoss().cuda()  # it already does softmax computation for use!
    optimizer = optim.Adam(net.parameters(), lr=0.0001)  # better convergency w.r.t simple SGD :)
    print("Optimizer and criterion instantiated")

    ###OPTIONAL:
    # print("####plotting output of conv1 layer:#####")
    # plot_kernel_output(net,images)
    ###

    train(net=net,
          trainloader=trainloader,
          testloader=testloader,
          criterion=criterion,
          optimizer=optimizer,
          nepochs=20)
