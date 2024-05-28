import torch.nn as nn
import torch
import torch.nn.functional as F

__all__ = ["cnn"]

class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5,bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5,bias=False)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.classifier = nn.Linear(hidden_dims[-1],output_dim)
        #self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.classifier(x)
        # x = self.fc3(x)
        return x,x


class CNN_MNIST(torch.nn.Module):
    def __init__(self, dataset):
        super(CNN_MNIST, self).__init__()
        self.name = "cnn"
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)
        self.softmax = nn.Softmax(dim=1)
        self.name = "cnn"
        self.activations = None

    def forward(self, x):
        # x.size() = [64, 1, 28, 28], 64张图像，每个图像1个通道，像素是28*28
        x = F.relu(self.pooling(self.conv1(x)))
        # x.size() = [64, 10, 12, 12]
        x = F.relu(self.pooling(self.conv2(x)))
        # x.size() = [64, 20, 4, 4]
        x = x.view(x.size(0), -1)  # x.size(0)，取列表中的第0个元素，每个图像16个像素点，共20个通道，即320个数据为一行
        # x.size() = [64, 320]
        x = self.fc(x)
        return self.softmax(x)


class CNN_EMNIST(torch.nn.Module):
    def __init__(self):
        super(CNN_EMNIST, self).__init__()
        self.name = "cnn"
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128 * 3 * 3, 47)
        self.activations = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.name = "cnn"
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128 * 4 * 4, 10)
        self.activations = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def cnn(conf, arch=None):
    dataset = conf.data
    if dataset == "emnist":
        model = CNN_EMNIST()
    elif "mnist" in dataset:
        model = CNN_MNIST(dataset)
    else:
        model = CNN_CIFAR()

    return model