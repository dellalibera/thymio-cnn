import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from torch.utils.tensorboard import SummaryWriter

import argparse
from PIL import Image
import random

# Author Alessio Della Libera and Andrea Bennati

class Dataset:
    def __init__(self, path, batch_size=16):
        self.path = path
        self.idx_to_class = {}
        self.class_to_idx = {}
        self.batch_size = batch_size

    def load_dataset(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset = torchvision.datasets.ImageFolder(root=self.path, transform=transform)
        self.class_to_idx = dataset.class_to_idx

        self.idx_to_class = {val: key for key, val in self.class_to_idx.items()}

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        return train_dataset, test_dataset

    def get_loaders(self):
        train_dataset, test_dataset = self.load_dataset()
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return trainloader, testloader


class CNN(nn.Module):
    def __init__(self, output=2, lr=0.001, epochs=10):
        super(CNN, self).__init__()
        self.lr = lr
        self.epochs = epochs

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 21 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 42)
        self.fc4 = nn.Linear(42, output)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 21 * 29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def evaluate(self, testloader):
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = (correct / total) * 100
            print('Accuracy={}'.format(accuracy))

        return accuracy

    def run(self, trainloader, testloader):
        # globaliter_train = 0
        # globaliter_test = 1
        # writer_train = SummaryWriter(log_dir="./log/train")
        # writer_test = SummaryWriter(log_dir="./log/test")

        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        print('Start training....')
        for epoch in range(self.epochs):  # loop over the dataset multiple times

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                print("Epoch={:<3d} | Batch={:<3d} | Loss={:<.10f}".format(epoch + 1, i + 1, loss.item()))

                # writer_train.add_scalar('loss', loss.item(), global_step=globaliter_train)
                # globaliter_train += 1

            accuracy = self.evaluate(testloader)
            # writer_test.add_scalar('accuracy', accuracy, global_step=globaliter_test)
            # globaliter_test += 1

        # writer_train.close()
        # writer_test.close()
        print('Training Finished !!')

    def predict(self, image, display=False):
        trans = transforms.ToTensor()
        img = trans(image)

        img = img.unsqueeze(0)
        outputs = self(img)
        softmax = F.softmax(outputs, dim=1)

        _, predicted = torch.max(softmax, 1)

        softmax = list(softmax.detach().numpy()[0])
        prob = round(softmax[predicted], 3)

        if display:
            title = "Predicted Room {} with probability {:.3f}".format(predicted.item()+1, prob)
            print("{}".format(title))
            plt.title(title)
            plt.imshow(image)
            plt.show()

        return predicted.item()+1, prob


def display_images(net, src, ncol=4, nrow=2):

    images = []
    labels = []

    while len(images) != nrow*ncol:
        room = random.randint(1, 2)
        image = random.randint(0, 900)
        try:
            images.append(Image.open('{}/room{}/img{}.png'.format(src, room, image)))
            labels.append(room)
        except:
            pass

    results = [net.predict(image, display=True) for image in images]

    fig = plt.figure()
    fig.subplots_adjust(hspace=0, wspace=0.1)

    for i, image in enumerate(images):
        predicted, prob = results[i]
        label = labels[i]

        plt.subplot(nrow, ncol, i + 1)

        text = "Label={}\nPredicted={} ({:.3f})".format(label, predicted, prob)
        plt.text(0, -10, text, fontsize=18, ha='left')
        plt.imshow(image)
        plt.axis('off')

    print('GroundTruth: {}'.format(' '.join(str(labels[j]) for j in range(len(images)))))
    print('Predicted:   {}'.format(' '.join(str(results[j][0]) for j in range(len(images)))))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Thymio CNN')
    parser.add_argument('--src', help='Source Images Directory', required=True)
    parser.add_argument('--path_model', help='Output Model Path', default="./")
    parser.add_argument('--action', help='Action',  choices=['train', 'display'], default='display')

    args = parser.parse_args()
    src = args.src
    path_model = args.path_model
    action = args.action

    net = CNN()
    if action == 'train':
        dataset = Dataset(path=src, batch_size=16)
        trainloader, testloader = dataset.get_loaders()
        net.run(trainloader, testloader)
        net.save_model(path_model)
        net.evaluate(testloader)
    else:
        net.load_model(path_model)
        display_images(net, src)

    # tensorboard --host localhost --port 8008 --logdir=./log
