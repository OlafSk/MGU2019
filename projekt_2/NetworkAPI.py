import numpy as np
import time
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + ' Days '
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + ' hours '
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + ' minutes '
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + ' seconds '
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + ' miliseconds '
        i += 1
    if f == '':
        f = '0 miliseconds'
    return f


class NetworkAPI():
    def __init__(self, model, dataloaders, name_to_save, optimizer, lr=0.01):
        self.model=model,
        self.model=self.model[0]
        self.dataloaders=dataloaders

        self.name_to_save=name_to_save
        self.lr=lr
        self.best_acc=0
        self.epochs=0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def train(self):

        running_loss = 0.0
        running_corrects = 0

        self.model.train()

        for inputs, labels in self.dataloaders['train']:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward
            outputs = self.model(inputs)
            preds = torch.argmax(outputs, dim=1)
            loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()

        epoch_loss = running_loss / len(self.dataloaders['train'].dataset)
        epochs_acc = running_corrects / len(self.dataloaders['train'].dataset) * 100

        return epoch_loss, epochs_acc

    def evaluate(self):

        running_loss = 0
        running_corrects = 0

        self.model.eval()

        for inputs, labels in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            preds = torch.argmax(outputs, dim=1)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()

        epoch_loss = running_loss / len(self.dataloaders['val'].dataset)
        epochs_acc = running_corrects / len(self.dataloaders['val'].dataset) * 100

        return epoch_loss, epochs_acc

    def train_loop(self, epochs):
        print("====== HYPERPARAMETERS ======")
        print("starting epoch=", self.epochs)
        print("epochs to go=", epochs)
        print("Starting learning rate=", self.lr)
        print("=" * 30)
        elapsed_start_time = time.time()
        best_val_loss = np.inf
        bad_epochs = 0

        for epoch in range((self.epochs),(self.epochs+epochs)):
            print(68*'-')
            train_start_time = time.time()

            train_loss, train_acc = self.train()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            print('| Epoch: {:3d} | Time: {:6.2f}s | Train loss: {:5.2f} | Train acc: {:4.2f}|'
              .format(epoch+1, (time.time() - train_start_time), train_loss, train_acc))

            val_start_time = time.time()

            val_loss, val_acc = self.evaluate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            print('| Epoch: {:3d} | Time: {:6.2f}s | Val loss:   {:5.2f} | Val acc:   {:4.2f}|'
              .format(epoch+1, (time.time() - val_start_time), val_loss, val_acc))

            if val_loss < best_val_loss:
                state = {
                            'net': self.model.state_dict(),
                            'acc': val_acc,
                            'epoch': epoch,
                            'train_losses': self.train_losses,
                            'train_accuracies': self.train_accuracies,
                            'val_losses': self.val_losses,
                            'val_accuracies': self.val_accuracies,
                        }
                torch.save(state, self.name_to_save+'.pth.tar')

                best_val_loss = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs == 10:
                print(68*'-')
                print('| Total time elapsed: {:20}'.format(format_time(time.time() - elapsed_start_time)))
                break
        print(68*'-')
        print('| Total time elapsed: {:20}'.format(format_time(time.time() - elapsed_start_time)))

        return self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies

    def load_checkpoint(self):
        checkpoint = torch.load(self.name_to_save+'.pth.tar')
        self.epochs=checkpoint['epoch']
        self.best_acc=checkpoint['acc']
        self.model.load_state_dict(checkpoint['net'])
        self.train_losses=checkpoint['train_losses']
        self.train_accuracies=checkpoint['train_accuracies']
        self.val_losses=checkpoint['val_losses']
        self.val_accuracies=checkpoint['val_accuracies']

    def class_accuracy(self, classes):
        train_correct = list(0. for i in range(10))
        train_total = list(0. for i in range(10))
        val_correct = list(0. for i in range(10))
        val_total = list(0. for i in range(10))

        with torch.no_grad():
            for inputs, labels in self.dataloaders['train']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                c = (preds==labels)
                for i in range(len(labels)):
                    label = labels[i]
                    train_correct[label] += c[i].item()
                    train_total[label] += 1

            for inputs, labels in self.dataloaders['val']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                c = (preds==labels)
                for i in range(len(labels)):
                    label = labels[i]
                    val_correct[label] += c[i].item()
                    val_total[label] += 1

        print('| Class'+19*' '+'| Train  | Test  |')
        print('-'*44)
        for i in range(10):
            print('| Accuracy of: {:10} | {:4.2f}% | {:4.2f}%|'
                  .format(classes[i], 100 * train_correct[i] / train_total[i], 100 * val_correct[i] / val_total[i]))

    def plot_errors(self):
        plt.plot(self.train_losses)
        plt.plot(self.val_losses)
        plt.legend(['Train error', 'Val error'], loc='upper right')
        plt.title('Training set and validation set errors')
        plt.xlabel('Iterations (epochs)')
        plt.ylabel('Error')
        plt.show()

    def plot_accuracy(self):
        plt.plot(self.train_accuracies)
        plt.plot(self.val_accuracies)
        plt.legend(['Train accuracy', 'Val accuracy'], loc='lower right')
        plt.title('Training set and validation set accuracy')
        plt.xlabel('Iterations (epochs)')
        plt.ylabel('Accuracy')
        plt.show()
