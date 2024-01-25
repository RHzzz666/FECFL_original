import numpy as np
import copy

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Client_ClusterFL(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device,
                 train_ds_local=None, test_ds_local=None):

        self.name = name
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr
        self.momentum = momentum
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()
        self.ds_train = train_ds_local
        self.ds_test = test_ds_local
        self.ldr_train = DataLoader(train_ds_local, batch_size=self.local_bs, shuffle=True, drop_last=True)
        self.ldr_test = DataLoader(test_ds_local, batch_size=self.local_bs, shuffle=False)
        self.acc_best = 0
        self.count = 0
        self.save_best = True

        self.ds_id = None

    def train(self, is_print=False):
        self.net.to(self.device)
        self.net.train()

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)

        epoch_loss = []
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                # optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        #         if self.save_best:
        #             _, acc = self.eval_test()
        #             if acc > self.acc_best:
        #                 self.acc_best = acc

        return sum(epoch_loss) / len(epoch_loss)

    def train_unsupervised(self, epoch=0):
        if epoch == 0:
            epoch = self.local_ep
        self.net.to(self.device)
        self.net.train()

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)
        self.loss_func = nn.MSELoss()

        epoch_loss = []
        for iteration in range(epoch):
            batch_loss = []
            for batch_idx, (images, _) in enumerate(self.ldr_train):  # Note the discard of labels
                images = images.to(self.device)
                self.net.zero_grad()
                outputs = self.net(images)
                loss = self.loss_func(outputs, images)  # Target is the input image itself
                loss.backward()

                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return sum(epoch_loss) / len(epoch_loss)

    def get_state_dict(self):
        return self.net.state_dict()

    def get_best_acc(self):
        return self.acc_best

    def get_count(self):
        return self.count

    def get_net(self):
        return self.net

    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)

    def set_ds_id(self, ds_id):
        self.ds_id = ds_id

    def eval_test(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy

    def eval_test_glob(self, glob_dl):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in glob_dl:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(glob_dl.dataset)
        accuracy = 100. * correct / len(glob_dl.dataset)
        return test_loss, accuracy

    def eval_train(self):
        self.net.to(self.device)
        self.net.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy

    def eval_test_unsupervised(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in self.ldr_test:
                data = data.to(self.device)
                output = self.net(data)
                test_loss += F.mse_loss(output, data, reduction='mean').item()
        test_loss /= len(self.ldr_test)
        return test_loss

    def eval_test_glob_unsupervised(self, glob_dl):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in glob_dl:
                data = data.to(self.device)
                output = self.net(data)
                test_loss += F.mse_loss(output, data, reduction='mean').item()
            test_loss /= len(self.ldr_test)
        return test_loss

    def eval_train_unsupervised(self):
        self.net.to(self.device)
        self.net.eval()
        train_loss = 0
        with torch.no_grad():
            for data, _ in self.ldr_train:
                data = data.to(self.device)
                output = self.net(data)
                train_loss += F.mse_loss(output, data, reduction='mean').item()
        train_loss /= len(self.ldr_train)
        return train_loss

    def refresh_dl(self):
        self.ldr_train = torch.utils.data.DataLoader(self.ds_train, batch_size=self.local_bs, shuffle=True, drop_last=True)
        self.ldr_test = torch.utils.data.DataLoader(self.ds_test, batch_size=self.local_bs, shuffle=False)

