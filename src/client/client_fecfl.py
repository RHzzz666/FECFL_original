import numpy as np
import copy

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class Client_FECFL(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device,
                 train_data_local=None, test_data_local=None):

        self.name = name
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr
        self.momentum = momentum
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()
        self.ds_train = train_data_local
        self.ds_test = test_data_local
        self.ldr_train = DataLoader(self.ds_train, batch_size=self.local_bs, shuffle=True, drop_last=True)
        self.ldr_test = DataLoader(self.ds_test, batch_size=self.local_bs, shuffle=False)
        self.ldr_fe = DataLoader(self.ds_train, batch_size=self.local_bs, shuffle=False, drop_last=True)  # only for FE
        self.acc_best = 0
        self.count = 0
        self.save_best = True
        self.F_0 = None
        self.F_1 = None
        self.ds_id = None

    def train(self, is_print=False, epoch=0):
        if epoch == 0:
            epoch = self.local_ep
        self.net.to(self.device)
        self.net.train()

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)

        epoch_loss = []
        for iteration in range(epoch):
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

    def get_F_0(self):
        return self.F_0

    def get_F_1(self):
        return self.F_1

    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)

    def set_F_0(self, F_0):
        self.F_0 = F_0

    def set_F_1(self, F_1):
        self.F_1 = F_1

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

    def extract_features_avg(self):
        self.net.to(self.device)
        self.net.eval()
        features = []
        with torch.no_grad():
            for data, _ in self.ldr_fe:
                data = data.to(self.device)
                feature = self.net.extract_features(data).cpu().detach().numpy()
                features.append(feature)
        # features shape: [batch_num, batch_size(128), feature_dim(256)]
        res = np.average(features, axis=(0, 1))
        # shape: [feature_dim(256)]
        return res

    def extract_features_norm(self):
        self.net.to(self.device)
        self.net.eval()
        v_list = []
        U_list = {}
        unique_labels = list(set([y for _, y in self.ldr_train.dataset]))
        for i in unique_labels:
            U_list[i] = []

        with torch.no_grad():
            for data, label in self.ldr_train.dataset:
                data = data.to(self.device)
                feature = self.net.extract_features(data).cpu().detach().numpy().squeeze()
                feature_norm = np.linalg.norm(feature)
                U_list[label].append(feature_norm)

        for l in unique_labels:
            v_list.append(np.average(U_list[l]))
        return np.array(v_list)

    # def extract_features_norm_2(self):
    #     self.net.to(self.device)
    #     self.net.eval()
    #     U_list = []
    #
    #     with torch.no_grad():
    #         for data, label in self.ldr_train.dataset:
    #             data = data.to(self.device)
    #             feature = self.net.extract_features(data).cpu().detach().numpy().squeeze()
    #             feature_norm = np.linalg.norm(feature)
    #             U_list.append(feature_norm)
    #
    #     v = np.average(U_list)
    #     return v

    def extract_features_emd(self):
        self.net.to(self.device)
        self.net.eval()
        features = []
        with torch.no_grad():
            for data, _ in self.ldr_fe:
                data = data.to(self.device)
                feature = self.net.extract_features(data).cpu().detach().numpy()
                features.append(feature)
        # features shape: [batch_num, batch_size(128), feature_dim(256)]
        # res = np.average(features, axis=(0, 1))
        # # shape: [feature_dim(256)]
        res = np.concatenate(features, axis=0)
        # shape: [batch_num * batch_size(128), feature_dim(256)]
        return res

    def refresh_dl(self):
        self.ldr_train = torch.utils.data.DataLoader(self.ds_train, batch_size=self.local_bs, shuffle=True, drop_last=True)
        self.ldr_test = torch.utils.data.DataLoader(self.ds_test, batch_size=self.local_bs, shuffle=False)
        self.ldr_fe = torch.utils.data.DataLoader(self.ds_train, batch_size=self.local_bs, shuffle=False, drop_last=True)
