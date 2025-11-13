# -*- coding: utf-8 -*-
"""
@author: Zi-Chao ZHANG
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import numpy as np


# __all__ = [
#     'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#     'resnet152', 'resnet200'
# ]


class Embedding(nn.Module):
    def __init__(self, patch_size, inplanes, outsize=4, stride=1, downsample=None):
        super(Embedding, self).__init__()

        self.outsize = outsize
        patch_size = np.array(patch_size)

        self.conv1 = nn.Conv3d(inplanes, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=0)

        self.conv3 = nn.Conv3d(16, 32, kernel_size=2, stride=1, padding=0)
        self.conv4 = nn.Conv3d(32, 32, kernel_size=2, stride=1, padding=0)

        self.conv5 = nn.Conv3d(32, 32, kernel_size=2, stride=1, padding=0)
        self.conv6 = nn.Conv3d(32, 32, kernel_size=2, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool3D = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0)

        # fc_insize = int(np.prod(np.ceil(np.ceil(np.ceil(patch_size / 2) / 2) / 2))) * 32
        fc_insize = 32
        self.fc7 = nn.Sequential(nn.Linear(fc_insize, 32),
                                 nn.ReLU())
        self.drop = nn.Dropout(0.3)
        self.fc8 = nn.Sequential(nn.Linear(32, outsize),
                                 nn.ReLU())
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.uniform_(m.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool3D(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool3D(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.maxpool3D(x)
        x = x.view(x.shape[0], -1)

        x = self.fc7(x)
        x = self.drop(x)
        x = self.fc8(x)

        return x


class LDMIL(nn.Module):
    def __init__(self, num_patch, patch_size, inplanes, ):
        super(LDMIL, self).__init__()
        embed_size = 4
        self.criterion = nn.CrossEntropyLoss()
        self.embeddings = nn.ModuleList()
        for i in range(num_patch):
            self.embeddings.append(Embedding(patch_size, inplanes, outsize=embed_size))

        self.mlp = nn.Sequential(nn.Dropout(0.3),
                                 nn.Linear(embed_size * num_patch, 4 * num_patch), nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(4 * num_patch, 1 * num_patch), nn.ReLU(),
                                 nn.Dropout(0.25),
                                 nn.Linear(1 * num_patch, 2),
                                 )

        self.mlp[-1].weight = nn.init.uniform_(self.mlp[-1].weight)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         m.weight = nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        '''

        Args:
            x: x.shape = (N, C, X, Y, Z), where C is the num of patch

        Returns:

        '''
        fea_list = []
        for p in range(x.shape[1]):
            fea_list.append(self.embeddings[p](x[:, p, :, :, :, :]))

        gfeature = torch.cat(fea_list, dim=-1)
        out = self.mlp(gfeature)
        return out,

    def evaluate_data(self, val_loader, device, dtype='float32'):
        predicts = []
        groundtruths = []
        group_labels = []

        with torch.no_grad():
            self.train(False)
            for i, data in enumerate(val_loader, 0):
                inputs, aux_labels, labels, dis_label = data
                inputs = inputs.to(device=device, dtype=dtype)
                outputs = self(inputs)
                predicts.append(outputs)
                groundtruths.append(labels.numpy()[:, 0, :])  # multi patch
                group_labels.append(dis_label)

        _probs = torch.cat([j[0] for j in predicts], axis=0).cpu()
        _probs = _probs.squeeze()

        # for clf only
        predicts = np.array(
            [np.concatenate([j[i].softmax(dim=1)[:, -1:].cpu().numpy() for j in predicts], axis=0)
             for i in range(len(predicts[0]))])
        predicts = predicts.transpose((1, 0, 2))
        groundtruths = np.concatenate(groundtruths, axis=0)
        group_labels = np.concatenate([i.cpu().unsqueeze(-1).numpy() for i in group_labels], axis=0)

        groundtruths = groundtruths[:, :, -1:]
        predicts = predicts[:, :, -1:]

        val_loss = self.criterion(_probs, torch.from_numpy(groundtruths.squeeze().astype(int)))

        return predicts, groundtruths, group_labels, val_loss

    def fit(self, train_loader, optimizer, device, dtype):
        losses = torch.zeros(1, dtype=dtype, device=device, )
        self.train(True)

        c = 0
        batch_size = train_loader.batch_size
        inputs_buf = torch.Tensor()
        labels_buf = torch.Tensor()
        for n, data in enumerate(train_loader, 0):
            inputs, _, labels, _ = data

            ## to collect data for the case that input might contains nan
            inx = ~torch.isnan(labels.view(labels.shape[0], -1)[:, 0])
            inx = inx & (~torch.isnan(inputs.view(inputs.shape[0], -1)[:, 0]))
            inputs_buf = torch.cat([inputs_buf, inputs[inx]], 0)
            labels_buf = torch.cat([labels_buf, labels[inx]], 0)
            if (n + 1) < len(train_loader):
                if inputs_buf.shape[0] < batch_size + 2:
                    continue
                else:
                    inputs = inputs_buf[:batch_size]
                    labels = labels_buf[:batch_size]
                    inputs_buf = inputs_buf[batch_size:]
                    labels_buf = labels_buf[batch_size:]
            else:
                inputs = inputs_buf
                labels = labels_buf
            c += 1
            ##

            # multi patch
            labels = labels[:, 0, 0, :]
            inputs = inputs.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=torch.int64)
            optimizer.zero_grad()
            outputs = self(inputs)

            # for i in range(labels.shape[1]):
            #     assert outputs[i].shape == labels[:, i, :].shape
            #     non_nan = ~torch.isnan(labels[:, i, :])
            #     if non_nan.any():
            #         loss = criterion(outputs[i][non_nan], labels[:, i, :][non_nan])
            #         loss.backward(retain_graph=True)
            #         losses[i] += loss.detach()

            # softmax
            # assert outputs[0].shape == labels.shape
            labels = labels.view(-1)
            non_nan = ~torch.isnan(labels)
            if non_nan.any():
                loss = self.criterion(outputs[0][non_nan], labels[non_nan])
                loss.backward()
                losses[0] += loss.detach()
            optimizer.step()
        return losses / c


if __name__ == '__main__':
    test_data = []
    num_patch = 10
    num_sample = 8
    patch_size = (25, 25, 25)
    test_data = torch.randn((num_sample, num_patch) + patch_size)

    net = LDMIL(num_patch, patch_size, inplanes=1)

    a = net.embeddings[0](test_data[0:1, 0:1])

    o = net(test_data)
