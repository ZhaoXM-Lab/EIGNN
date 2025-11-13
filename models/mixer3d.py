from torch import nn
from functools import partial
import torch
import numpy as np
import math
import torch.nn.functional as F
import os
import logging


class GELU(nn.Module):
    """
    https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/utils/gelu.py
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


class MLPMixer3D(nn.Module):
    def __init__(self, num_patches, channels, patch_size, dim, depth, num_classes,
                 expansion_factor=4, dropout=0.):
        super(MLPMixer3D, self).__init__()

        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.criterion = nn.BCELoss()
        # self.rearr = Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)',
        #                        p1=patch_size, p2=patch_size, p3=patch_size)
        self.embed = nn.Linear(int(np.prod(patch_size) * channels), dim)

        self.mixer = nn.Sequential(
            # nn.Linear(int(np.prod(self.embed.out_size[-3:]) * out_planes), dim),
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
            ) for _ in range(depth)],
        )

        self.ln0 = nn.LayerNorm(dim)
        self.output = nn.Sequential(nn.Linear(dim, num_classes), nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                torch.nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def embedding(self, x):
        # x: batch, num_patch, channel, patch_size, patch_size, patch_size

        x = x.view(*x.shape[:2], -1)
        x = self.embed(x)
        x = self.mixer(x)
        x = self.ln0(x)
        x = x.mean(axis=1)
        return x,

    def forward(self, x):
        # x: batch, num_patch, channel, patch_size, patch_size, patch_size
        x = self.embedding(x)[0]
        return self.output(x),

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
                group_labels.append(dis_label[:, 0])

        _probs = torch.stack([torch.cat([j[i] for j in predicts], dim=0) for i in range(len(predicts[0]))], dim=0)
        _probs = _probs.transpose(0, 1).cpu()

        predicts = np.array(
            [np.concatenate([j[i].cpu().numpy() for j in predicts], axis=0) for i in range(len(predicts[0]))])
        predicts = predicts.transpose((1, 0, 2))
        groundtruths = np.concatenate(groundtruths, axis=0)
        group_labels = np.concatenate([i.cpu().unsqueeze(-1).numpy() for i in group_labels], axis=0)


        groundtruths = groundtruths[:, :, -1:]
        predicts = predicts[:, :, -1:]

        non_nan = [torch.from_numpy(~np.isnan(groundtruths[:, i, :])) for i in range(groundtruths.shape[1])]
        val_loss = sum([self.criterion(_probs[:, i, :][non_nan[i]], torch.from_numpy(groundtruths[:, i, :])[non_nan[i]])
                        for i in range(groundtruths.shape[1])])

        return predicts, groundtruths, group_labels, val_loss

    def embed_data(self, val_loader, device, dtype='float32'):
        predicts = []
        clflabels = []
        groups = []

        with torch.no_grad():
            self.train(False)
            for i, data in enumerate(val_loader, 0):
                inputs, aux_labels, labels, dis_label = data
                inputs = inputs.to(device=device, dtype=dtype)
                outputs = self.embedding(inputs)
                predicts.append(outputs)
                clflabels.append(labels[:, 0, -1].view(-1))  # multi patch)
                groups.append(dis_label[:, 0].view(-1))

        fea = torch.cat([torch.cat([j[i] for j in predicts], dim=0) for i in range(len(predicts[0]))], dim=1)
        fea = fea.cpu().numpy()
        clflabels = torch.cat(clflabels, dim=-1).cpu().numpy()
        groups = torch.cat(groups, dim=-1).cpu().numpy()
        return fea, clflabels, groups

    def fit(self, train_loader, optimizer, device, dtype):
        losses = torch.zeros(train_loader.dataset.labels.shape[1], dtype=dtype, device=device, )
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

            # multi patch
            labels = labels[:, 0, :]

            inputs = inputs.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=dtype)
            optimizer.zero_grad()
            outputs = self(inputs)

            for i in range(labels.shape[1]):
                assert outputs[i].shape == labels[:, i, :].shape
                non_nan = ~torch.isnan(labels[:, i, :])
                if non_nan.any():
                    loss = self.criterion(outputs[i][non_nan], labels[:, i, :][non_nan])
                    loss.backward(retain_graph=True)
                    losses[i] += loss.detach()
            optimizer.step()
        return losses / len(train_loader)


def mixer3d(**kwargs):
    return MLPMixer3D(channels=1, num_classes=1, **kwargs)

