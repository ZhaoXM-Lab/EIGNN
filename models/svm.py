# https://github.com/lucidrains/vit-pytorch
import logging
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SVM(nn.Module):
    def __init__(self, num_classes=2, kernel='linear'):
        super().__init__()
        self._ = nn.Linear(1, 1)  # only for pytorch compatibility
        self.non_torch_model = True
        self.scaler = StandardScaler()
        self.model = SVC(kernel=kernel, probability=True)

    def evaluate_data(self, val_loader, device, dtype='float32'):
        # for i, data in enumerate(val_loader, 0):
        #     inputs, aux_labels, labels, dis_label = data

        data = np.array([i for i in val_loader.dataset.imgdata.aux_labels[0]])
        data[np.isnan(data)] = 0
        X_test = self.scaler.transform(data)
        predicts = self.model.predict_proba(X_test)[:, 1].reshape(-1, 1, 1)
        groundtruths = val_loader.dataset.imgdata.labels.reshape(-1, 1, 1)
        group_labels = val_loader.dataset.imgdata.dis_label.reshape(-1, 1, 1)
        val_loss = torch.tensor(0)

        # (N, 1, 1)
        return predicts, groundtruths, group_labels, val_loss

    def fit(self, train_loader, optimizer, device, dtype):
        logging.warning("SVM is now only implemented for non-image data, modals[0] will be ommited")

        data = np.array([i for i in train_loader.dataset.imgdata.aux_labels[0]])
        label = train_loader.dataset.imgdata.labels.ravel()
        inx = ~np.isnan(data[:, 0])

        X = self.scaler.fit_transform(data[inx])
        y = label[inx]

        self.model = self.model.fit(X=X, y=y)

        # for n, data in enumerate(train_loader, 0):
        #     inputs, aux_labels, labels, dis_label = data

        return torch.tensor(0)

    def forward(self, *args):
        raise NotImplementedError
