# https://github.com/lucidrains/vit-pytorch
import logging
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, recall_score, confusion_matrix
from sklearn.metrics import auc


class PRS(nn.Module):
    def __init__(self, fixed_th=False):
        super().__init__()
        self.non_torch_model = True
        self._ = nn.Linear(1, 1)  # only for pytorch compatibility

        path = '/data/home/zhangzc/project/Brain_IMGEN/data/PRS/PRSice/data/'
        prs = []
        for i in ['ADNI_1', 'ADNI_2', 'ADNI_3', 'ADNI_GO', 'ADC1', 'ADC2', 'ADC3', ]:
            p = os.path.join(path, '%s.all_score' % i)
            df = pd.read_csv(p, sep=' ', header=0, dtype=str)
            assert not pd.isnull(df).values.any()
            prs.append(df)
            # prs.append(df[['IID', 'FID', 'Pt_0.05']])
        prs = pd.concat(prs, axis=0, join='inner')

        prs['IID'] = prs['IID']
        prs = prs.set_index('IID')
        self.prs_th = prs.columns[1:]

        del prs['FID']
        prs = prs.astype(float)

        # *note*: impute the nan (since diff threshold may result in same snp set, hence PRSice ignore some threshold)
        self.prs = prs.ffill(axis=1)

        self.fixed_th = fixed_th
        if fixed_th:
            self.inx = np.where(self.prs_th == 'Pt_0.005')[0].item()

    def evaluate_data(self, val_loader, device, dtype='float32'):

        label = val_loader.dataset.imgdata.labels.ravel()
        label_pid = val_loader.dataset.imgdata.id_info.astype(str)
        no_prs = label_pid[[(i not in self.prs.index) for i in label_pid]]
        if len(no_prs) > 0:
            logging.warning('%d subjects do not have prs scores' % len(no_prs))
            _df = pd.DataFrame(data=self.prs.median(0).values.reshape(1, -1).repeat(len(no_prs), 0),
                               columns=self.prs.columns, index=no_prs)
            df = pd.concat([self.prs, _df])
        else:
            df = self.prs

        prs = df.loc[label_pid]
        prs = prs.values
        predicts = prs[:, self.inx].reshape(-1, 1, 1)
        groundtruths = label.reshape(-1, 1, 1)
        group_labels = val_loader.dataset.imgdata.dis_label.reshape(-1, 1, 1)

        # (N, 1, 1)
        return predicts, groundtruths, group_labels, 0

    def fit(self, train_loader, optimizer, device, dtype):
        logging.warning("PRS is now only implemented for non-image data, modals[0] will be ommited")

        if self.fixed_th:
            return torch.tensor(0)

        label = train_loader.dataset.imgdata.labels.ravel()
        label_pid = train_loader.dataset.imgdata.id_info.astype(str)
        #

        inx = [i in self.prs.index.values for i in label_pid]
        label = label[inx]
        label_pid = label_pid[inx]
        prs = self.prs.loc[label_pid]
        prs_pid = prs.index.values
        prs = prs.values

        assert len(label_pid) == len(prs_pid)
        assert (label_pid == prs_pid).all()

        perf = []
        for n, th in enumerate(self.prs_th):
            pred = prs[:, n]
            fpr, tpr, thr = roc_curve(label, pred)
            perf.append(auc(fpr, tpr))
        self.inx = np.argmax(perf)
        print(self.prs_th[self.inx])
        print(perf[self.inx])
        return torch.tensor(0)

    def forward(self, *args):
        raise NotImplementedError
