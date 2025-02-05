# https://github.com/lucidrains/vit-pytorch
import logging
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, recall_score, confusion_matrix
from sklearn.metrics import auc
from pandas_plink import read_plink1_bin, Chunk, read_plink
import os


class APOE(nn.Module):
    def __init__(self, eqtl_path):
        super().__init__()
        self.non_torch_model = True
        self._ = nn.Linear(1, 1)  # only for pytorch compatibility
        eqtl = pd.read_csv(eqtl_path, sep='\t')
        eqtl.columns = ['snp', 'gene']
        eqtl_snps = eqtl['snp'].drop_duplicates().values
        self.snp_list = eqtl_snps

        assert 'rs7412' in self.snp_list
        assert 'rs429358' in self.snp_list

        bim, _, _ = read_plink(os.path.join('/data/datasets/ADNI/Genetics/Preprocessed/ADNI/Array/',
                                            'ADNI_3/02_QC_step2/ADNI_3_imputation_QCstep2.bed'),
                               verbose=False)
        bim = bim.set_index('snp')
        # a0 is ALT
        assert bim.loc['rs7412', 'a0'] == 'T'
        assert bim.loc['rs429358', 'a0'] == 'C'

    def evaluate_data(self, val_loader, device, dtype='float32'):
        # dfs = []
        # for i in [1, 2, 3, 'GO']:
        #     path = '/data/datasets/ADNI/Genetics/Preprocessed/ADNI/Array/APOE4_CHECK/ADNI_%s_APOE.csv' % i
        #     ap = pd.read_csv(path, index_col=None, dtype=str)
        #     ap = ap[~ap['APOE'].isnull()]
        #     dfs.append(ap)
        # df = pd.concat(dfs, ignore_index=True, axis=0)
        # df[df['0'].isin(val_loader.dataset.imgdata.id_info)]

        label = val_loader.dataset.imgdata.labels.ravel()
        label_pid = val_loader.dataset.imgdata.id_info.astype(str)

        snps = np.array([i for i in val_loader.dataset.imgdata.aux_labels[0]])
        # in the read SNP array, 0 means two a0 (ALT), 2 means two a1 (REF)

        snp1 = 2 - snps[:, self.snp_list == 'rs7412']
        snp2 = 2 - snps[:, self.snp_list == 'rs429358']
        nan_inx = np.isnan(snp1)
        apoe = ((snp1 == 0) & (snp2 == 1)) * 1.  # e3/e4
        apoe = apoe + ((snp1 == 1) & (snp2 == 1)) * 1.  # e2/e4
        apoe = apoe + ((snp1 == 0) & (snp2 == 2)) * 2.  # e4/e4
        apoe[nan_inx] = np.nan
        apoe = apoe / 2.

        predicts =apoe.reshape(-1, 1, 1)
        groundtruths = label.reshape(-1, 1, 1)
        group_labels = val_loader.dataset.imgdata.dis_label.reshape(-1, 1, 1)

        return predicts, groundtruths, group_labels, 0

    def fit(self, train_loader, optimizer, device, dtype):
        return torch.tensor(0)

    def forward(self, *args):
        raise NotImplementedError
