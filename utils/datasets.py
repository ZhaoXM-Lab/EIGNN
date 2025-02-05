import logging

logging.basicConfig(level='WARNING')
import torch
import numpy as np
import os
import torch.utils.data as data
import pickle as pkl
from nilearn.image import smooth_img
import nibabel as nib
import pandas as pd
from scipy import ndimage as nd
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Union, Iterable
from pandas_plink import read_plink
import copy

from .opts import DTYPE, BASEDIR, VAL_R, TEST_R, IMG_SIZE
from .opts import ADNI_PATH, ADNI_GENETIC_PATH, eQTL_PATH

TRAINDATA_RATIO = eval(os.getenv('TRAINDATA_RATIO', '1.'))


def batch_sampling(imgs, labels, center_mat, aux_labels, dis_labels, patch_size=(25, 25, 25), random=False,
                   shift=False, flip_axis=None):
    shift_range = [-2, -1, 0, 1, 2]
    flip_pro = 0.3
    num_patch = len(center_mat[0])
    batch_size = len(imgs)
    margin = [int(np.floor((i - 1) / 2.0)) for i in patch_size]

    batch_img = torch.tensor(data=np.zeros([num_patch * batch_size, 1, patch_size[0], patch_size[1], patch_size[2]]),
                             dtype=DTYPE)
    batch_label = torch.tensor(data=np.zeros([num_patch * batch_size] + list(labels.shape[1:])), dtype=DTYPE)
    batch_aux_label = torch.tensor(data=np.zeros([batch_size] + list(aux_labels.shape[1:])), dtype=DTYPE)
    batch_dis_label = torch.tensor(data=np.zeros([batch_size] + list(dis_labels.shape[1:])), dtype=DTYPE)

    for num, data in enumerate(zip(imgs, labels, aux_labels, dis_labels), start=0):
        img, label, aux_label, dis_label = data
        if not random:
            for ind, cors in enumerate(zip(center_mat[0], center_mat[1], center_mat[2])):
                x_cor, y_cor, z_cor = cors
                if shift:
                    x_scor = x_cor + shift_range[torch.randint(high=len(shift_range), size=(1,))]
                    y_scor = y_cor + shift_range[torch.randint(high=len(shift_range), size=(1,))]
                    z_scor = z_cor + shift_range[torch.randint(high=len(shift_range), size=(1,))]
                else:
                    x_scor, y_scor, z_scor = x_cor, y_cor, z_cor

                single_patch = img[:,
                               max(x_scor - margin[0], 0): x_scor + margin[0] + 1,
                               max(y_scor - margin[1], 0): y_scor + margin[1] + 1,
                               max(z_scor - margin[2], 0): z_scor + margin[2] + 1]
                if (not (flip_axis is None)) and (torch.rand(1) < flip_pro):
                    if isinstance(flip_axis, list):
                        single_patch = single_patch.flip(flip_axis[torch.randint(high=len(flip_axis), size=(1,))])
                    single_patch = single_patch.flip(flip_axis)

                batch_img[ind + num * num_patch,
                :single_patch.shape[0],
                :single_patch.shape[1],
                :single_patch.shape[2],
                :single_patch.shape[3]] = single_patch

                batch_label[ind + num * num_patch] = label

        else:
            raise NotImplementedError

        batch_aux_label[num] = aux_label
        batch_dis_label[num] = dis_label

    return batch_img, batch_aux_label, batch_label, batch_dis_label


def get_array_snp(path, cohort, eqtl_snps):
    bim, fam, bed = read_plink(os.path.join(path, cohort), verbose=False)
    plink_df1 = pd.DataFrame(bed[bim.snp.isin(eqtl_snps)].compute().astype(np.float16).T,
                             index=fam.iid.values,
                             columns=bim.snp[bim.snp.isin(eqtl_snps)].values)

    plink_df1 = plink_df1.loc[:, ~plink_df1.columns.duplicated()]
    ori_snps = plink_df1.columns

    inplink = np.array([i in plink_df1.columns for i in eqtl_snps])
    inx = eqtl_snps.copy()
    inx[~inplink] = eqtl_snps[inplink][0]
    plink_df1 = plink_df1.loc[:, inx]
    plink_df1.iloc[:, ~inplink] = 0
    plink_df1.columns = eqtl_snps
    return plink_df1, ori_snps


class Dataset(object):
    def __init__(self, image_path, subset, clfsetting, modals: Union[list, str], no_smooth=False,
                 only_bl=False, ids_exclude=(), ids_include=(),
                 maskmat=None, seed=1234, preload=True, dsettings=None, strict_input=True):
        '''Meta class for every dataset
        should at least implement self.get_table, self._filtering, self.dx_mapping,

        Args:
            image_path:
            subset:
            clfsetting:
            modals:
            no_smooth:
            only_bl:
            ids_exclude:
            ids_include:
            maskmat:
            seed:
            preload:
            dsettings: special setting for a dataset, set to {} in default.
            strict_input: if true, All input modality must not be null.
        '''

        if dsettings is None:
            dsettings = {}
        if not isinstance(modals, list):
            modals = [modals]

        assert modals[0] in ['mw', 'mwp1', 'mwp2', 'mwp3']

        self.preload = preload
        self.smooth = not no_smooth
        self.maskmat = maskmat
        self.clfsetting = clfsetting
        self.modals = modals
        self.strict_input = strict_input

        # get the infomation table
        info = self.get_table(image_path, modals, dsettings)
        info['GROUP'] = self.g_mapping(info['DX'].values)

        # the columns that the table should at least contain
        for col in ['DX', 'ID', 'GROUP', 'VISIT'] + modals:
            assert col in info.columns

        # subject level selection
        info = self._filtering(info)

        # temp = info[~pd.isnull(info['DX'])]
        # pid_list = temp.groupby('ID')['IMAGE'].any()
        # pid_list = pid_list[pid_list].index.values
        # temp = temp[temp['ID'].isin(pid_list)].drop_duplicates('ID')[['ID', 'VISCODE', 'DX']].values
        # np.savetxt('./data/ADNI_HAVE_IMG_AND_DX_20221020.txt', temp,
        #            fmt='%s')

        if len(ids_include):
            info = info[info['ID'].isin(ids_include)]
        if len(ids_exclude):
            info = info[~info['ID'].isin(ids_exclude)]

        # entry level selection
        info['VISIT'] = info['VISIT'].values.astype(int)
        info = info.sort_values(['ID', 'VISIT'])
        _group = info['GROUP'].values.copy()
        _group[_group < 0] = 9
        info['stratify'] = _group + sum([(~info[m].isnull()).astype(int).values * 10 ** (i + 1)
                                         for i, m in enumerate(modals)])
        if 'COHORT' in info.columns:
            info['stratify'] += info['COHORT'].astype('category').cat.codes.astype(int) * 10 ** (len(modals) + 1)

        # *note*: if a subject got several DX in diff time point, then only use the earliest DX (should be sorted first)
        # this avoid using the same subject in different clfsettings.
        info['DX_LABEL'] = self.dx_mapping(info['DX'].values, clfsetting)
        for i in np.sort(info['ID'].unique()):  # using -1 to represent the unavailable dx
            dx = info.loc[info['ID'] == i, 'DX_LABEL'].values
            dx = dx[~np.isnan(dx)]
            if dx[0] == -1:
                info.loc[info['ID'] == i, 'DX_LABEL'] = -1

        labels = self.get_label(info, dsettings)
        inx = ~np.isnan(labels).all(axis=-1).all(axis=-1)
        inx &= labels[:, -1, -1] != -1  # exclusion for dx == -1
        labels = labels[inx]
        info = info[inx]

        # datasplit inx, no id overlap between different splits
        self.inx_list = self.data_split(stratify_var=info['stratify'].values,
                                        id_info=info['ID'].values, kfold=10, seed=seed)
        # bl inx
        self.bl_binx = ~info.duplicated('ID').values
        # modal inx
        self.allmod_binx = ~info[self.modals].isnull().any(axis=1).values
        self.onemod_binx = ~info[self.modals].isnull().all(axis=1).values

        # set to nan array for the elements of missing modal
        for modal in self.modals[1:]:
            nan_ent = info[~info[modal].isnull()][modal].values[0] * np.nan
            modal_data = info[modal].values.copy()
            for i in range(len(modal_data)):
                ent = modal_data[i]
                if isinstance(ent, float):
                    assert np.isnan(ent)
                    modal_data[i] = nan_ent.copy()
            info[modal] = modal_data

        self._info = info
        self._labels = labels
        self.setup_inx_data(subset, only_bl, strict_input, fold=None, kfold=None)

    def setup_inx_data(self, subset: list, only_bl, strict_input, fold=None, kfold=None):
        info = self._info.copy()
        labels = self._labels.copy()

        assert VAL_R == TEST_R == 0.2
        assert len(self.inx_list) == 10
        if fold is None and kfold is None:
            fold = 0
            s = 2  # size of testing set
        elif fold is not None and kfold is not None:
            assert kfold in [5, 10]
            assert 0 <= fold < kfold
            s = 10 // kfold  # size of testing set
        else:
            raise NotImplementedError

        all_inx_inx = list(range(10))
        test_inx_inx = list(range(fold * s, (fold + 1) * s))
        val_inx_inx = [(i + 10) % 10 for i in range(fold * s - 2, fold * s)]  # transform to positive
        train_inx_inx = list(set(all_inx_inx) - set(val_inx_inx) - set(test_inx_inx))

        inx = []
        if 'training' in subset:
            inx = np.append(inx, np.concatenate(self.inx_list[train_inx_inx]))
        if 'validation' in subset:
            inx = np.append(inx, np.concatenate(self.inx_list[val_inx_inx]))  # 20% for validation
        if 'testing' in subset:
            inx = np.append(inx, np.concatenate(self.inx_list[test_inx_inx]))
        inx = inx.astype(int)

        if only_bl:
            inx_keep = self.bl_binx.copy() & (~info['ID'].isnull().values)
        else:
            inx_keep = ~info['ID'].isnull().values
        if strict_input:
            inx_keep &= self.allmod_binx
        else:
            inx_keep &= self.onemod_binx  # at least have one modal of data

        if 'training' in subset and TRAINDATA_RATIO < 1:
            logging.warning(f'TRAINDATA_RATIO: {TRAINDATA_RATIO}')
            np.random.seed(1234)
            keep_ids = info[inx_keep]['ID'].drop_duplicates().values
            keep_ids = np.random.choice(keep_ids, size=int(np.ceil(len(keep_ids) * TRAINDATA_RATIO)), replace=False)
            inx_keep &= info['ID'].isin(keep_ids).values
        inx = list(filter(lambda x: x in np.where(inx_keep)[0], inx))

        self.labels = labels[inx]  # shape for classification setting: (n, 1, 1)
        self.dis_label = info['GROUP'].values[inx]  # shape : (n,)
        self.id_info = info['ID'].values[inx]  # shape : (n)
        self.namelist = info[self.modals[0]].values[inx]  # shape : (n,)

        self.aux_labels = np.array([info[modal].values[inx] for modal in self.modals[1:]]) if self.modals[1:] else None
        # shape : (n, k, *), k: len(modals) - 1,
        for m in self.modals[1:]:
            del info[m]

        self.info = info.iloc[inx]  # shape : (n, c)
        self.data = [None for i in self.namelist]
        return self

    def resetup_inx_data(self, *args, **kwargs):
        return self.setup_inx_data(*args, **kwargs)

    def empty_cache(self):
        self.data = [None for i in self.namelist]

    def get_table(self, path, modals: Union[list, str], dsettings: Union[None, dict]) -> pd.DataFrame:
        '''return the pd.DataFrame contains all needed information, which should at least include following columns:
            ['DX', 'ID', 'VISIT'] + modals

        Args:
            path: root path for the dataset
            modals: a list of modalities to use. length should be large than 1,
                    first element must be an image modal, e.g. mwp1
            dsettings: specific settings for the dataset

        Returns:
            A pd.DataFrame contains the infomation for the dataset.
            The columns at least include ['DX', 'ID', 'VISIT'] + modals.
            For image modality, the element of the column should be strings of the image pathes. For other modalities

        '''
        raise NotImplementedError

    def _filtering(self, info: pd.DataFrame) -> Union[list, tuple, set]:
        '''filtering for each dataset
        will be called after getting the information table using the method get_table.

        Args:
            info:  pd.DataFrame, contains DX, ID, VISIT columns

        Returns:
            a list for ids that should be included in the dataset
        '''
        if self.clfsetting in ['sMCI-pMCI', 'CN-AD']:
            # delete pMCI/AD -> sMCI/CN, AD->MCI
            p_rid = []
            _info = info.copy()[['ID', 'VISIT', 'DX']]
            _info = _info[~pd.isnull(_info['DX'])]
            _info = _info[_info['DX'].isin(['CN', 'NC', 'sMCI', 'MCI', 'pMCI', 'AD'])]
            _info['DX'] = _info['DX'].apply(
                func=lambda x: {'CN': 0, 'NC': 0, 'sMCI': 0, 'MCI': 0, 'pMCI': 1, 'AD': 2}[x])

            _info = _info.sort_values(by=['ID', 'VISIT'], kind='mergesort')
            _info = _info[_info['DX'] != -1]
            for i in _info['ID'].drop_duplicates().values:
                temp = _info[_info['ID'] == i]
                if not (temp.sort_values(by=['VISIT'], kind='mergesort')['DX'].values ==
                        temp.sort_values(by=['DX'], kind='mergesort')['DX'].values).all():
                    p_rid.append(i)
            ids = set(info['ID'].values) - set(p_rid)

            # delete NC/MCI age < 60 for the last visit
            # because the onset age of AD is after 60 in most cases
            # see: C. L. Masters et al., Nature Reviews Disease Primers. 1, 1–18 (2015).
            df = info[info['DX'].isin(['CN', 'sMCI'])]
            df['AGE'] = df['AGE'].astype(float)
            df = df.groupby('ID')['AGE'].max()
            ids = ids - set(df[df < 60].index.values)

            # delete sMCI subjects with follow ups of less one year
            dx_vis = info[info['DX'].isin(['sMCI'])].sort_values(['ID', 'VISIT']).drop_duplicates('ID')
            # last visit of sMCI subjects
            last_vis = info[info['ID'].isin(dx_vis['ID'].values) & (~info['DX'].isnull())]
            last_vis = last_vis.sort_values(['ID', 'VISIT'], ascending=[True, False]).drop_duplicates('ID')
            assert (last_vis['ID'].values == dx_vis['ID'].values).all()
            dx_vis['fup_months'] = (last_vis['DATE'].values.astype('datetime64[M]') -
                                    dx_vis['DATE'].values.astype('datetime64[M]')).astype(int)
            ids = ids - set(dx_vis[dx_vis['fup_months'] < 12]['ID'].values)

            ## exclude the apoe4 not consistent with the officially provideds
            if 'APOE_CONSISTENT' in info.columns:
                ids = ids - set(info[~info['APOE_CONSISTENT']]['ID'].values)

            # entry level filtering
            ## delete CN visit for MCI subjects
            mci_ids = info[info['DX'].isin(['sMCI', 'pMCI'])]['ID'].drop_duplicates().values
            info = info[~(info['ID'].isin(mci_ids) & (info['DX'] == 'CN'))]

            # delte CN visit for AD subjects
            ad_ids = info[info['DX'] == 'AD']['ID'].drop_duplicates().values
            info = info[~(info['ID'].isin(ad_ids) & (info['DX'] == 'CN'))]

            info = info[info['ID'].isin(ids)]
        else:
            raise NotImplementedError
        return info

    @staticmethod
    def dx_mapping(dx: Iterable, clfsetting: str) -> np.ndarray:
        '''mapping DX to int, where -1 is for excluding.
        For example, when clfsetting is CN-AD, the sMCI, pMCI and nan are mapped to -1.

        Args:
            dx: shape: (n,), str type diagnoses
            clfsetting: classification setting

        Returns:
            np.ndarray with shape of (n,) and dtype of int
        '''
        raise NotImplementedError

    @staticmethod
    def g_mapping(dx: Iterable):
        '''mapping str diagnosis to global int code

        Args:
            dx: shape: (n, ), str type diagnoses

        Returns:
            np.ndarray with shape of (n,) and dtype of int
        '''
        dis_map = {np.nan: -1, '': -1, 'nan': -1, 'Other': -1,
                   'CN': 0, 'NC': 0, 'No_Known_Disorder': 0,
                   'sMCI': 1, 'pMCI': 2, 'AD': 3,
                   'MCI': 4, 'SCS': 5, 'SCD': 6,
                   'Bipolar_Disorder': 7, 'Schizoaffective': 8, 'Schizophrenia_Strict': 9,
                   }
        gcode = np.array(list(map(lambda x: dis_map[x], dx)))
        return gcode

    @staticmethod
    def get_label(info: pd.DataFrame, dsettings: dict) -> np.ndarray:
        '''return labels for training

        Args:
            info:
            dsettings:

        Returns:
            labels: Has a shape of (n, 1, 1) for classification.
        '''
        if dsettings == {}:
            labels = info['DX_LABEL'].values.reshape([-1, 1, 1])
        else:
            raise NotImplementedError

        return labels

    @staticmethod
    def data_split(stratify_var: np.ndarray, id_info: np.ndarray, kfold: int, seed: int):
        '''return the index for training set, validation set, and testing set,
        where each subset do not have subject level overlapping

        Args:
            stratify_var: int type, shape: (n,). Data is split in a stratified fashion, using this as
        the class labels.

            id_info: shape: (n,) the id for each subject
            seed: random seed for reproduction

        Returns:
            train_inx
            val_inx
            test_inx
        '''
        inx_list = split_nooverlap(data_size=len(id_info), kfold=kfold, id_info=id_info,
                                   seed=seed, stratify=stratify_var)
        return inx_list

    @staticmethod
    def check_imgpath(subj_path, modal):
        if modal in ['mwp1']:
            if os.path.exists(os.path.join(subj_path, 'report', 'catreport_T1w.pdf')):
                return os.path.join(subj_path, 'mri', modal + 'T1w.nii')
            else:
                return None
        else:
            raise NotImplementedError

    @staticmethod
    def load_data(name, smooth=False) -> np.ndarray:
        '''load MRI data from disk

        Args:
            img_path:
            name:
            smooth: if Ture, Smooth MRI images by applying a Gaussian filter with kernel size of 8.

        Returns:
            ori_img: the MRI data matrix in which nan is set to 0.
        '''
        if (isinstance(name, float) and np.isnan(name)) or len(name) == 0:
            return np.zeros(IMG_SIZE, dtype=np.float32) * np.nan
        else:
            if 'NATIVE_GM_' in name:
                dir, file = os.path.split(name)
                file = file.replace('NATIVE_GM_', '')
                ori_img = nib.load(os.path.join(dir, '../', file))
                brain_label = nib.load(os.path.join(dir, 'p0' + file)).get_fdata()
            elif 'NATIVE_WM_' in name:
                dir, file = os.path.split(name)
                file = file.replace('NATIVE_GM_', '')
                ori_img = nib.load(os.path.join(dir, '../', file))
                brain_label = nib.load(os.path.join(dir, 'p0' + file)).get_fdata()
            else:
                ori_img = nib.load(name)

            if smooth:
                ori_img = smooth_img(ori_img, 4).get_fdata()
            else:
                ori_img = ori_img.get_fdata()

            if 'NATIVE_GM_' in name:
                ori_img[brain_label < 1.5] = 0
                ori_img[brain_label >= 2.5] = 0
            elif 'NATIVE_WM_' in name:
                ori_img[brain_label < 2.5] = 0

            ori_img[np.isnan(ori_img)] = 0
            ori_img = np.array(ori_img, dtype=np.float32)
            return ori_img

    def __getitem__(self, index):
        if self.preload:
            if self.data[index] is not None:
                bat_data = self.data[index]
            else:
                name = self.namelist[index]
                if self.maskmat is None:
                    bat_data = (self.load_data(name, self.smooth))
                else:
                    bat_data = (self.load_data(name, self.smooth) * self.maskmat)
                self.data[index] = bat_data
        else:
            name = self.namelist[index]
            if self.maskmat is None:
                bat_data = (self.load_data(name, self.smooth))
            else:
                bat_data = (self.load_data(name, self.smooth) * self.maskmat)

        bat_data = torch.from_numpy(bat_data).unsqueeze(0).unsqueeze(0)  # channel, batch
        bat_labels = torch.Tensor(self.labels[index: index + 1])
        if self.aux_labels is not None:
            try:
                bat_aux_label = torch.cat([torch.Tensor(self.aux_labels[i][index])
                                           for i in range(self.aux_labels.shape[0])], dim=-1)
                bat_aux_label = bat_aux_label.unsqueeze(0)
                #
            except Exception as e:
                print('aux_label loading error: %s' % e)
                raise NotImplementedError
        else:
            bat_aux_label = bat_labels * torch.from_numpy(np.array([np.nan]))

        if self.dis_label is not None:
            bat_dis_label = torch.Tensor([self.dis_label[index]])
        else:
            bat_dis_label = torch.from_numpy(np.array([np.nan]))

        return bat_data, bat_labels, bat_aux_label, bat_dis_label

    def __add__(self, other):
        assert isinstance(other, Dataset)
        assert self.preload == other.preload
        assert self.smooth == other.smooth
        assert self.maskmat == other.maskmat
        assert self.clfsetting == other.clfsetting
        assert type(self.aux_labels) == type(other.aux_labels)
        assert self.modals == other.modals
        assert self.strict_input == other.strict_input

        new_ins = Dataset.__new__(Dataset)
        new_ins.preload = self.preload
        new_ins.smooth = self.smooth
        new_ins.maskmat = self.maskmat
        new_ins.clfsetting = self.clfsetting
        new_ins.modals = self.modals
        new_ins.strict_input = self.strict_input

        new_ins._info = self._info.append(other._info)
        new_ins._labels = np.concatenate([self._labels, other._labels], axis=0)
        new_ins.inx_list = np.array([i + [k + len(self._info) for k in j]
                                     for i, j in zip(self.inx_list, other.inx_list)], dtype=object)
        for _ in new_ins.inx_list:
            assert len(_) == len(set(_))

        new_ins.bl_binx = np.concatenate([self.bl_binx, other.bl_binx])
        new_ins.onemod_binx = np.concatenate([self.onemod_binx, other.onemod_binx])
        new_ins.allmod_binx = np.concatenate([self.allmod_binx, other.allmod_binx])

        new_ins.labels = np.concatenate([self.labels, other.labels], axis=0)
        new_ins.info = self.info.append(other.info)
        new_ins.dis_label = np.append(self.dis_label, other.dis_label, axis=0)
        new_ins.namelist = np.append(self.namelist, other.namelist, axis=0)
        new_ins.id_info = np.append(self.id_info, other.id_info, axis=0)

        if self.aux_labels is None:
            new_ins.aux_labels = None
        else:
            new_ins.aux_labels = np.concatenate([self.aux_labels, other.aux_labels], axis=1)

        new_ins.data = self.data + other.data

        return new_ins

    def __len__(self):
        return len(self.id_info)


class Patch_Data(data.Dataset):
    def __init__(self, imgdata, patch_size, center_mat, shift, flip_axis, resample_patch=None):
        self.patch_size = patch_size
        self.center_mat = center_mat
        self.shift = shift
        self.flip_axis = flip_axis
        self.imgdata = imgdata
        self.labels = imgdata.labels
        self.resample_patch = resample_patch

    def __getitem__(self, index):
        bat_data, bat_labels, bat_aux_label, bat_dis_label = self.imgdata[index]
        inputs, aux_labels, labels, dis_label = batch_sampling(imgs=bat_data, labels=bat_labels,
                                                               center_mat=self.center_mat,
                                                               aux_labels=bat_aux_label, dis_labels=bat_dis_label,
                                                               patch_size=self.patch_size,
                                                               random=False, shift=self.shift,
                                                               flip_axis=self.flip_axis,
                                                               )

        if self.resample_patch is not None:
            assert len(self.resample_patch) == 3
            _inputs = inputs.numpy()
            resam = np.zeros(list(inputs.shape[:-3]) + self.resample_patch)
            dsfactor = [w / float(f) for w, f in zip(self.resample_patch, _inputs.shape[-3:])]
            for i in range(_inputs.shape[0]):
                resam[i, 0, :] = nd.interpolation.zoom(_inputs[i, 0, :], zoom=dsfactor)
            inputs = torch.from_numpy(resam)

        if self.center_mat.shape[1] == 1:
            return inputs.squeeze(0), aux_labels.squeeze(0), labels.squeeze(0), dis_label.squeeze(0)
        else:
            return inputs, aux_labels, labels, dis_label

    def __len__(self):
        return len(self.imgdata)


def split_nooverlap(data_size, kfold, id_info, seed=None, stratify=None):
    '''

    Args:
        data_size:
        kfold:
        id_info: the id list, can be duplicated.
        seed:
        stratify:

    Returns:

    '''
    inxes = np.arange(data_size)

    if stratify is not None:
        df = pd.DataFrame([id_info.ravel(), stratify.ravel()]).transpose()
        df.columns = ['id', 'stf']
        ids_set, stratify = df.drop_duplicates('id').values.T
        sort_inx = ids_set.argsort()
        stratify = stratify.astype(np.int)[sort_inx]
        ids_set = ids_set[sort_inx]
    else:
        ids_set = np.unique(id_info)
        ids_set = np.sort(ids_set)  # remove randomness induced by memory issue

    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
    ids_splits = [ids_set[j] for i, j in skf.split(ids_set, stratify)]

    ent_splits = []
    for ids_s in ids_splits:
        ent_s = []
        for id in ids_s:
            ent_s += (inxes[id_info == id]).tolist()
        ent_splits.append(ent_s)
    ent_splits = np.array(ent_splits)
    return ent_splits


class ADNI(Dataset):
    path = ADNI_PATH
    data_tables = {}

    def __init__(self, subset, clfsetting, modals, no_smooth=False, only_bl=False,
                 ids_exclude=(), ids_include=(), maskmat=None, seed=1234, preload=True,
                 dsettings=None, strict_input=True):

        if dsettings is None:
            dsettings = {'cohort': 'ALL', 'label_names': ['DX'], 'final_dx': False}
        else:
            assert 'cohort' in dsettings.keys()
            assert 'label_names' in dsettings.keys()
            assert 'final_dx' in dsettings.keys()
        if dsettings['final_dx']:
            if clfsetting not in ['CN-AD']:
                raise NotImplementedError

        image_path = ADNI_PATH
        super(ADNI, self).__init__(self.path, subset, clfsetting, modals, no_smooth,
                                   only_bl, ids_exclude, ids_include, maskmat, seed, preload, dsettings, strict_input)

    @classmethod
    def get_info(cls):
        if 'demo' not in cls.data_tables.keys():
            info = pd.read_csv(os.path.join(BASEDIR, 'data/ADNI/ADNIMERGE_28Jun2023.csv'), dtype=str)
            info = info[['RID', 'VISCODE', 'COLPROT', 'EXAMDATE', ]]
            cls.data_tables['demo'] = info
        else:
            info = cls.data_tables['demo']
        return info

    @classmethod
    def get_aux(cls):
        if 'aux' not in cls.data_tables.keys():
            info = pd.read_csv(os.path.join(BASEDIR, 'data/ADNI/ADNIMERGE_28Jun2023.csv'), dtype=str)
            demo_info = pd.read_csv(os.path.join(BASEDIR, 'data/ADNI/PTDEMOG_28Jun2023.csv'), dtype=str)
            # add birth Year Month
            info = pd.merge(how='left', left=info, right=demo_info[['RID', 'PTDOBYY']].drop_duplicates().dropna(),
                            on=['RID'])
            info['PTGENDER'] = info['PTGENDER'].apply(lambda x: {'Male': 1, 'Female': 2}[x])
            info['AGE'] = (pd.to_datetime(info['EXAMDATE']).dt.year - pd.to_datetime(info['PTDOBYY']).dt.year)
            info.rename(columns={'EXAMDATE': 'EXAMDATE_AUX'}, inplace=True)
            info = info[['RID', 'VISCODE', 'EXAMDATE_AUX',
                         'AGE', 'PTGENDER', 'APOE4', 'mPACCdigit', 'ADASQ4', 'Hippocampus', 'MMSE']]
            info['AGE'] = info['AGE'].astype(float)
            info['ADASQ4'] = info['ADASQ4'].astype(float)

            cls.data_tables['aux'] = info
        else:
            info = cls.data_tables['aux']
        return info

    @classmethod
    def get_dx(cls):
        if 'dx' not in cls.data_tables.keys():
            info = pd.read_csv(os.path.join(BASEDIR, 'data/ADNI/ADNIMERGE_28Jun2023.csv'), dtype=str)

            info['NEW_DX'] = info[pd.isnull(info['DX'])]['DX'].drop_duplicates()  # set to null

            info.loc[info['DX_bl'] == 'EMCI', 'DX_bl'] = 'MCI'
            info.loc[info['DX_bl'] == 'LMCI', 'DX_bl'] = 'MCI'
            info.loc[info['DX_bl'] == 'SMC', 'DX_bl'] = 'CN'  # it was officially treated as CN in DX
            info.loc[info['DX'] == 'Dementia', 'DX'] = 'AD'
            # some blank DX in bl is filled with DX_bl
            info.loc[(info['VISCODE'] == 'bl')
                     & pd.isnull(info['DX']), 'DX'] = info.loc[
                (info['VISCODE'] == 'bl') & pd.isnull(info['DX']), 'DX_bl']

            info = info[['RID', 'VISCODE', 'DX', 'NEW_DX']]
            # define sMCI, pMCI
            for i in info.index:
                if info.loc[i]['DX'] == 'CN':
                    info.loc[i, 'NEW_DX'] = 'CN'
                elif info.loc[i]['DX'] == 'AD':
                    info.loc[i, 'NEW_DX'] = 'AD'
                elif info.loc[i]['DX'] == 'MCI':
                    rid = info.loc[i]['RID']
                    dxs = info[info['RID'] == rid]['DX'].values
                    if 'AD' in dxs:
                        info.loc[i, 'NEW_DX'] = 'pMCI'
                    else:
                        info.loc[i, 'NEW_DX'] = 'sMCI'

            info = info[['RID', 'VISCODE', 'NEW_DX']]

            cls.data_tables['dx'] = info
        else:
            info = cls.data_tables['dx']
        return info

    @classmethod
    def get_snp(cls, snp_modal, path):
        if snp_modal not in cls.data_tables.keys():
            snpseq, tissue, tp = snp_modal.split('.')
            eqtl = pd.read_csv(os.path.join(eQTL_PATH, '%s.eQTL_2_gene.%s.txt' % (tissue, tp)),
                               sep='\t')
            eqtl.columns = ['snp', 'gene']
            eqtl_snps = eqtl['snp'].drop_duplicates().values
            eqtl_genes = eqtl['gene'].drop_duplicates().values

            if snpseq == 'array_snp':
                plink_df1, ori_snps1 = get_array_snp(path=ADNI_GENETIC_PATH,
                                                     cohort='ADNI_1', eqtl_snps=eqtl_snps)
                plink_df2, ori_snps2 = get_array_snp(path=ADNI_GENETIC_PATH,
                                                     cohort='ADNI_2', eqtl_snps=eqtl_snps)
                plink_df3, ori_snps3 = get_array_snp(path=ADNI_GENETIC_PATH,
                                                     cohort='ADNI_3', eqtl_snps=eqtl_snps)
                plink_dfgo, ori_snpsgo = get_array_snp(path=ADNI_GENETIC_PATH,
                                                       cohort='ADNI_GO', eqtl_snps=eqtl_snps)

                assert ((plink_df1.columns == plink_df2.columns) &
                        (plink_df1.columns == plink_df3.columns) &
                        (plink_df1.columns == plink_dfgo.columns)).all()
                plink_df = pd.concat([plink_df1, plink_df2, plink_df3, plink_dfgo], ignore_index=False)

                assert plink_df.index.duplicated().sum() == 0

            else:
                raise NotImplementedError

            plink_df = plink_df.fillna(0)
            plink_snps = plink_df.columns.values
            snp_df = pd.DataFrame(index=plink_df.index, columns=[snp_modal], dtype=object)
            snp_df.index.name = 'RID'
            snp_df[snp_modal] = [plink_df.loc[i].values for i in plink_df.index]

            cls.data_tables[snp_modal] = snp_df
        else:
            snp_df = cls.data_tables[snp_modal]

        return snp_df

    @staticmethod
    def dx_mapping(dx, clfsetting):
        if clfsetting == 'CN-AD':
            dx_label = np.array(list(map(lambda x: {'CN': 0, 'sMCI': -1,
                                                    'pMCI': -1, 'MCI': -1, 'AD': 1, np.nan: -1}[x], dx)))
        elif clfsetting == 'CN_sMCI-pMCI_AD':
            dx_label = np.array(list(map(lambda x: {'CN': 0, 'sMCI': 0, 'pMCI': 1, 'AD': 1, np.nan: -1}[x], dx)))
        elif clfsetting == 'sMCI-pMCI':
            dx_label = np.array(list(map(lambda x: {'CN': -1, 'sMCI': 0, 'pMCI': 1, 'AD': -1, np.nan: -1}[x], dx)))
        elif clfsetting == 'DIS-NC':
            dx_label = np.array(list(map(lambda x: {'CN': 0, 'sMCI': -1, 'pMCI': -1, 'AD': 1, np.nan: -1}[x], dx)))
        elif clfsetting == 'regression':
            raise NotImplementedError
            # dx_label = np.array(list(map(lambda x: {'CN': 0, 'sMCI': 1, 'pMCI': 2, 'AD': 3, np.nan: 4}[x], dx)))
        else:
            raise NotImplementedError
        return dx_label

    def get_table(self, path, modals: Union[list, str], dsettings):
        if not isinstance(modals, list):
            modals = [modals]

        cohort = dsettings['cohort']
        label_names = dsettings['label_names']
        snp_modal = list(filter(lambda x: 'snp.' in x, modals))
        assert cohort in ['ADNI1', 'ADNI2', 'ADNI1-P', 'ADNI2-P', 'PET', 'NO-PET', 'ALL']
        assert len(snp_modal) <= 1
        assert len(modals) <= 2  # img, [snp]

        info = self.get_info()
        AUX_INFO = self.get_aux()
        DX_INFO = self.get_dx()

        info['IMAGE'] = info[['RID', 'VISCODE']].apply(axis=1, func=lambda x: os.path.exists(
            os.path.join(path, x['RID'], 'report', 'catreport_' + x['VISCODE'] + '.pdf')))
        info.loc[info['IMAGE'], modals[0]] = info.loc[info['IMAGE'], ['RID', 'VISCODE']].apply(
            axis=1, func=lambda x: os.path.join(path, x['RID'], 'mri', modals[0] + x['VISCODE'] + '.nii'))

        for df in [DX_INFO, AUX_INFO]:
            info = info.merge(df, how='left', on=['RID', 'VISCODE'], )

        # load snp
        if len(snp_modal):
            SNP_INFO = self.get_snp(snp_modal[0], path)
            info = info.merge(SNP_INFO, how='left', on=['RID'], )

        info['VISIT'] = info['VISCODE'].apply(lambda x: 0 if x == 'bl' else int(x.replace('m', '')))

        # to garantee no overlapping
        adni1_subs = info[info['COLPROT'] == 'ADNI1']['RID'].drop_duplicates().values
        adni2_subs = info[info['COLPROT'] == 'ADNI2']['RID'].drop_duplicates().values

        inx = ~pd.isnull(info).all(axis=1)
        if cohort in ['ADNI1', 'ADNI2']:
            if cohort == 'ADNI1':
                inx &= info['RID'].isin(adni1_subs)
            else:
                inx &= info['RID'].isin(adni2_subs) & (~info['RID'].isin(adni1_subs))
        elif cohort == 'ALL':
            pass
        else:
            raise NotImplementedError

        info = info.loc[inx]

        if dsettings['final_dx'] == True:
            info['final_dx_filter'] = True

            temp = info.loc[info['NEW_DX'] == 'AD', ['RID', 'VISIT']]
            temp = temp.sort_values(['RID', 'VISIT'])
            temp = temp.drop_duplicates('RID')

            for rid, vis in temp.values:
                info.loc[(info['RID'] == rid) & (info['VISIT'] < vis), 'final_dx_filter'] = False
            info = info[info['final_dx_filter']]
            del info['final_dx_filter']

        info = info.rename(columns={'RID': 'ID', 'NEW_DX': 'DX', 'PTGENDER': 'GENDER'})

        # exclude the samples APOE4 not consistent with the officially provided
        ex_ids = []
        dx = info[~info['DX'].isna()].drop_duplicates('ID')
        for i in [1, 2, 3, 'GO']:
            path = os.path.join(BASEDIR, 'data/ADNI/ADNI_%s_APOE.csv' % i)
            ap = pd.read_csv(path, index_col=None, dtype=str)
            ap = ap.merge(right=dx[['ID', 'DX', 'APOE4']], left_on='0', right_on='ID')
            ap = ap[~ap['APOE4'].isnull()]
            ex_ids = np.append(ex_ids, ap[ap['APOE4'] != ap['Num_e4']]['ID'].drop_duplicates().values)
        info['APOE_CONSISTENT'] = True
        info.loc[info['ID'].isin(ex_ids), 'APOE_CONSISTENT'] = False
        info['DATE'] = np.array(info['VISIT'].values, dtype='datetime64[M]')
        info['COHORT'] = info['COLPROT']

        return info

    @staticmethod
    def get_label(info, dsettings):
        '''return labels for training and groups of subjects
        '''
        cohort = dsettings['cohort']
        label_names = dsettings['label_names']

        AUX_LABEL = info[['AGE', 'GENDER', 'APOE4', 'mPACCdigit', 'ADASQ4', 'Hippocampus']].astype(float).values

        DX = info['DX_LABEL'].values
        LABEL = []
        DX = DX.reshape(-1, 1)
        if 'DX' in label_names:
            LABEL.append(DX)
            assert len(label_names) == 1
        if 'ADASQ4' in label_names:
            LABEL.append(info['ADASQ4'].astype(float).values.reshape([-1, 1]))
        if 'mPACCdigit' in label_names:
            LABEL.append(info['mPACCdigit'].astype(float).values.reshape([-1, 1]))
        LABEL = np.array(list(zip(*LABEL)))
        return LABEL


def get_data(dataset, clfsetting, modals, no_smooth, seed, only_bl, strict_input):
    data_train = None
    data_val = None
    data_test = None
    if dataset == 'ADNI_DX':
        # mix ADNI1/2/3/GO
        data_train = ADNI(subset=['training'], clfsetting=clfsetting, modals=modals,
                          no_smooth=no_smooth, seed=seed, only_bl=only_bl, strict_input=strict_input,
                          dsettings={'cohort': 'ALL', 'label_names': ['DX'], 'final_dx': False})

    else:
        raise NotImplementedError

    # some datasets are only used as testing sets
    if data_val is None and data_train is not None:
        data_val = copy.copy(data_train).resetup_inx_data(subset=['validation'], only_bl=True, strict_input=True)
    if data_test is None and data_train is not None:
        data_test = copy.copy(data_train).resetup_inx_data(subset=['testing'], only_bl=True, strict_input=True)
    return data_train, data_val, data_test


def get_dataset(dataset, clfsetting, modals, patch_size, batch_size, center_mat, flip_axises, no_smooth, no_shuffle,
                no_shift, n_threads, seed=1234, resample_patch=None, trtype='single', only_bl=True, strict_input=False):
    data_train, data_val, data_test = get_data(dataset, clfsetting, modals, no_smooth, seed, only_bl, strict_input)
    # verification
    # if not the datasets only for testing
    if TRAINDATA_RATIO > 0.999:
        checkpath = os.path.join(BASEDIR, 'utils', 'datacheck',
                                 dataset + '_' + clfsetting + '_' + '_'.join(modals) + '.pkl')
        if os.path.exists(checkpath):
            with open(checkpath, 'rb') as f:
                id_infos = pkl.load(f)
            assert (np.sort(np.unique(id_infos[0])) == np.sort(np.unique(data_train.id_info))).all()
            assert (np.sort(np.unique(id_infos[1])) == np.sort(np.unique(data_val.id_info))).all()
            assert (np.sort(np.unique(id_infos[2])) == np.sort(np.unique(data_test.id_info))).all()
        else:
            if not os.path.exists(os.path.join(BASEDIR, 'utils', 'datacheck')):
                os.mkdir(os.path.join(BASEDIR, 'utils', 'datacheck'))
            with open(checkpath, 'wb') as f:
                pkl.dump([data_train.id_info,
                          data_val.id_info,
                          data_test.id_info], f)
        a = set(data_train.id_info)
        b = set(data_val.id_info)
        c = set(data_test.id_info)
        assert len(a.intersection(b)) == len(a.intersection(c)) == len(b.intersection(c)) == 0

    if trtype == 'single':
        datalist = [[data_train, data_val, data_test]]
    elif trtype == '5-rep':
        datalist = [[data_train, data_val, data_test] for i in range(5)]
    elif trtype == '5cv':
        datalist = [[copy.copy(data_train).setup_inx_data(subset=['training'], only_bl=only_bl,
                                                          strict_input=data_train.strict_input, fold=i, kfold=5),
                     copy.copy(data_val).setup_inx_data(subset=['validation'], only_bl=True,
                                                        strict_input=True, fold=i, kfold=5),
                     copy.copy(data_test).setup_inx_data(subset=['testing'], only_bl=True,
                                                         strict_input=True, fold=i, kfold=5)]
                    for i in range(5)]
    elif trtype == '10cv':
        raise NotImplementedError
        datalist = [[copy.copy(data_train).setup_inx_data(subset=['training'], only_bl=only_bl,
                                                          strict_input=data_train.strict_input, fold=i, kfold=10),
                     copy.copy(data_val).setup_inx_data(subset=['validation'], only_bl=True,
                                                        strict_input=True, fold=i, kfold=10),
                     copy.copy(data_test).setup_inx_data(subset=['testing'], only_bl=True,
                                                         strict_input=True, fold=i, kfold=10)]
                    for i in range(10)]
    else:
        raise NotImplementedError

    dataloader_list = []
    for data_train, data_val, data_test in datalist:
        if data_train:
            logging.warning('Training data size: %d' % len(data_train))
            data_train = Patch_Data(data_train, patch_size=patch_size, center_mat=center_mat,
                                    shift=not no_shift, flip_axis=flip_axises, resample_patch=resample_patch)
            data_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=not no_shuffle,
                                                     num_workers=n_threads, pin_memory=True)
        if data_val:
            logging.warning('Validation data size: %d' % len(data_val))
            data_val = Patch_Data(data_val, patch_size=patch_size, center_mat=center_mat,
                                  shift=False, flip_axis=None, resample_patch=resample_patch)
            data_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False,
                                                   num_workers=n_threads, pin_memory=True)
        data_test = Patch_Data(data_test, patch_size=patch_size, center_mat=center_mat,
                               shift=False, flip_axis=None, resample_patch=resample_patch)
        data_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False,
                                                num_workers=n_threads, pin_memory=True)

        dataloader_list.append([data_train, data_val, data_test])

    if trtype in ['5cv', '10cv']:
        info_test = []
        info_val = []
        info_train = []
        for datas in dataloader_list:
            train, val, test = datas
            info_test = np.concatenate([info_test, test.dataset.imgdata.id_info])
            info_val = np.concatenate([info_val, val.dataset.imgdata.id_info])
            info_train = np.concatenate([info_train, train.dataset.imgdata.id_info])
        assert (np.sort(np.unique(info_test)) == np.sort(np.unique(info_val))).all()
    return dataloader_list
