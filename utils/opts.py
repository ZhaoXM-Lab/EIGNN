import os
import argparse
import pickle as pkl
import torch
import numpy as np
import copy
import logging
import pandas as pd

DTYPE = torch.float32
BASEDIR = os.path.dirname(os.path.dirname(__file__))
NUM_LMKS = 50

VAL_R = 0.2
TEST_R = 0.2
DATASEED = 0
## optimization
LR_PATIENCE = 10
LR_MUL_FACTOR = 0.5

MOMENTUM = 0.9
DAMPENING = 0.9
WEIGHT_DECAY = 1e-3

FULL_PATCH_SIZE = 117, 141, 117
IMG_SIZE = 121, 145, 121
FULL_CENTER_MAT = [[], [], []]
FULL_CENTER_MAT[0].append(int(np.floor((IMG_SIZE[0] - 1) / 2.0)))
FULL_CENTER_MAT[1].append(int(np.floor((IMG_SIZE[1] - 1) / 2.0)))
FULL_CENTER_MAT[2].append(int(np.floor((IMG_SIZE[2] - 1) / 2.0)))
FULL_CENTER_MAT = np.array(FULL_CENTER_MAT)

ADNI_PATH = '/data/datasets/ADNI/ADNI_T1'
ADNI_GENETIC_PATH = os.path.join(BASEDIR, 'data/ADNI/SNPArray')

eQTL_PATH = os.getenv('eQTL_PATH', os.path.join(BASEDIR, 'data/gtex_link.eqtl.plus_gwas_2021_2021_2022'))
logging.warning('eQTL path: %s' % eQTL_PATH)

GRAPH_PATH = os.getenv('GRAPH_PATH', os.path.join(eQTL_PATH, 'reactome_graph'))
logging.warning('Graph path: %s' % GRAPH_PATH)


def gen_center_mat(pat_size: list):
    center_mat = [[], [], []]
    for x in np.arange(pat_size[0] // 2, IMG_SIZE[0] // pat_size[0] * pat_size[0], pat_size[0]):
        for y in np.arange(pat_size[1] // 2, IMG_SIZE[1] // pat_size[1] * pat_size[1], pat_size[1]):
            for z in np.arange(pat_size[2] // 2, IMG_SIZE[2] // pat_size[2] * pat_size[2], pat_size[2]):
                center_mat[0].append(x + (IMG_SIZE[0] % pat_size[0]) // 2)
                center_mat[1].append(y + (IMG_SIZE[1] % pat_size[1]) // 2)
                center_mat[2].append(z + (IMG_SIZE[2] % pat_size[2]) // 2)
    center_mat = np.array(center_mat)
    return center_mat


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='ADNI_DX',
        type=str,
        help='Select dataset')

    parser.add_argument(
        '--pre_datasets',
        default=None,
        nargs='+',
        required=False,
        help='dataset list for prediction (Optional)')
    parser.add_argument(
        '--pre_testonly_datasets',
        default=None,
        nargs='+',
        required=False,
        help='dataset list for prediction (the datasets should be test-only) (Optional)')
    parser.add_argument(
        '--label_names',
        default=['FDG', 'AV45'],
        type=str,
        nargs='+',
        help='regression lables (FDG, AV45)')

    parser.add_argument(
        '--modals',
        default=['mwp1'],
        type=str,
        nargs='+',
        help='modalities of MRI (mwp1|NATIVE_GM_|wm)')

    parser.add_argument(
        '--clfsetting',
        default='CN-AD',
        type=str,
        help='classification setting (regression|CN-AD|CN_sMCI-pMCI_AD|sMCI-pMCI)')

    parser.add_argument(
        '--trtype',
        default='single',
        type=str,
        help='training type, single|5-rep')

    parser.add_argument(
        '--method',
        default='EIGNN',
        type=str,
        help=
        'choose a method.'
    )
    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help='Batch Size')
    parser.add_argument(
        '--patch_size',
        default=(25, 25, 25),
        type=int,
        nargs=3,
        help='patch size, only available for some methods')

    parser.add_argument(
        '--optimizer',
        default='adam',
        type=str,
        help=
        'Optimizer, adam|sgd')

    parser.add_argument(
        '--cuda_index',
        default=0,
        type=int,
        help='Specify the index of gpu')

    parser.add_argument(
        '--n_epochs',
        default=100,
        type=int,
        help='Number of total epochs to run')

    parser.add_argument(
        '--learning_rate',
        default=1e-4,
        type=float,
        help=
        'Initial learning rate (divided by factor while training by lr scheduler)')

    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--no_log',
        action='store_true',
        help='If true, tensorboard logging is not used.')
    parser.set_defaults(no_log=False)

    parser.add_argument(
        '--n_threads',
        default=8,
        type=int,
        help='Number of threads for multi-thread loading')

    parser.add_argument(
        '--pretrain_path',
        default=[],
        type=str,
        nargs='+',
        help='Pretrained model paths (.pth)')

    parser.add_argument(
        '--log_label',
        default='',
        type=str,
        help='additional label for logging')

    parser.add_argument(
        '--result_path',
        default=None,
        type=str,
        help='path for saving results')

    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)

    parser.add_argument(
        '--flip_axises',
        default=[0, 1, 2],
        type=int,
        nargs='+',
        help='flip axises (0, 1, 2)')

    parser.add_argument(
        '--no_smooth',
        action='store_true',
        help='no smooth apply to MRI')
    parser.set_defaults(no_smooth=False)

    parser.add_argument(
        '--no_shift',
        action='store_true',
        help='no shift apply to MRI for data augmentation')
    parser.set_defaults(no_shift=False)

    parser.add_argument(
        '--no_shuffle',
        action='store_true',
        help='no shuffle apply to batch sampling')
    parser.set_defaults(no_shuffle=False)

    parser.add_argument(
        '--method_para',
        default='{}',
        type=str,
        help='specify method parameters in dict form. eg: {"para1": values1}')

    args = parser.parse_args()
    return args


def mod_opt(method, opt):
    opt = copy.copy(opt)
    opt.mask_mat = None
    opt.resample_patch = None
    # Specify the hyper-parameters for each method

    if method == 'EIGNN':
        if len(opt.modals) > 1:
            _, tissue, pt = opt.modals[1].split('.')
        else:
            tissue, pt = 'Brain_All', 'gwas_p_5-e8'

        eqtl_file = os.path.join(eQTL_PATH, '%s.eQTL_2_gene.%s.txt' % (tissue, pt))
        assert os.path.exists(eqtl_file)

        opt.center_mat = gen_center_mat(opt.patch_size)
        opt.method_setting = {
            'num_patches': opt.center_mat.shape[1], 'patch_size': opt.patch_size, 'channels': 1,
            'img_dim': 32, 'num_classes': 1, 'depth': 4, 'dropout': 0.5, 'snp_dropout': 0.,
            'gene_dropout': 0.3, 'embed_size': 4, 'cross_int': False, 'gene_side_path': False,
            'useimg': True, 'usesnp': True, 'useeqtl': True, 'usegraph': True,
            'eqtl_path': eqtl_file, 'graph_dir': GRAPH_PATH}

    elif method == 'DAMIDL':
        with open('./utils/DLADLMKS_%d.pkl' % opt.patch_size[0], 'rb') as f:
            opt.center_mat, opt.patch_size, _ = pkl.load(f)

        opt.center_mat = opt.center_mat[:, :NUM_LMKS]
        if not opt.patch_size[0] == 25:
            opt.resample_patch = [25, 25, 25]
        # opt.optimizer = 'adam'
        # opt.learning_rate = 0.001
        opt.method_setting = {
            'patch_num': opt.center_mat.shape[1], 'feature_depth': [32, 64, 128, 128]
        }
    elif method == '3D-Mixer':
        opt.center_mat = gen_center_mat(opt.patch_size)

        # opt.optimizer = 'sgd'
        opt.method_setting = {
            'patch_size': opt.patch_size, 'num_patches': opt.center_mat.shape[1],
            'dim': 64, 'depth': 4, 'dropout': 0.1
        }
    elif method == 'NNP':
        if len(opt.modals) > 1:
            _, tissue, pt = opt.modals[1].split('.')
        else:
            tissue, pt = 'Brain_All', 'gwas_p_5-e8'

        eqtl_path = os.path.join(eQTL_PATH, '%s.eQTL_2_gene.%s.txt' % (tissue, pt))
        eqtls = pd.read_csv(eqtl_path, sep='\t')
        eqtls.columns = ['snp', 'gene']
        n_input = len(eqtls['snp'].drop_duplicates().values)  # number of snps

        opt.center_mat = gen_center_mat(opt.patch_size)
        opt.method_setting = {'n_input': n_input, 'num_classes': 1}
    elif method in ['RF', 'SVM', 'LR', 'PRS']:
        opt.center_mat = gen_center_mat(opt.patch_size)
        opt.n_epochs = 1
        opt.method_setting = {}

    elif method == 'MADDi':
        # MADDi multi-modal attention model configuration
        # Use single full patch like ResNet for direct 3D image access
        opt.center_mat = FULL_CENTER_MAT
        opt.patch_size = FULL_PATCH_SIZE

        # Set SNP dimension - should match the framework's SNP data
        if len(opt.modals) > 1:
            _, tissue, pt = opt.modals[1].split('.')
        else:
            raise NotImplementedError

        # Get SNP dimension from eQTL file for compatibility
        eqtl_path = os.path.join(eQTL_PATH, '%s.eQTL_2_gene.%s.txt' % (tissue, pt))
        eqtls = pd.read_csv(eqtl_path, sep='\t')
        eqtls.columns = ['snp', 'gene']
        snp_dim = len(eqtls['snp'].drop_duplicates().values)

        opt.method_setting = {
            'mode': 'MM_SA_BA',  # Default attention mode (None, MM_SA, MM_BA, MM_SA_BA)
            'snp_dim': snp_dim,  # SNP feature dimension
            'img_channels': 3,  # 3 orthogonal slices
            'num_classes': 1,  # Binary classification like EIGN
            'img_size': IMG_SIZE  # Full image size for slice extraction
        }
    elif method == 'ADCNN':
        # Use single full patch like ResNet for direct 3D image access
        opt.center_mat = FULL_CENTER_MAT
        opt.patch_size = [96, 96, 96]

        # Liu CNN doesn't use SNP data but we keep it for interface compatibility
        if len(opt.modals) > 1:
            _, tissue, pt = opt.modals[1].split('.')
        else:
            tissue, pt = 'Brain_All', 'gwas_p_5-e8'

        # Get SNP dimension for compatibility (though not used)
        eqtl_path = os.path.join(eQTL_PATH, '%s.eQTL_2_gene.%s.txt' % (tissue, pt))
        eqtls = pd.read_csv(eqtl_path, sep='\t')
        eqtls.columns = ['snp', 'gene']
        snp_dim = len(eqtls['snp'].drop_duplicates().values)

        opt.method_setting = {
            'num_classes': 1,  # Binary classification
            'img_size': opt.patch_size,  # Full image size
            'snp_dim': snp_dim,  # For compatibility (not used)
            'in_channel': 1,  # Input channels - prepare_model default
            'feat_dim': 1024,  # CNN feature dimension - prepare_model default
            'n_hid_main': 512,  # Hidden layer size - prepare_model default
            'expansion': 8,  # Channel expansion factor - prepare_model default
            'type_name': 'conv3x3x3',  # CNN type - prepare_model default
            'norm_type': 'Instance'  # Normalization type - prepare_model default
        }
    elif method == 'AlzCLIP':
        # AlzCLIP multi-modal contrastive learning with two-stage training
        # Use single full patch like ResNet for AAL3v1 brain region extraction
        opt.center_mat = FULL_CENTER_MAT
        opt.patch_size = FULL_PATCH_SIZE

        # Get SNP dimension from eQTL file
        if len(opt.modals) > 1:
            _, tissue, pt = opt.modals[1].split('.')
        else:
            tissue, pt = 'Brain_All', 'gwas_p_5-e8'

        eqtl_path = os.path.join(eQTL_PATH, '%s.eQTL_2_gene.%s.txt' % (tissue, pt))
        eqtls = pd.read_csv(eqtl_path, sep='\t')
        eqtls.columns = ['snp', 'gene']
        snp_dim = len(eqtls['snp'].drop_duplicates().values)

        opt.method_setting = {
            'embedding_dim': 256,  # CLIP embedding dimension
            'projection_dim': 128,  # CLIP projection dimension
            'num_classes': 1,  # Binary classification like EIGN
            'dropout': 0.1,  # Dropout rate
            'temperature': 1.0,  # Contrastive learning temperature
            'pretrain_epochs': 100,  # Number of pretraining epochs before switching to fine-tuning
            'pretrain_checkpoint_save_path': os.path.join(BASEDIR, 'saved_model/best_pretrain_alzclip.pt'),
            'img_feature_dim': 170,  # AAL3v1 brain regions (170 ROIs)
            'snp_input_dim': snp_dim,  # SNP input dimension from eQTL
            'pretrain_lr': 0.001,  # Learning rate for pretraining stage
            'finetune_lr': 0.0001,  # Learning rate for fine-tuning stage
            'weight_decay': 0.0001,  # Weight decay for both stages
        }

    else:
        raise NotImplementedError

    return opt
