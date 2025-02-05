import numpy as np
import os
import torch
import logging
import re
from models import EIGNN, RF, PRS, SVM, LR, NNP, APOE

method_map = {'EIGNN': EIGNN, 'RF': RF, 'PRS': PRS, 'SVM': SVM, 'LR': LR,
              'NNP': NNP, 'APOE': APOE}


def mk_dirs(basedir):
    dirs = [os.path.join(basedir, 'runs'),
            os.path.join(basedir, 'utils', 'datacheck'),
            os.path.join(basedir, 'saved_model'),
            os.path.join(basedir, 'results')]

    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)


def get_models(method, method_setting, trtype: str, pretrain_paths: list, device):
    models = []
    Model = method_map[method]
    if trtype == 'single':
        num_models = 1
    elif trtype in ['5-rep', '5cv']:
        num_models = 5
    elif trtype in ['10cv']:
        num_models = 10
    else:
        raise NotImplementedError

    for i in range(num_models):
        model = Model(**method_setting)
        if pretrain_paths[i] is not None:
            model = mf_load_model(model, pretrain_paths[i], device=device)
        models.append(model)
    return models


def mf_save_model(model, path, framework):
    if framework == 'pytorch':
        if not hasattr(model, 'non_torch_model'):
            torch.save(model.state_dict(), path)
        else:
            torch.save({'model': model}, path)
    elif framework == 'keras':
        model.save(path)
    else:
        raise NotImplementedError


def mf_load_model(model, path, framework='pytorch', device='cpu'):
    if framework == 'pytorch':
        if not hasattr(model, 'non_torch_model'):
            w = torch.load(path, map_location=device)
            try:
                model.load_state_dict(w, strict=True)
            except RuntimeError as e:
                model.load_state_dict(w, strict=False)
                logging.warning('Loaded pretrain train model Unstrictly! Msg: %s' % e)
        else:
            model = torch.load(path, map_location=device)['model']
    elif framework == 'keras':
        model.load_weights(path)
    else:
        raise NotImplementedError
    return model


def check_mem(cuda_device):
    devices_info = os.popen(
        '"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split(
        "\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used


def count_params(model, framework='pytorch'):
    if framework == 'pytorch':
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
    elif framework == 'keras':
        params = model.count_params()
    else:
        raise NotImplementedError
    print('The network has {} params.'.format(params))


def info_parse(fname, prefix):
    _, img, tissue, _, pt, graph = re.search(
        '%s_IMG-GO_(None)?([A-Za-z]{1,})?-(.*)-(None)?(gwas.*(_[A-Za-z]{1,})?)?\.txt' % prefix,
        fname).groups()
    if img is None:
        img = 'None'
    if pt is None:
        pt = 'None'
    if graph is None:
        graph = 'None'

    return img, tissue, pt
