import sys
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import logging
import numpy as np
import torch
from utils.utils import mk_dirs, get_models
from utils.datasets import get_dataset
from utils.training import train, predict
from utils.opts import parse_opts, mod_opt, BASEDIR, DATASEED


logging.basicConfig(level='WARNING')

if __name__ == "__main__":
    opt = parse_opts()
    logging.info("%s" % opt)
    if opt.no_cuda:
        device = 'cpu'
    else:
        device = torch.device("cuda:%d" % opt.cuda_index if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    mk_dirs(basedir=BASEDIR)
    opt.num_classes = 1
    TB_COMMENT = '_'.join([opt.method, '-'.join(opt.modals), opt.clfsetting, opt.log_label])
    opt = mod_opt(opt.method, opt)
    opt.method_setting.update(eval(opt.method_para))

    ## generate datasets
    print('Training Dataset: ', opt.dataset)
    dataloader_list = get_dataset(dataset=opt.dataset, clfsetting=opt.clfsetting, modals=opt.modals,
                                  patch_size=opt.patch_size, batch_size=opt.batch_size,
                                  center_mat=opt.center_mat, flip_axises=opt.flip_axises,
                                  no_smooth=opt.no_smooth, no_shuffle=opt.no_shuffle, no_shift=opt.no_shift,
                                  n_threads=opt.n_threads, seed=DATASEED, resample_patch=opt.resample_patch,
                                  trtype=opt.trtype)
    # get models
    pretrain_paths = [None for i in range(len(dataloader_list))]
    if len(opt.pretrain_path) > 0:
        assert len(opt.pretrain_path) == len(dataloader_list)
        pretrain_paths = opt.pretrain_path

    models = get_models(opt.method, opt.method_setting, opt.trtype, pretrain_paths=pretrain_paths, device=device)

    if not opt.no_train:
        ## training
        models, saved_paths = train(models, opt.n_epochs,
                                    dataloader_list,
                                    optimname=opt.optimizer,
                                    learning_rate=opt.learning_rate,
                                    device=device, logstring=TB_COMMENT,
                                    no_log=opt.no_log,
                                    no_val=opt.no_val)

    ## prediction
    result = dict()
    reduced_result, metric_figlist, metric_strlist = predict(models, dataloader_list, device)
    result[opt.dataset] = reduced_result
    if opt.pre_datasets is not None:
        for pre_dataset in opt.pre_datasets:
            dataloader_list = get_dataset(dataset=pre_dataset, clfsetting=opt.clfsetting, modals=opt.modals,
                                          patch_size=opt.patch_size, batch_size=opt.batch_size,
                                          center_mat=opt.center_mat, flip_axises=opt.flip_axises,
                                          no_smooth=opt.no_smooth, no_shuffle=opt.no_shuffle, no_shift=opt.no_shift,
                                          n_threads=opt.n_threads, seed=DATASEED, resample_patch=opt.resample_patch,
                                          trtype=opt.trtype)
            reduced_result, metric_figlist, metric_strlist = predict(models, dataloader_list, device)
            result[pre_dataset] = reduced_result
    if opt.pre_testonly_datasets is not None:
        for pre_dataset in opt.pre_testonly_datasets:
            dataloader_list = get_dataset(dataset=pre_dataset, clfsetting=opt.clfsetting, modals=opt.modals,
                                          patch_size=opt.patch_size, batch_size=opt.batch_size,
                                          center_mat=opt.center_mat, flip_axises=opt.flip_axises,
                                          no_smooth=opt.no_smooth, no_shuffle=opt.no_shuffle, no_shift=opt.no_shift,
                                          n_threads=opt.n_threads, seed=DATASEED, resample_patch=opt.resample_patch,
                                          trtype='single')
            reduced_result, metric_figlist, metric_strlist = predict(models, dataloader_list, device)
            result[pre_dataset] = reduced_result

    print('*****************************\r\n')
    print('Testing Result: \r\n %s\r\n' % result)
    print('*****************************')

    if not opt.no_train:
        output_str = {'Method': opt.method,
                      'Dataset': opt.dataset,
                      'test_result': '%s' % result,
                      'saved_paths': '%s' % saved_paths,
                      'opt': '%s' % opt}
        print(output_str)
        if opt.result_path is None:
            fname = os.path.split(saved_paths[0])[-1].replace('.pth', '.txt')
            opt.result_path = os.path.join(BASEDIR, 'results', fname)

        with open(opt.result_path, 'a') as f:
            f.write(str(output_str))
