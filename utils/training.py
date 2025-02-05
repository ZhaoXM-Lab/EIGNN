import logging
from datetime import datetime
import socket
import os
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, recall_score, confusion_matrix
from sklearn.metrics import auc
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .opts import BASEDIR, DTYPE, LR_PATIENCE, LR_MUL_FACTOR, MOMENTUM, DAMPENING, WEIGHT_DECAY
from .utils import mf_save_model, mf_load_model
from .tools import ModifiedReduceLROnPlateau, Save_Checkpoint


def train_epoch(epoch, global_step, train_loader, model, optimizer, writer, device):
    metric_strs = {}
    losses = model.fit(train_loader, optimizer, device, dtype=DTYPE)
    metric_strs['loss'] = losses.sum().item()
    if writer:
        for k, v in metric_strs.items():
            writer.add_scalar(k, v, global_step=global_step)
    print('Epoch: [%d], loss_sum: %.2f' %
          (epoch + 1, losses.sum().item()))


def evaluation(model, val_loader, device):
    predicts, groundtruths, group_labels, val_loss = model.evaluate_data(val_loader, device, dtype=DTYPE)
    try:
        val_loss = val_loss.detach().cpu().item()
    except:
        pass
    predict1 = predicts[:, 0, :]
    groundtruth1 = groundtruths[:, 0, :]

    predict1 = np.array(predict1)
    groundtruth1 = np.array(groundtruth1)
    return predict1, groundtruth1, val_loss


def validate_epoch(global_step, val_loader, model, device, writer):
    # monitor the performance for every subject
    pre, label, val_loss = evaluation(model=model, val_loader=val_loader, device=device)
    pre[np.isnan(pre)] = 0
    prec, rec, thr = precision_recall_curve(label, pre)
    fpr, tpr, thr = roc_curve(label, pre)
    tn, fp, fn, tp = confusion_matrix(y_pred=pre.round(), y_true=label).ravel()

    metric_strs = {}
    metric_figs = {}
    metric_strs['AUC'] = auc(fpr, tpr)
    metric_strs['AUPR'] = auc(rec, prec)
    metric_strs['ACC'] = accuracy_score(y_pred=pre.round(), y_true=label)
    metric_strs['SEN'] = tp / (tp + fn)
    metric_strs['SPE'] = tn / (tn + fp)
    metric_strs['Val_Loss'] = val_loss

    if writer:
        for k, v in metric_strs.items():
            writer.add_scalar(k, v, global_step=global_step)

    val_metric = -metric_strs['Val_Loss'] + metric_strs['AUC']
    return metric_figs, metric_strs, val_metric


def train_single(model, n_epochs, dataloaders, optimname, learning_rate,
                 device, logstring, no_log, no_val):
    # the change for model is inplace
    train_loader, val_loader, test_loader = dataloaders
    model.to(device=device, dtype=DTYPE)

    if optimname == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=MOMENTUM,
                              dampening=DAMPENING, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    if not no_log:
        writer = SummaryWriter(comment=logstring)
    else:
        writer = None

    scheduler = ModifiedReduceLROnPlateau(optimizer, 'min', patience=LR_PATIENCE, factor=LR_MUL_FACTOR, verbose=True)
    saved_path = os.path.join(BASEDIR, 'saved_model', datetime.now().strftime('%b%d_%H-%M-%S') + '_' +
                              socket.gethostname() + logstring + '.pth')
    if os.path.exists(saved_path):
        saved_path = os.path.join(BASEDIR, 'saved_model', datetime.now().strftime('%b%d_%H-%M-%S-%f') + '_' +
                                  socket.gethostname() + logstring + '.pth')
    save_checkpoint = Save_Checkpoint(save_func=mf_save_model, verbose=True,
                                      path=saved_path, trace_func=print, mode='min')

    for epoch in range(n_epochs):
        global_step = len(train_loader.dataset) * epoch
        train_epoch(epoch, global_step, train_loader, model, optimizer, writer, device)
        if not no_val:
            metric_figs, metric_strs, val_metric = validate_epoch(
                global_step, val_loader, model, device, writer)
            print({k: '%.3f' % v for k, v in metric_strs.items()})
        else:
            val_metric = epoch

        scheduler.step(-val_metric)
        save_checkpoint(-val_metric, model)

    model = mf_load_model(model=model, path=saved_path, device=device)
    return model, saved_path


def train(models, n_epochs, dataloader_list: list, optimname, learning_rate,
          device, logstring, no_log, no_val):
    trainedmodels = []
    saved_paths = []

    for i, dataloaders in enumerate(dataloader_list):
        trainedmodel, saved_path = train_single(models[i], n_epochs, dataloaders, optimname,
                                                learning_rate, device, logstring,
                                                no_log, no_val)

        trainedmodels.append(trainedmodel.to('cpu'))
        saved_paths.append(saved_path)

        for d in dataloaders:
            d.dataset.imgdata.empty_cache()

    return trainedmodels, saved_paths


def predict(models, dataloader_list: list, device):
    metric_figlist = []
    metric_strlist = []

    if len(dataloader_list) == 1:
        dataloader_list = [dataloader_list[0] for i in range(len(models))]

    assert len(models) == len(dataloader_list)
    for i, model in enumerate(models):
        _, _, test_loader = dataloader_list[i]
        model = model.to(device)
        metric_figs, metric_strs, _ = validate_epoch(None, test_loader, model, device, writer=None)
        metric_figlist.append(metric_figs)
        metric_strlist.append(metric_strs)


        model = model.to('cpu')  # remove model from gpu
        test_loader.dataset.imgdata.empty_cache()

    if len(models) == 1:
        reduced_result = {k: '%.3f' % v for k, v in metric_strlist[0].items()}
    else:
        values = np.array([[v for v in d.values()] for d in metric_strlist])
        keys = metric_strlist[0].keys()
        avg = values.mean(axis=0)
        std = values.std(axis=0)
        reduced_result = {k: '%.3f ± %.3f' % (avg, std) for k, avg, std in zip(keys, avg, std)}

    return reduced_result, metric_figlist, metric_strlist
