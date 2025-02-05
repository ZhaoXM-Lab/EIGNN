# Zhou, X. et al. Deep learning-based polygenic risk analysis for Alzheimer’s disease prediction. Commun Med 3, 1–20 (2023).
import logging
import torch
from torch import nn
import numpy as np


class NNP(nn.Module):
    def __init__(self, n_input, num_classes=1, ):
        '''pytorch implementation of the original keras implementation:
            model = Sequential()
            model.add(Dropout(0.3, input_shape=(N_var,)))
            model.add(Dense(50, activation='sigmoid', kernel_regularizer=l1(5e-5)))
            model.add(Dropout(0.1))
            model.add(Dense(20, activation='sigmoid', kernel_regularizer=l1(5e-5)))
            model.add(Dense(10, activation='sigmoid', ))
            model.add(Dense(5, activation='sigmoid', ))
            model.add(Dense(1, activation='sigmoid', ))
            model.summary()
        '''
        super(NNP, self).__init__()
        if num_classes > 1:
            raise NotImplementedError
        logging.warning("NNP is now only implemented for non-image data, modals[0] will be ommited")
        # *note*: in the original implement, dropout1 is 0.3.
        # However, we found that large dropout rate significantly degraded the performance in our settings.
        self.dropout1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(n_input, 50)
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 5)
        self.fc5 = nn.Linear(5, 1)

        self.lambda_l1 = 5e-5

    def forward(self, img, snp, std_out=False):
        x = self.dropout1(snp)
        x = torch.sigmoid(self.fc1(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)  # *note*: sigmoid has been implemented in the loss function
        if std_out:
            return x
        else:
            return x,

    def l1_regularization(self):
        l1_loss = 0
        l1_loss += torch.sum(torch.abs(self.fc1.weight))
        l1_loss += torch.sum(torch.abs(self.fc2.weight))
        return self.lambda_l1 * l1_loss

    def criterion(self, outputs, targets):
        loss = nn.functional.binary_cross_entropy_with_logits(outputs, targets)
        l1_loss = self.l1_regularization()
        return loss + l1_loss

    def evaluate_data(self, val_loader, device, dtype='float32'):
        predicts = []
        groundtruths = []
        group_labels = []

        with torch.no_grad():
            self.train(False)
            for i, data in enumerate(val_loader, 0):
                inputs, aux_labels, labels, dis_label = data
                inputs = inputs.to(device=device, dtype=dtype)
                aux_labels = aux_labels.to(device=device, dtype=dtype)
                outputs, = self(inputs, aux_labels[:, 0])
                predicts.append(outputs)
                groundtruths.append(labels[:, 0, :])  # multi patch
                group_labels.append(dis_label)

            device = next(self.parameters()).device
            pred = predicts
            pred = torch.cat(pred, 0)
            groundtruths = torch.cat(groundtruths, dim=0).squeeze(-1).to(dtype)
            group_labels = torch.cat(group_labels, dim=0).to(torch.long)
            val_loss = self.criterion(pred.to(device),
                                      groundtruths.to(device=device))
            pred = torch.sigmoid(pred)
            pred = pred.unsqueeze(-1).cpu().numpy()
            groundtruths = groundtruths.unsqueeze(-1).cpu().numpy()
            group_labels = group_labels.cpu().numpy()
            val_loss = val_loss.cpu().item()
        return pred, groundtruths, group_labels, val_loss

    def fit(self, train_loader, optimizer, device, dtype):
        self.train(True)
        losses = torch.zeros(1, dtype=dtype, device=device, )

        c = 0
        batch_size = train_loader.batch_size
        inputs_buf = torch.Tensor()
        aux_labels_buf = torch.Tensor()
        labels_buf = torch.Tensor()
        for n, data in enumerate(train_loader, 0):
            inputs, aux_labels, labels, dis_label = data
            ## to collect data for the case that input might contains nan
            inx = ~torch.isnan(labels.view(labels.shape[0], -1)[:, 0])
            inx = inx & (~torch.isnan(aux_labels.view(aux_labels.shape[0], -1)[:, 0]))
            inputs_buf = torch.cat([inputs_buf, inputs[inx]], 0)
            aux_labels_buf = torch.cat([aux_labels_buf, aux_labels[inx]], 0)
            labels_buf = torch.cat([labels_buf, labels[inx]], 0)
            if (n + 1) < len(train_loader):
                if inputs_buf.shape[0] < batch_size + 2:  # batch norm must use more than 1 sample
                    continue
                else:
                    inputs = inputs_buf[:batch_size]
                    aux_labels = aux_labels_buf[:batch_size]
                    labels = labels_buf[:batch_size]

                    inputs_buf = inputs_buf[batch_size:]
                    aux_labels_buf = aux_labels_buf[batch_size:]
                    labels_buf = labels_buf[batch_size:]
            else:
                inputs = inputs_buf
                aux_labels = aux_labels_buf
                labels = labels_buf
            c += 1
            ##

            # multi patch
            labels = labels[:, 0, :].to(device=device, dtype=dtype)
            aux_labels = aux_labels.to(device=device, dtype=dtype)
            inputs = inputs.to(device=device, dtype=dtype)

            optimizer.zero_grad()
            outputs, = self(inputs, aux_labels[:, 0])

            assert labels.shape[1] == 1
            loss = self.criterion(outputs, labels[:, 0, :])

            loss.backward(retain_graph=True)
            losses += loss.detach()
            optimizer.step()
        return losses / c


if __name__ == '__main__':
    def count_params(model, framework='pytorch'):
        if framework == 'pytorch':
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
        elif framework == 'keras':
            params = model.count_params()
        else:
            raise NotImplementedError
        print('The network has {} params.'.format(params))


    model = NNP(846, num_classes=1, )
    count_params(model)

    n = 16
    device = 'cuda:0'
    img = torch.randn(n, 80, 1, 25, 25, 25, dtype=torch.float32).to(device)
    snp = torch.randint(0, 2, [n, 846], dtype=torch.float32).to(device)
    model = model.to(device).to(dtype=torch.float32)

    pred = model(img, snp, std_out=True)
    loss = model.criterion(pred, (torch.rand(n, 1) > .5).to(torch.float32).to(device))
