# coding=utf-8
# NAME OF THE PROGRAM THIS FILE BELONGS TO
#
#       BERTMHC
#
#   file: utils_model.py
#
#    Authors: Jun Cheng (jun.cheng@neclab.eu)
#             Brandon Malone (brandon.malone@neclab.eu)
#
# NEC Laboratories Europe GmbH, Copyright (c) 2020, All rights reserved.
#     THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#     PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
#
#                                                                                                                                                                           This is a license agreement ("Agreement") between your academic institution or non-profit organization or self (called "Licensee" or "You" in this Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this Agreement).  All rights not specifically granted to you in this Agreement are reserved for Licensor.
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive ownership of any copy of the Software (as defined below) licensed under this Agreement and hereby grants to Licensee a personal, non-exclusive, non-transferable license to use the Software for noncommercial research purposes, without the right to sublicense, pursuant to the terms and conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF LICENSOR’S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this Agreement, the term "Software" means (i) the actual copy of all or any portion of code for program routines made accessible to Licensee by Licensor pursuant to this Agreement, inclusive of backups, updates, and/or merged copies permitted hereunder or subsequently supplied by Licensor,  including all or any file structures, programming instructions, user interfaces and screen formats and sequences as well as any and all documentation and instructions related to it, and (ii) all or any derivatives and/or modifications created or made by You to any of the items specified in (i).
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is proprietary to Licensor, and as such, Licensee agrees to receive all such materials and to use the Software only in accordance with the terms of this Agreement.  Licensee agrees to use reasonable effort to protect the Software from unauthorized use, reproduction, distribution, or publication. All publication materials mentioning features or use of this software must explicitly include an acknowledgement the software was developed by NEC Laboratories Europe GmbH.
# COPYRIGHT: The Software is owned by Licensor.
#     PERMITTED USES:  The Software may be used for your own noncommercial internal research purposes. You understand and agree that Licensor is not obligated to implement any suggestions and/or feedback you might provide regarding the Software, but to the extent Licensor does so, you are not entitled to any compensation related thereto.
# DERIVATIVES: You may create derivatives of or make modifications to the Software, however, You agree that all and any such derivatives and modifications will be owned by Licensor and become a part of the Software licensed to You under this Agreement.  You may only use such derivatives and modifications for your own noncommercial internal research purposes, and you may not otherwise use, distribute or copy such derivatives and modifications in violation of this Agreement.
# BACKUPS:  If Licensee is an organization, it may make that number of copies of the Software necessary for internal noncommercial use at a single site within its organization provided that all information appearing in or on the original labels, including the copyright and trademark notices are copied onto the labels of the copies.
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except as explicitly permitted herein. Licensee has not been granted any trademark license as part of this Agreement. Neither the name of NEC Laboratories Europe GmbH nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in whole or in part, or provide third parties access to prior or present versions (or any parts thereof) of the Software.
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder without the prior written consent of Licensor. Any attempted assignment without such consent shall be null and void.
# TERM: The term of the license granted by this Agreement is from Licensee's acceptance of this Agreement by downloading the Software or by using the Software until terminated as provided below.
# The Agreement automatically terminates without notice if you fail to comply with any provision of this Agreement.  Licensee may terminate this Agreement by ceasing using the Software.  Upon any termination of this Agreement, Licensee will delete any and all copies of the Software. You agree that all provisions which operate to protect the proprietary rights of Licensor shall remain in force should breach occur and that the obligation of confidentiality described in this Agreement is binding in perpetuity and, as such, survives the term of the Agreement.
# FEE: Provided Licensee abides completely by the terms and conditions of this Agreement, there is no fee due to Licensor for Licensee's use of the Software in accordance with this Agreement.
#     DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON-INFRINGEMENT.  LICENSEE BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND RELATED MATERIALS.
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is provided as part of this Agreement.
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent permitted under applicable law, Licensor shall not be liable for direct, indirect, special, incidental, or consequential damages or lost profits related to Licensee's use of and/or inability to use the Software, even if Licensor is advised of the possibility of such damage.
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable export control laws, regulations, and/or other laws related to embargoes and sanction programs administered by law.
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be invalid, illegal, or unenforceable by a court or other tribunal of competent jurisdiction, the validity, legality and enforceability of the remaining provisions shall not in any way be affected or impaired thereby.
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right or remedy under this Agreement shall be construed as a waiver of any future or other exercise of such right or remedy by Licensor.
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance with the laws of Germany without reference to conflict of laws principles.  You consent to the personal jurisdiction of the courts of this country and waive their rights to venue outside of Germany.
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and entire agreement between Licensee and Licensor as to the matter set forth herein and supersedes any previous agreements, understandings, and arrangements between the parties relating hereto.



from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
import tempfile
from bertmhc.bertmhc import BERTMHC
from tape import ProteinBertConfig
from torch.utils.data import DataLoader
from bertmhc.dataloader import BertDataset
import pandas as pd
import torch
from copy import deepcopy
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import LambdaLR


def pred_bert(inp, model_path, peplen, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    # model
    if isinstance(model_path, str):
        config = ProteinBertConfig.from_pretrained('bert-base')
        model = BERTMHC(config)
        weights = torch.load(model_path)
        if list(weights.keys())[0].startswith('module.'):
            weights = {k[7:]: v for k, v in weights.items() if k.startswith('module.')}
        model.load_state_dict(weights)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(device)
    elif isinstance(model_path, torch.nn.Module):
        model = model_path
    else:
        raise ValueError('unknown model_path type')
    model.eval()

    # data
    valset = BertDataset(inp,
                         max_pep_len=peplen)
    val_data = DataLoader(valset,
                          batch_size=batch_size,
                          num_workers=16,
                          pin_memory=True,
                          collate_fn=valset.collate_fn)
    # pred

    aff_pred = []
    mass_pred = []
    with torch.no_grad():
        for batch in tqdm(val_data):
            batch = {name: tensor.to(device)
                     for name, tensor in batch.items()}
            logits, targets = model(**batch)
            pred = torch.sigmoid(logits).cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()
            aff_pred.append(pred[:, 0])
            mass_pred.append(pred[:, 1])

    aff_pred = np.concatenate(aff_pred)
    mass_pred = np.concatenate(mass_pred)
    if isinstance(inp, pd.DataFrame):
        dt = deepcopy(inp)
    else:
        dt = pd.read_csv(inp)
    dt['affinity_pred'] = aff_pred
    dt['mass_pred'] = mass_pred
    return dt


class MAData(object):
    '''
    Helper class to train a de-convolution model
    MA - multiple allele data, must be flatten each row one allele
    SA - single allele data

    args:
        data: contains ma and sa
        threshold: cutoff threshold on the predicted score, samples above the score are labeled as positive.
            if threshold=='max', we only label the maximum predicted allele as positive
        sa_epochs: epoch to train on SA only data
        negative: "max": maximum predicted, "all": use all negatives
    '''

    def __init__(self,
                 data,
                 sa_epochs=15,
                 tempdir='/tmp/',
                 calibrate=False,
                 negative='max'):
        data = pd.read_csv(data)
        ma = data[data['MA']].reset_index(drop=True)
        ma['instance_weights'] = 1.0
        self.ma = ma
        self.ma_pos = ma[ma['masslabel'] == 1].reset_index(drop=True)
        self.ma_neg = ma[ma['masslabel'] == 0].reset_index(drop=True)
        assert "," not in ma['allele'].tolist(
        )[0], '{0}, use reshape_ma function to flatten alleles'.format(ma['allele'][0])
        self.sa = data[~data['MA']].reset_index(drop=True)
        self.sa = self.sa[['peptide', 'masslabel', 'allele', 'mhc', 'label', 'MA']]
        self.sa['instance_weights'] = 1.0
        print("Number of SA {0}".format(len(self.sa)))
        print("Number of MA {0}".format(len(ma)))
        self.train_ma = False
        self._step = 0
        self.sa_epochs = sa_epochs
        self.outfile = None
        self.tempdir = tempdir
        self.calibrate = calibrate
        self.negative = negative

    def close(self):
        if self.outfile is not None:
            self.outfile.close()

    def get_training(self):
        ''' Get training data with pseudo labels
        '''
        pass

    def generate_training(self, model, peplen, **kwargs):
        ''' Get training data with pseudo labels
        Generate pseudo label with some model
          typically use the current iteration of the model
        '''
        self.outfile = tempfile.NamedTemporaryFile('w', dir=self.tempdir)
        if self._step >= self.sa_epochs:
            self.train_ma = True
            print("Training with multi-allele data.")
        if self.train_ma:
            print("Generating pseudo-labels for MA data")
            bs = kwargs.pop('batch_size', 128)
            if self.negative == 'all':
                pred = pred_bert(self.ma_pos, model, peplen, batch_size=bs)
            else:
                pred = pred_bert(self.ma, model, peplen, batch_size=bs)
            # calibrate prediction probability with calibrator trained on on SA data
            if self.calibrate:
                _pred = pred_bert(self.sa[self.sa['masslabel']!=-1], model, peplen, batch_size=bs)
                lr = IsotonicRegression(out_of_bounds='clip')
                print("calibrate probability")
                lr.fit(logit(_pred['mass_pred'].values), _pred['masslabel'].values)
                pred['mass_pred'] = lr.predict(logit(pred['mass_pred'].values))
            pred = self.pseudolabel(pred, **kwargs)
            pred = pred[['peptide', 'masslabel', 'allele',
                         'mhc', 'label', 'MA', 'instance_weights']]
            if self.negative == 'all':
                data = pd.concat([pred, self.ma_neg, self.sa]).reset_index(drop=True)
            else:
                data = pd.concat([pred, self.sa]).reset_index(drop=True)
            data.to_csv(self.outfile.name)
        else:
            print("Training on SA only")
            self.sa.to_csv(self.outfile.name)
        self._step += 1
        return self.outfile.name

    def pseudolabel(self, pred, score='mass_pred'):
        '''Generate pseudo label with some model
        '''
        max_ids = pred.groupby(['group_index'])[score].idxmax()
        pred = pred.loc[max_ids].reset_index(drop=True)
        pred['instance_weights'] = np.clip(pred[score], 0.01, 1.0)
        pred.drop(['affinity_pred', 'mass_pred'], axis=1, inplace=True)
        return pred

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7,
                 verbose=False,
                 delta=0,
                 saveto='checkpoint.pt',
                 save='best'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save: 'best' or 'all'
        """
        self.patience = patience
        self.verbose = verbose
        self.saveto = saveto
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.epoch = -1
        if save == "all":
            save = -1 # backward compatability
        try:
            self.save = int(save)
        except:
            self.save = save

    def __call__(self, val_loss, model, optimizer=None):

        score = -val_loss
        self.epoch += 1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if isinstance(self.save, int):
                if self.epoch >= self.save:
                    self.save_checkpoint(val_loss, model, optimizer)
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer=None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        to = deepcopy(self.saveto)
        optim_to = to.split(".pt")[0] + "_optim.pt"
        if isinstance(self.save, int):
            if self.epoch >= self.save:
                to += str(self.epoch)
                optim_to += str(self.epoch)
        torch.save(model.state_dict(), to)
        if optimizer:
            torch.save(optimizer.state_dict(), optim_to)
        self.val_loss_min = val_loss

    def reset(self):
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf


def train(model, optimizer, train_data, device, aff_criterion, mass_criterion, alpha, scheduler=None):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    for batch in tqdm(train_data):
        batch = {name: tensor.to(device)
                 for name, tensor in batch.items()}
        instance_weights = batch.pop('instance_weights',
                                     torch.ones(batch['targets'].shape[0], ).to(device))
        optimizer.zero_grad()
        logits, targets = model(**batch)

        label_valid = (targets[:, 0] != -1)
        mass_valid = (targets[:, 1] != -1)
        if torch.sum(label_valid) > 0:
            aff_loss = aff_criterion(logits[label_valid, 0], targets[label_valid, 0])
        else:
            aff_loss = 0
        if torch.sum(mass_valid) > 0:
            mass_loss = mass_criterion(logits[mass_valid, 1], targets[mass_valid, 1])
            mass_loss *= instance_weights[mass_valid]
            mass_loss = mass_loss.sum() / instance_weights.sum()
        else:
            mass_loss = 0
        loss = (1 - alpha) * aff_loss + alpha * mass_loss
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        if isinstance(scheduler, LambdaLR):
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_data.dataset)


def evaluate(model, val_data, device, aff_criterion, mass_criterion, alpha):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    label_np = []
    label_np = []
    masslabel_np = []
    aff_pred = []
    mass_pred = []
    ret = {}
    with torch.no_grad():
        for batch in val_data:
            batch = {name: tensor.to(device)
                     for name, tensor in batch.items()}
            instance_weights = batch.pop('instance_weights',
                                         torch.ones(batch['targets'].shape[0], ).to(device))
            logits, targets = model(**batch)

            label_valid = (targets[:, 0] != -1)
            mass_valid = (targets[:, 1] != -1)
            if torch.sum(label_valid) > 0:
                aff_loss = aff_criterion(logits[label_valid, 0], targets[label_valid, 0]).item()
            else:
                aff_loss = 0
            if torch.sum(mass_valid) > 0:
                mass_loss = mass_criterion(logits[mass_valid, 1], targets[mass_valid, 1])
                mass_loss *= instance_weights[mass_valid]
                mass_loss = mass_loss.sum() / instance_weights.sum()
                mass_loss = mass_loss.item()
            else:
                mass_loss = 0
            total_loss += (1 - alpha) * aff_loss + alpha * mass_loss

            pred = torch.sigmoid(logits).cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()
            aff_pred.append(pred[:, 0])
            mass_pred.append(pred[:, 1])
            label_np.append(targets[:, 0])
            masslabel_np.append(targets[:, 1])

    aff_pred = np.concatenate(aff_pred)
    mass_pred = np.concatenate(mass_pred)
    label_np = np.concatenate(label_np)
    masslabel_np = np.concatenate(masslabel_np)
    label_valid = (label_np != -1)
    mass_valid = (masslabel_np != -1)
    ret['val_cor'] = kendalltau(aff_pred[label_valid], label_np[label_valid])[0]
    label_bin = (label_np > 0.426).astype(float)[label_valid]
    try:
        ret['val_auc'] = roc_auc_score(label_bin, aff_pred[label_valid])
    except:
        ret['val_auc'] = np.nan
    try:
        ret['val_ap'] = average_precision_score(masslabel_np[mass_valid], mass_pred[mass_valid])
    except:
        ret['val_ap'] = np.nan
    if 'MA' in val_data.dataset.data.data:
        sa = ~val_data.dataset.data.data['MA']
        mass_valid = (masslabel_np != -1) & sa
        ret['val_ap_sa'] = average_precision_score(masslabel_np[mass_valid], mass_pred[mass_valid])
    try:
        ret['val_mass_auc'] = roc_auc_score(masslabel_np[mass_valid], mass_pred[mass_valid])
    except:
        ret['val_mass_auc'] = np.nan

    ret['val_loss'] = total_loss / len(val_data.dataset)

    return ret
