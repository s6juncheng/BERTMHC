# coding=utf-8
# NAME OF THE PROGRAM THIS FILE BELONGS TO
#
#       BERTMHC
#
#   file: cli.py
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



"""Console script for bertmhc."""
import sys
from bertmhc.utils_model import EarlyStopping, MAData
from tape import ProteinBertConfig
from bertmhc.dataloader import BertDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from bertmhc.bertmhc import BERTMHC
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from bertmhc.utils_model import train, evaluate

logging.basicConfig(format='%(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Console script for bertmhc."""
    parser = argparse.ArgumentParser(description='PyTorch BERTMHC model')
    subparsers = parser.add_subparsers()

    # train
    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train_bertmhc)
    train_parser.add_argument('--data', type=str, default='tests/data/',
                        help='location of the data corpus')
    train_parser.add_argument('--eval', type=str, default='eval.csv',
                        help='evaluation set')
    train_parser.add_argument('--train', type=str, default='train.csv',
                        help='training set')
    train_parser.add_argument('--peplen', type=int, default=22,
                        help='peptide epitope length')
    train_parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    train_parser.add_argument('--alpha', type=float, default=0.0,
                        help='alpha weight on mass loss, affinity loss weight with 1-alpha')
    train_parser.add_argument('--epochs', type=int, default=30,
                        help='upper epoch limit')
    train_parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size')
    train_parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay')
    train_parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    train_parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    train_parser.add_argument('--w_pos', type=float, default=1.0,
                        help='mass positive weight')
    train_parser.add_argument('--metric', type=str, default='val_auc',
                        help='validation metric, default auc')
    train_parser.add_argument('--random_init', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='If True, Initialize the model random')
    train_parser.add_argument('--calibrate', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Calibrate probability')
    train_parser.add_argument('--deconvolution', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='If True, need to give Single allele (SA) and multi-allele (MA) data')
    train_parser.add_argument('--patience', type=int, default=5,
                        help='Earlystopping patience')
    train_parser.add_argument('--sa_epoch', type=int, default=15,
                        help="Number of epochs to train with single-allele data before deconvolution starts")
    train_parser.add_argument('--instance_weight', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='If True, use instance weights from the input data frame')
    train_parser.add_argument('--negative', type=str, default='max',
                        help="'max: maximum predicted, 'all: use all negatives")

    # predict
    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(func=predict)

    predict_parser.add_argument('--data', type=str, default='tests/data/eval.csv',
                              help='location of the data to predict')
    predict_parser.add_argument('--model', type=str, default='model.pt',
                              help='path to the trained model file')
    predict_parser.add_argument('--batch_size', type=int, default=512,
                              help='batch size')
    predict_parser.add_argument('--peplen', type=int, default=22,
                              help='peptide epitope length')
    predict_parser.add_argument('--task', type=str, choices=['binding', 'presentation'],
                                help='which prediction task, binding or presentation')
    predict_parser.add_argument('--output', type=str, default='output.csv',
                                help='path to the output file')
    args = parser.parse_args()
    args.func(args)
    logging.info("Arguments: %s", args)


def train_bertmhc(args):

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###############################################################################
    # Load data
    ###############################################################################
    if args.deconvolution:
        trainMa = MAData(args.data + args.train,
                         sa_epochs=args.sa_epoch,
                         calibrate=args.calibrate,
                         negative=args.negative)
        valMa = MAData(args.data + args.eval,
                       sa_epochs=args.sa_epoch,
                       calibrate=args.calibrate,
                       negative=args.negative)
    else:
        trainset = BertDataset(args.data + args.train,
                               max_pep_len=args.peplen,
                               instance_weight=args.instance_weight)
        valset = BertDataset(args.data + args.eval,
                             max_pep_len=args.peplen,
                             instance_weight=args.instance_weight)
        train_data = DataLoader(trainset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=16,
                                pin_memory=True,
                                collate_fn=trainset.collate_fn)
        val_data = DataLoader(valset,
                              batch_size=args.batch_size*2,
                              num_workers=16,
                              pin_memory=True,
                              collate_fn=valset.collate_fn)
        logger.info("Training on {0} samples, eval on {1}".format(len(trainset), len(valset)))

    ################
    # Load model
    ################
    device = torch.device(device)

    if args.random_init:
        config = ProteinBertConfig.from_pretrained('bert-base')
        model = BERTMHC(config)
    else:
        model = BERTMHC.from_pretrained('bert-base')

    for p in model.bert.parameters():
        p.requires_grad = True

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # loss
    aff_criterion = nn.BCEWithLogitsLoss()
    w_pos = torch.tensor([args.w_pos]).to(device)
    mass_criterion = nn.BCEWithLogitsLoss(pos_weight=w_pos, reduction='none')

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, nesterov=True)

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2, min_lr=1e-4, factor=0.1)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, saveto=args.save)

    for epoch in range(args.epochs):
        if args.deconvolution:
            trainset = BertDataset(trainMa.generate_training(model, args.peplen, score='mass_pred',
                                                             batch_size=args.batch_size*2),
                                   max_pep_len=args.peplen,
                                   instance_weight=args.instance_weight)
            valset = BertDataset(valMa.generate_training(model, args.peplen, score='mass_pred',
                                                         batch_size=args.batch_size*2),
                                 max_pep_len=args.peplen,
                                 instance_weight=args.instance_weight)
            train_data = DataLoader(trainset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=16,
                                    pin_memory=True,
                                    collate_fn=trainset.collate_fn)
            val_data = DataLoader(valset,
                                  batch_size=args.batch_size,
                                  num_workers=16,
                                  pin_memory=True,
                                  collate_fn=valset.collate_fn)
            trainMa.close()
            valMa.close()
            if epoch == trainMa.sa_epochs:
                print('Reset early stopping')
                # reset early stopping and scheduler
                early_stopping.reset()
                scheduler._reset()

        print("Training epoch {}".format(epoch))
        train_metrics = train(model, optimizer, train_data, device, aff_criterion, mass_criterion, args.alpha, scheduler)
        eval_metrics = evaluate(model, val_data, device, aff_criterion, mass_criterion, args.alpha)
        eval_metrics['train_loss'] = train_metrics
        logs = eval_metrics

        scheduler.step(logs.get(args.metric))
        logging.info('Sample dict log: %s' % logs)

        # callbacks
        early_stopping(-logs.get(args.metric), model, optimizer)
        if early_stopping.early_stop or logs.get(args.metric) <= 0:
            if args.deconvolution and not trainMa.train_ma:
                # still training SA only model, now switch to training on MA immediately
                trainMa.train_ma = True
                valMa.train_ma = True
                print("Start training with multi-allele data.")
            else:
                print("Early stopping")
                break

def predict(args):
    inp = args.data
    config = ProteinBertConfig.from_pretrained('bert-base')
    model = BERTMHC(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    valset = BertDataset(inp,
                         max_pep_len=args.peplen,
                         train=False)
    val_data = DataLoader(valset,
                          batch_size=args.batch_size,
                          num_workers=16,
                          pin_memory=True,
                          collate_fn=valset.collate_fn)
    pred = []
    for batch in tqdm(val_data):
        batch = {name: tensor.to(device)
                 for name, tensor in batch.items()}
        logits, _ = model(**batch)
        pred.append(torch.sigmoid(logits).cpu().detach().numpy())
    dt = pd.read_csv(inp)
    pred = np.concatenate(pred)
    if args.task == 'binding':
        dt['bertmhc_pred'] = pred[:,0]
    else:
        dt['bertmhc_pred'] = pred[:,1]
    dt.to_csv(args.output, index=None)
    return 0

if __name__ == "__main__":
    sys.exit(main())
