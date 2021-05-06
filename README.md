# BERTMHC

Predict peptide MHC binding and presentation with BERT model.
Paper: [BERTMHC: Improves MHC-peptide class II interaction prediction with transformer and multiple instance learning](https://www.biorxiv.org/content/10.1101/2020.11.24.396101v1)

## Licence
The code is only allowed for accedemic research. Commercial usage/research is not granted. Before using the code, please make sure you read and agree with the [Licence](https://github.com/s6juncheng/BERTMHC/blob/master/LICENSE)

## Installation
The package can be installed with ``pip``. In the root directory of this repo:

```
pip install .
```

## Training and prediction
The model can be trained with `bertmhc train`.
```
bertmhc train --help
```

An example input data format is provided in ``tests/data/{train,eval}.csv``.

### Training the binding model
To train a binding model, it is important to set `--alpha 0`.
See example input file ``tests/data/train.csv``. The required columns are `[allele, mhc, peptide, label]`.

```
bertmhc train --lr 0.15 --batch_size 64 --alpha 0 --wd 0.0
--peplen 24 --epochs 30
--data <data folder>
--train <train.csv.gz>
--eval <eval.csv.gz>
--save <model.pt>
```

### Training the presentation model
To train a presentation model with multiple alleles setting, the data need to be process as `test/data/train.csv`.
Specifically, a `group_index` column of integers and a `MA` column of boolean are required.
The `MA` column indicates whether the sample is from multi-allele or single allele. The `group_index` column use
integer values to track which alleles belonging to the same bag. Consider the following multi-allele data:

```
allele1, allele2  peptide1  1
allele1, allele3  peptide2  0
```

It needs to be expanded as:

```
allele  peptide masslabel  MA   group_index
allele1  peptide1  1    True    0
allele2  peptide1  1    True    0
allele1  peptide2  0    True    1
allele3  peptide2  0    True    1
```

After preparing the data, the presentation model can be trained with:

```
bertmhc train --lr 0.001 --batch_size 64 --alpha 0 --wd 0.0001 --deconvolution True
--metric val_ap --peplen 24 --epochs 30 --sa_epoch 15
--data <data folder>
--train <train.csv.gz>
--eval <eval.csv.gz>
--save <model.pt>
```

`--sa_epoch` is the number of epochs to train first on the SA data only. Use this if the input data consist of both SA and MA samples
(distinguished by the `MA` column in the input data).

## Prediction
After model training, to predict with trained models, use `bertmhc predict`. The required columns are `[allele, mhc, peptide]`.
```
bertmhc predict --data <test.csv.gz>
--model <trained_model.pt>
--peplen 24
--batch_size 64
--task <binding,presentation> # use 'binding' or 'presentation'
--output <output.csv.gz>
```

## Webserver
We provide a webserver to run our trained models described in the paper.
To use the webserver, please read and accept the terms of use.

https://bertmhc.privacy.nlehd.de/

You can submit maximum of 5k peptides for each query. The server might return error when overloaded. Please try again later if it does not work temporarily. Please feel free to open a github issue if you think the server is not running properly. 
