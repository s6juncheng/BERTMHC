# BERTMHC

Predict peptide MHC binding and presentation with BERT model.

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

After model training, to predict with trained models, use `bertmhc predict`
```
bertmhc predict --help
```
