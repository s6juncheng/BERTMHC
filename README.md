# BERTMHC

Predict peptide MHC binding and presentation with BERT model.

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
