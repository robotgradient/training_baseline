# Baseline Repository for building pytorch training packages


## Installation

Build Conda Environment

```angular2html
 conda env create -f environment.yml
```

activate environment and install library

```
pip install -e .
```


## Examples

### training example

The example trains a Behavioural Cloning policy with a FCNN
```angular2html
python scripts/train_example.py
```

You can visualize the training process in tensorboard
```angular2html
cd logs/simple_exp && tensorboard --logdir=summaries
```