# Deep-B3
A multi-model framework for blood-brain barrier permeability discovery
## Dependencies

This repository has been tested on CentOS 7.5. We strongly recommend you to have [Anaconda3](https://www.anaconda.com/distribution/) installed, which contains most of required packages for running this model.

### Must installed packages or softwares

- FastAI  1.0.61

- PyTorch 1.9.0

- Torchvision 0.10.0

- Pandas 1.3.2

## Get Started

### Generate features from the SMILES file and train a new deep-b3 model
- SMILES file is a csv format, within head (id,smi, label)
- The train and test feature file was stored in `./train` and `./test`, respectively.
- The iamges for training and testing the model were stored in `./train_images` and `./test_images`, respectively.
- You can run the script `deep-b3.py` to train a new model for the data or on your new data, and the models will be stored in `./models`.  Sample code:
```
python deep-b3.py train --train_feature train.csv --test_feature test.csv --bs 64 --vis_out 512 --text_out 128
```

### test the pre-trian deep-b3 model on the test data
- You can run the script `deep-b3-test.py` to test the pre-trained model on the test data used in this study, the results are stored in file `result.csv`.  Sample code:
```
python deep-b3-test.py
```
