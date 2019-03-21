# GP-DNN-Hybrids

This projects explores a way of extracting uncertainty estimates from DNNs by combining them with Gaussian Process
Classifiers, producing hybrid models. Results obtained from this project demonstrate that these hybrid models achieve
high predictive accuracy on normal samples, whilst reporting high uncertainty on noisy samples. Furthermore, this
project demonstrates that uncertainty estimations of these hybrid models may be used for adversarial sample detection.


## Setup

### Prerequisites:

Ensure you have the following packages installed (these can all be installed with pip3):

- Keras
- Cleverhans
- GPFlow
- pyYAML
- SciPy


### MNIST Model

This project requires a trained MNIST classification model, which can be generated
using the _mnist_cnn.py_ file, found [here](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py).



## Usage

- Create MNIST model, as specified in the previous section
- Specify it's path in the _config.yml_ file
- Optionally, download the NMNIST dataset (available [here](http://yann.lecun.com/exdb/mnist/))
, and specify it's path in the _config.yml_ file
- Navigate into the repository directory
- Run _main.py_:
```python
python3 ./main.py
```



## Files/Directories

- gpc_models: directory includes files for training/running a single
Gaussian Process Classifier (_single_GP.py_), or an ensemble of such classifiers (_ensemble_GP.py_)

- sample_generator: includes the _adv_sample_generator.py_ file, 
which can be used for generating adversarial samples using the CleverHans
toolbox

- utils: includes the _utils.py_ file, which is used for loading and parsing
the MNIST and NMNIST datasets 

- config.yml: configuration file storing path names

- main.py: file used for loading and running the GPC-DNN hybrid model


