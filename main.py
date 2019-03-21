from __future__ import print_function
import os, ssl
from keras.models import Model, load_model
import numpy as np
from keras import backend
from utils.utils import load_MNIST_data, load_NMNIST_test
from gpc_models.single_GP import run_single_GPC
from gpc_models.ensemble_GP import run_ensembles
from sample_generator.adv_sample_generator import get_DeepFool_samples, get_FGSM_samples, get_BIM_samples
import yaml



if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context




def run_GPC(test_data = "noisy", use_ensembles=False):

    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    model_name = cfg['model_name']
    nmnist_path = cfg['nmnist_path']


    # Load the CNN model
    backend.set_learning_phase(False)
    loaded_model = load_model(model_name)

    # Create new model which ignores the last two layers
    new_model = Model(loaded_model.inputs, loaded_model.layers[-3].output)
    new_model.set_weights(loaded_model.get_weights())

    # Load MNIST data
    x_train, y_train, x_test, y_test = load_MNIST_data()

    # Extract subset of testing samples
    indices = np.arange(x_test.shape[0])
    np.random.shuffle(indices)
    indices = indices[:500]
    x_test = x_test[indices]
    y_test = y_test[indices]

    # Select test samples

    if test_data == "noisy":
        x_test, y_test = load_NMNIST_test(nmnist_path)
        adv_x = x_test

    elif test_data == "fgsm":
        adv_x = get_FGSM_samples(loaded_model=loaded_model, samples=x_test, eps=0.1)

    elif test_data == "bim":
        adv_x = get_BIM_samples(loaded_model=loaded_model, samples=x_test, nb_iter=5)

    elif test_data == "deepfool":
        adv_x = get_DeepFool_samples(loaded_model=loaded_model, samples=x_test, max_iter=100)

    else:
        raise ValueError("Error. Incorrect test sample type entered...")


    # Run the test cases through the model
    pred = np.argmax(loaded_model.predict(x_test), axis=1)
    acc = np.mean(np.equal(pred, y_test))
    print("Original model accuracy: ", acc)


    if test_data != "noisy":
        # Run adversarial test cases through original model
        adv_pred = np.argmax(loaded_model.predict(adv_x), axis=1)
        adv_acc = np.mean(np.equal(adv_pred, y_test))
        print("Original model adversarial accuracy: ", adv_acc)


    # Extract high-level dataset features from model
    gpc_x_train = new_model.predict(x_train).astype('float64')
    gpc_x_test = new_model.predict(x_test).astype('float64')
    gpc_adv_x = new_model.predict(adv_x).astype('float64')
    gpc_y_train = np.array(y_train.reshape(-1, 1)).astype('float64')



    if not use_ensembles:

        n_features = 128
        n_samples = 400

        predcition, variance, adv_p, adv_v = \
            run_single_GPC(x_train=gpc_x_train, y_train=gpc_y_train,
                               x_test=gpc_x_test, adv_test=gpc_adv_x,
                               n_samples=n_samples, n_features=n_features)

        acc = np.mean(np.equal(predcition, y_test))
        print("GPC accuracy: ", acc)
        print("GPC average variance: ", np.mean(variance))


        n_test_samples = len(y_test)
        n_correct_samples = np.equal(adv_p, y_test).sum()
        print("Adv correctly classified: ", n_correct_samples/n_test_samples)



    else:
        max_var = 0.05
        n_models = 4
        n_samples = 200
        n_features = 128

        predcition, variance, adv_p, adv_v= \
            run_ensembles(n_models, n_samples, n_features, max_var, gpc_x_train, gpc_y_train, gpc_x_test, gpc_adv_x, y_test)



        gpc_acc = np.mean(np.equal(predcition, y_test))
        print("GPC accuracy: ", gpc_acc)
        print("GPC average variance: ", np.mean(variance))

        gpc_acc = np.mean(np.equal(adv_p, y_test))
        print("GPC adversarial accuracy: ", gpc_acc)

        n_correct = np.equal(adv_p, y_test).sum()
        n_incorrect = len(y_test) - n_correct
        n_high_p_and_v = (adv_p == -2).sum()
        n_low_p = (adv_p == -1).sum()
        n_misclassified = n_incorrect - n_high_p_and_v - n_low_p
        print("High Probability and High variance: ", n_high_p_and_v / n_incorrect)
        print("Low probability: ", n_low_p / n_incorrect)
        print("Misclassified: ", n_misclassified / n_incorrect)


run_GPC()





