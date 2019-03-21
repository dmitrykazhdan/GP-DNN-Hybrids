from __future__ import print_function
import keras
from keras.datasets import mnist
from keras import backend as K
import gpflow
import os, ssl
from keras.models import Model, load_model
import numpy as np
from keras import backend
import matplotlib.pyplot as plt



if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context



model_path = '...'

model_fname = model_path + 'mnist_model.h5'

nmnist_path = '...'




def run_GPC(model_name):



    # Extract subset of testing samples
    # indices = np.arange(x_test.shape[0])
    # np.random.shuffle(indices)
    # indices = indices[:500]
    # x_test = x_test[indices]
    # y_test = y_test[indices]

    # Load noisy test samples
    # x_test, y_test = load_NMNIST(nmnist_path)
    # adv_x = x_test


    # Load the CNN model
    backend.set_learning_phase(False)
    loaded_model = load_model(model_name)


    # Run the test cases through the model
    pred = np.argmax(loaded_model.predict(x_test), axis=1)
    acc = np.mean(np.equal(pred, y_test))
    print("Initial accuracy: ", acc)


    # Generate adversarial samples
    adv_x = get_FGSM_samples(loaded_model=loaded_model, samples=x_test, eps=0.1)
    adv_x = get_DeepFool_samples(loaded_model=loaded_model, samples=x_test, max_iter=100)
    # adv_x = get_BIM_samples(loaded_model=loaded_model, samples=x_test, nb_iter=5)

    # Run adversarial test cases through original model
    adv_pred = np.argmax(loaded_model.predict(adv_x), axis=1)
    adv_acc = np.mean(np.equal(adv_pred, y_test))

    print("Adversarial accuracy: ", adv_acc)


    # Create new model which ignores the last two layers
    new_model = Model(loaded_model.inputs, loaded_model.layers[-3].output)
    new_model.set_weights(loaded_model.get_weights())


    # Run the datasets through the model
    gpc_x_train = new_model.predict(x_train).astype('float64')
    gpc_x_test = new_model.predict(x_test).astype('float64')
    gpc_adv_x = new_model.predict(adv_x).astype('float64')
    gpc_y_train = np.array(y_train.reshape(-1, 1)).astype('float64')


#----------------------------------Demonstration of single case-----------------------------------------

    n_features = 128
    n_samples = 400


    simple_clean_p, simple_clean_v, simple_adv_p, simple_adv_v = \
        run_simple_example(x_train=gpc_x_train, y_train=gpc_y_train,
                           x_test=gpc_x_test, adv_test=gpc_adv_x,
                           n_samples=n_samples, n_features=n_features)

    simple_acc = np.mean(np.equal(simple_clean_p, y_test))
    print("GPC clean accuracy: ", simple_acc)
    print("GPC clean average variance: ", np.mean(simple_clean_v))


    n_test_samples = len(y_test)
    n_correct_samples = np.equal(simple_adv_p, y_test).sum()
    print("Adv correctly classified: ", n_correct_samples/n_test_samples)


    exit(-1)


#--------------------------------------------------------------------------------------------------

    def run_ensembles(n_models, n_samples, n_features, max_var, x_train, y_train, x_test, adv_test, y_test):

        gpc_ensemble = train_GPC_models(n_models=n_models, n_samples=n_samples, n_features=n_features,
                                        x_train=x_train, y_train=y_train)

        # Predict adverse test data labels
        adv_p, adv_v = get_ensemble_predictions(models=gpc_ensemble, n_models=n_models,
                                                max_var=max_var, test_samples=adv_test)

        gpc_acc = np.mean(np.equal(adv_p, y_test))
        print("GPC adversarial accuracy: ", gpc_acc)

        n_correct = np.equal(adv_p, y_test).sum()
        n_incorrect = len(y_test) - n_correct
        n_high_p_and_v = (adv_p == -2).sum()
        n_low_p = (adv_p == -1).sum()
        n_misclassified = n_incorrect - n_high_p_and_v - n_low_p
        print("High Probability and High variance: ",  n_high_p_and_v / n_incorrect)
        print("Low probability: ", n_low_p / n_incorrect)
        print("Misclassified: ", n_misclassified / n_incorrect)



        # Predict clean test data labels
        clean_p, clean_v = get_ensemble_predictions(models=gpc_ensemble, n_models=n_models,
                                                    max_var=max_var, test_samples=x_test)

        gpc_acc = np.mean(np.equal(clean_p, y_test))
        print("GPC accuracy: ", gpc_acc)



    # Train the GPC ensemble
    max_var = 0.05
    n_models = 1
    n_samples = 200
    n_features = 128


    run_ensembles(n_models, n_samples, n_features, max_var, gpc_x_train, gpc_y_train, gpc_x_test, gpc_adv_x, y_test)
    print("")
    print("")



# train_MNIST_classifier(model_fname)
run_GPC(model_fname)





