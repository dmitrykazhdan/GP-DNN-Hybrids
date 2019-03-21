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



# Get variations of predictions only
def get_variations(variations, predictions):

    selected_vars = []

    for i in range(len(predictions)):
        selected_vars.append(variations[i, predictions[i]])

    selected_vars = np.array(selected_vars)

    return selected_vars


def get_ensemble_predictions(models, n_models, test_samples, max_var):

    def majority_vote(x):
        pr = (np.bincount(x).max() / n_models)

        if pr > 0.0:
            return np.bincount(x).argmax()
        else:
            return -1


    for i in range(n_models):

        if i == 0:
            new_p, _ = models[i].predict_y(test_samples)
            _, new_v = models[i].predict_f(test_samples)

            predictions = np.argmax(new_p, axis=1).reshape(-1, 1)
            prediction_values = np.max(new_p, axis=1).reshape(-1, 1)
            variations = get_variations(new_v, predictions)

        else:
            new_p, _ = models[i].predict_y(test_samples)
            _, new_v = models[i].predict_f(test_samples)

            new_p_vals = np.max(new_p, axis=1).reshape(-1, 1)
            new_p = np.argmax(new_p, axis=1).reshape(-1, 1)
            new_v = get_variations(new_v, new_p)

            predictions = np.hstack((predictions, new_p))
            prediction_values = np.hstack((prediction_values, new_p_vals))
            variations = np.hstack((variations, new_v))

    majority_predictions = np.apply_along_axis(func1d=majority_vote, axis=1, arr=predictions)
    mean_variations = np.apply_along_axis(func1d=np.max, axis=1, arr=variations)
    mean_pred_values = np.apply_along_axis(func1d=np.mean, axis=1, arr=prediction_values)


    # Reject samples with high variance or low probability
    for i in range(len(test_samples)):

        if mean_pred_values[i] < 0.6:
            majority_predictions[i] = -1

        elif mean_variations[i] > max_var:
            majority_predictions[i] = -2

    return majority_predictions, mean_variations



def train_GPC_models(n_models, n_samples, n_features, x_train, y_train):

    models = {}

    for i in range(n_models):

        # Randomly extract set of training samples
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        indices = indices[:n_samples]
        gpc_x_train = x_train[indices, :]
        gpc_y_train = y_train[indices, :]

        # Randomly select set of features
        features = np.arange(128)
        np.random.shuffle(features)
        features = features[:n_features]

        # Use two different kernels
        if i % 2 == 0:
            kernel = gpflow.kernels.RBF(input_dim=n_features, ARD=True, active_dims=features)
        else:
            kernel = gpflow.kernels.Matern12(input_dim=n_features, ARD=True, active_dims=features)


        # Create the model
        gpc = gpflow.models.VGP(gpc_x_train, gpc_y_train, num_latent=10,
                                kern=kernel,
                                likelihood=gpflow.likelihoods.MultiClass(10))



        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(gpc, maxiter=100)

        models[i] = gpc

    return models



def run_simple_example(x_train, y_train, x_test, adv_test, n_samples, n_features):

    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    indices = indices[:n_samples]
    gpc_x_train = x_train[indices]
    gpc_y_train = y_train[indices]
    features = np.arange(128)

    gpc = gpflow.models.SVGP(gpc_x_train, gpc_y_train, num_latent=10,
                            kern=gpflow.kernels.Matern52(input_dim=128, ARD=False, active_dims=features), Z=gpc_x_train[::5].copy(),
                             likelihood=gpflow.likelihoods.MultiClass(10))

    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(gpc, maxiter=1000)

    clean_pred, _ = gpc.predict_y(x_test)
    _, clean_var = gpc.predict_f(x_test)
    clean_pred_vals = np.max(clean_pred, axis=1)
    clean_pred = np.argmax(clean_pred, axis=1)

    adv_pred, _ = gpc.predict_y(adv_test)
    _, adv_var = gpc.predict_f(adv_test)
    adv_pred_vals = np.max(adv_pred, axis=1)
    adv_pred = np.argmax(adv_pred, axis=1)

    clean_pred_vars, adv_pred_vars = [], []

    for i in range(len(x_test)):
        clean_pred_vars.append(clean_var[i, clean_pred[i]])
        adv_pred_vars.append(adv_var[i, adv_pred[i]])

    return clean_pred, clean_pred_vars, adv_pred, adv_pred_vars



def run_GPC(model_name, n_test_samples=500):



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





