import gpflow
import numpy as np




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






def run_ensembles(n_models, n_samples, n_features, max_var, x_train, y_train, x_test, adv_test, y_test):

    gpc_ensemble = train_GPC_models(n_models=n_models, n_samples=n_samples, n_features=n_features,
                                    x_train=x_train, y_train=y_train)

    # Predict adverse test data labels
    adv_p, adv_v = get_ensemble_predictions(models=gpc_ensemble, n_models=n_models,
                                            max_var=max_var, test_samples=adv_test)


    # Predict clean test data labels
    clean_p, clean_v = get_ensemble_predictions(models=gpc_ensemble, n_models=n_models,
                                                max_var=max_var, test_samples=x_test)


    return clean_p, clean_v, adv_p, adv_v