import gpflow
import numpy as np




def run_single_GPC(x_train, y_train, x_test, adv_test, n_samples, n_features):

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
    clean_pred = np.argmax(clean_pred, axis=1)

    adv_pred, _ = gpc.predict_y(adv_test)
    _, adv_var = gpc.predict_f(adv_test)
    adv_pred = np.argmax(adv_pred, axis=1)

    clean_pred_vars, adv_pred_vars = [], []

    for i in range(len(x_test)):
        clean_pred_vars.append(clean_var[i, clean_pred[i]])
        adv_pred_vars.append(adv_var[i, adv_pred[i]])

    return clean_pred, clean_pred_vars, adv_pred, adv_pred_vars
