from keras import backend
from cleverhans.attacks import FastGradientMethod, DeepFool, BasicIterativeMethod
from cleverhans.utils_keras import KerasModelWrapper


def get_FGSM_samples(loaded_model, samples, eps):

    sess = backend.get_session()
    wrap = KerasModelWrapper(loaded_model)

    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': eps,
                   'clip_min': 0.,
                   'clip_max': 1.}

    adv_x = fgsm.generate_np(samples, **fgsm_params)

    return adv_x


def get_DeepFool_samples(loaded_model, samples, max_iter):

    sess = backend.get_session()
    wrap = KerasModelWrapper(loaded_model)

    deepfool = DeepFool(wrap, sess=sess)
    deepfool_params = {'max_iter' : max_iter,
                    'clip_min': 0.,
                   'clip_max': 1.,
                    'nb_candidate': 10}

    adv_x = deepfool.generate_np(samples, **deepfool_params)

    return adv_x


def get_BIM_samples(loaded_model, samples, nb_iter):

    sess = backend.get_session()
    wrap = KerasModelWrapper(loaded_model)

    bim = BasicIterativeMethod(wrap, sess=sess)
    bim_params = {'eps_iter': 0.05,
                  'nb_iter': nb_iter,
                  'clip_min': 0.,
                  'clip_max': 1.}

    adv_x = bim.generate_np(samples, **bim_params)

    return adv_x
