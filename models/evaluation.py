"""Import keras library."""
from keras import backend as K


def vae_loss(x, t_decoded):
    """Total loss for the plain UAE."""
    return K.mean(reconstruction_loss(x, t_decoded))


def reconstruction_loss(x, t_decoded):
    """Reconstruction loss for the plain UAE."""
    return K.sum((K.batch_flatten(x) - K.batch_flatten(t_decoded)) ** 2,
                 axis=-1)


def relative_error(x, t_decoded):
    """Reconstruction loss for the plain UAE."""
    return K.mean(K.abs(x - t_decoded) / x)
