from tensorflow.keras import layers
import tensorflow.keras.backend as K


class RevIN(layers.Layer):
    """Reversible Instance Normalization for Accurate Time-Series Forecasting
       against Distribution Shift, ICLR2022.

    Parameters
    ----------
    eps: float, a value added for numerical stability, default 1e-5.
    affine: bool, if True(default), RevIN has learnable affine parameters.
    """
    def __init__(self, eps=1e-5, affine=True, **kwargs):
        super(RevIN, self).__init__(**kwargs)
        self.eps = eps
        self.affine = affine

    def build(self, input_shape):
        self.affine_weight = self.add_weight(name='affine_weight',
                                 shape=(1, input_shape[-1]),
                                 initializer='ones',
                                 trainable=True)

        self.affine_bias = self.add_weight(name='affine_bias',
                                 shape=(1, input_shape[-1]),
                                 initializer='zeros',
                                 trainable=True)
        super(RevIN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mode = kwargs.get('mode', None)
        if mode == 'norm':
            self._get_statistics(inputs)
            x = self._normalize(inputs)
        elif mode == 'denorm':
            x = self._denormalize(inputs)
        else:
            raise NotImplementedError('Only modes norm and denorm are supported.')
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, len(x.shape) - 1))
        self.mean = K.stop_gradient(K.mean(x, axis=dim2reduce, keepdims=True))
        self.stdev = K.stop_gradient(K.sqrt(K.var(x, axis=dim2reduce, keepdims=True) + self.eps))
        print(self.stdev)

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

    def get_config(self):
        config = {'eps': self.eps,
                  'affine': self.affine}
        base_config = super(RevIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    import tensorflow as tf

    x = tf.reshape(tf.range(0, 24), shape=(4, 3, 2))/24
    layer = RevIN(3)
    y = layer(x, mode='norm')
    z = layer(y, mode='denorm')

    print(x)
    print(y)
    print(z)