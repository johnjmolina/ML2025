from flax import nnx
import jax.numpy as np


class MLP(nnx.Module):
    def __init__(self, dims, *, activation_fn=nnx.tanh, output_fn=np.squeeze, **kwargs):
        r"""Initializes MLP / Feed-Forward Network using Linear Hidden Layers
        Args:
          dims: [n_in, n_1, n_2, ..., n_out] dimensions of each layer
          activation_fn : non-linear activation function for hidden layers
          output_fn : activation function for last output layer

        Keyword Args:
          use_bias : whether to add bias to the output (default:True)
          dtype : dtype of the computation
          param_dtype : dtype for parameter initializers (default: float32)
          precision : numerical precision for computation
          kernel_init: initializer function for the weight matrix
          bias_init : initializer function for the bias
          dot_general: dot product function
          rngs: rng key
        """
        layers = []
        n = len(dims)
        for i in range(n - 1):
            layers.append(nnx.Linear(dims[i], dims[i + 1], **kwargs))
        self.layers = layers
        self.activation_fn = activation_fn
        self.output_fn = output_fn

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        return self.output_fn(self.layers[-1](x))
