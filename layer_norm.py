import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
from config import Config


class LayerNorm(eqx.Module):
    config: Config

    def __call__(self):
        """

        """
        cfg = self.config
        pass
