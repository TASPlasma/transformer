from config import Config

class MLP(nn.Module):
    config: Config
    block: bool = False

    def setup(self):
        cfg = self.config
        array_sizes = cfg.ff_sizes if self.block else cfg.sizes
        self.layers = [nn.Dense(features=size) for size in array_sizes]
        self.activations = [nn.relu for _ in array_sizes[:-1]] + [lambda x: x]

    def __call__(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x