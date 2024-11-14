from Algorithms import TreeInitialization, TreeEmbedding
from Preprocessing import DataPreprocessing
import numpy as np


class NeuralNetwork:
    def __init__(self, X, hidden_neurons, tree):
        self.input_size, self.hidden_size, self.output_size = self.get_layer_size(X, hidden_neurons, tree)
        self.params = self.initialize_params()


    def get_layer_size(self, X, hidden_neurons, tree):
        input_size = X.shape[0]
        hidden_size = hidden_neurons
        output_size = len(tree.leaves) - 1

        return input_size, hidden_size, output_size


    def initialize_params(self, init_type='Xavier', seed=123):
        np.random.seed(seed)
        params = {}

        if init_type == 'Xavier':
            sd = np.sqrt(2./(self.hidden_size + self.input_size))
        elif init_type == 'He':
            sd = np.

