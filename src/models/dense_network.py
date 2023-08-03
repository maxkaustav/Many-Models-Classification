from typing import Any
import tensorflow as tf


class DenseNetwork:
    def __init__(self,
                 n_layers:int,
                 hidden_units:[int],
                 dropout:float,
                 labels:int,
                 activation=None,
                 kernel = 'glorot_uniform',
                 bias='zero',
                 ):
        
        """
        Builds a Neural Network Based Classification Model
        
        Parameters:

            n_layers : Number of hidden layers 
            hidden_units : A list of hidden units per layer
            dropout : add dropout after each layer
            labels : number of categories to clasify
            activation (Optional) : activation function for each layer. Default : None
            kernel (Optional) : kernel initializer for each layer. They are strings for Tensorflow kernel Initializer. Default : glorot_uniform
            bias (Optional) : bias initializer. Default : Zero Initializer

        Returns:
            __call__ function : Call the function to get the model
        """
        
        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.labels = labels
        self.activation = activation
        self.kernel = kernel
        self.bias = bias

        # Assert number of layers match with the given hidden units
        assert len(self.hidden_units) == self.n_layers
        
        layers = [tf.keras.layers.Dense(units,activation=self.activation,kernel_initializer=self.kernel,bias_initializer=self.bias) for units in self.hidden_units]
        
        dropout_layers =[]
        for layer in layers:
            dropout_layers.append(layer)
            if dropout:
                dropout_layers.append(tf.keras.layers.Dropout(self.dropout))
            
        dropout_layers.append(tf.keras.layers.Dense(self.labels))
        
        self.model = tf.keras.Sequential(dropout_layers)
        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        
        return self.model