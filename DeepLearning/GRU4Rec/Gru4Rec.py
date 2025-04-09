import os
import tensorflow
from tensorflow.python.ops import rnn_cell
import pandas as pd
import numpy as np


class Gru4Rec:
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.layers = args.layers
        self.rnn_size = args.rnn_size
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.dropout_p_hidden = args.dropout_p_hidden
        self.learning_rate = args.learning_rate
        self.decay = args.decay
        self.decay_steps = args.decay_steps
        self.sigma = args.sigma
        self.init_as_normal = args.init_as_normal
        self.reset_after_session = args.reset_after_session
        self.session_key = args.session_key
        self.item_key = args.item_key
        self.time_key = args.time_key
        self.grad_cap = args.grad_cap
        self.n_items = args.n_items
        if args.hidden_act == 'tanh':
            self.hidden_act = tensorflow.nn.tanh
        else :
            raise NotImplementedError
        
        if arg.loss == 'cross_entropy':
            if args.final_act == 'tanh':
            