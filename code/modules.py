# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
#limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper 
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

import logging
logging.basicConfig(level=logging.DEBUG)

def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist

class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self, keep_prob):
        self.keep_prob = keep_prob

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """

        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output

class BasicAttnPlusOne(object):
    def __init__(self, batch_size, context_len, hidden_size, keep_prob, key_vec_size, value_vec_size):
        self.batch_size = batch_size
        self.context_len = context_len
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

        self.rnn_cell_fw_1 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw_1 = DropoutWrapper(self.rnn_cell_fw_1, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw_1 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw_1 = DropoutWrapper(self.rnn_cell_bw_1, input_keep_prob=self.keep_prob)

        self.rnn_cell_fw_2 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw_2 = DropoutWrapper(self.rnn_cell_fw_2, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw_2 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw_2 = DropoutWrapper(self.rnn_cell_bw_2, input_keep_prob=self.keep_prob)

        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, values, values_mask, keys, keys_mask):

        with vs.variable_scope("BasicAttn"):
            # Encoder Module
            n_keys = tf.concat([keys, tf.random_uniform([tf.shape(keys)[0], 1, tf.shape(keys)[2]], minval=-0.25, maxval=0.25, dtype=tf.float32)], 1) # (batch_size, num_keys + 1, key_vec_size)
            logging.debug('Bicoattn n_keys shape: %s', n_keys.shape)
            n_keys_mask = tf.concat([keys_mask, tf.ones([tf.shape(keys_mask)[0], 1], dtype=tf.int32)], 1) # (batch_size, num_keys + 1)
            logging.debug('Bicoattn n_keys_mask shape: %s', n_keys_mask.shape)
            n_values = tf.concat([values, tf.random_uniform([tf.shape(values)[0], 1, tf.shape(values)[2]], minval=-0.25, maxval=0.25, dtype=tf.float32)], 1) # (batch_size, num_values + 1, value_vec_size)
            logging.debug('Bicoattn n_values shape: %s', n_values.shape)
            n_values_mask = tf.concat([values_mask, tf.ones([tf.shape(values_mask)[0], 1], dtype=tf.int32)], 1) # (batch_size, num_values + 1)
            logging.debug('Bicoattn n_values_mask shape: %s', n_values_mask.shape)

            # the num_keys in the comment is the origin keys' shape, which is different from the "num_keys" below; the same as num_values
            num_keys = n_keys.shape[1]
            num_values = n_values.shape[1]

            with vs.variable_scope('BiCoattn_encoder'):
                n_c_values = tf.contrib.layers.fully_connected(n_values, self.value_vec_size, activation_fn=tf.tanh) # (batch_size, num_values + 1, value_vec_size)
                logging.debug('Bicoaatn n_c_values shape: %s', n_c_values.shape)

            # Calculate attention distribution
            values_t = tf.transpose(n_c_values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values+1)
            attn_logits = tf.matmul(n_keys, values_t) # shape (batch_size, num_keys+1, num_values+1)
            attn_logits_mask = tf.expand_dims(n_values_mask, 1) # shape (batch_size, 1, num_values+1)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys+1, num_values+1). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, n_c_values)[:, :-1, :] # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

        return attn_dist, output

class BasicAttnPlusTwo(object):
    def __init__(self, batch_size, context_len, hidden_size, keep_prob, key_vec_size, value_vec_size):
        self.batch_size = batch_size
        self.context_len = context_len
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

        self.rnn_cell_fw_1 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw_1 = DropoutWrapper(self.rnn_cell_fw_1, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw_1 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw_1 = DropoutWrapper(self.rnn_cell_bw_1, input_keep_prob=self.keep_prob)

        self.rnn_cell_fw_2 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw_2 = DropoutWrapper(self.rnn_cell_fw_2, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw_2 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw_2 = DropoutWrapper(self.rnn_cell_bw_2, input_keep_prob=self.keep_prob)

        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, values, values_mask, keys, keys_mask):

        with vs.variable_scope("BasicAttn"):
            # Encoder Module
            n_keys = tf.concat([keys, tf.random_uniform([tf.shape(keys)[0], 1, tf.shape(keys)[2]], minval=-0.25, maxval=0.25, dtype=tf.float32)], 1) # (batch_size, num_keys + 1, key_vec_size)
            logging.debug('Bicoattn n_keys shape: %s', n_keys.shape)
            n_keys_mask = tf.concat([keys_mask, tf.ones([tf.shape(keys_mask)[0], 1], dtype=tf.int32)], 1) # (batch_size, num_keys + 1)
            logging.debug('Bicoattn n_keys_mask shape: %s', n_keys_mask.shape)
            n_values = tf.concat([values, tf.random_uniform([tf.shape(values)[0], 1, tf.shape(values)[2]], minval=-0.25, maxval=0.25, dtype=tf.float32)], 1) # (batch_size, num_values + 1, value_vec_size)
            logging.debug('Bicoattn n_values shape: %s', n_values.shape)
            n_values_mask = tf.concat([values_mask, tf.ones([tf.shape(values_mask)[0], 1], dtype=tf.int32)], 1) # (batch_size, num_values + 1)
            logging.debug('Bicoattn n_values_mask shape: %s', n_values_mask.shape)

            # the num_keys in the comment is the origin keys' shape, which is different from the "num_keys" below; the same as num_values
            num_keys = n_keys.shape[1]
            num_values = n_values.shape[1]

            with vs.variable_scope('BiCoattn_encoder'):
                n_c_values = tf.contrib.layers.fully_connected(n_values, self.value_vec_size, activation_fn=tf.tanh) # (batch_size, num_values + 1, value_vec_size)
                logging.debug('Bicoaatn n_c_values shape: %s', n_c_values.shape)

            # Calculate attention distribution
            values_t = tf.transpose(n_c_values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values+1)
            attn_logits = tf.matmul(n_keys, values_t) # shape (batch_size, num_keys+1, num_values+1)
            attn_logits_mask = tf.expand_dims(n_values_mask, 1) # shape (batch_size, 1, num_values+1)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys+1, num_values+1). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, n_c_values)[:, :-1, :] # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            g = tf.concat([keys, output], 2) # shape (batch_size, num_keys, value_vec_size * 2)
            g = tf.contrib.layers.fully_connected(g, num_outputs=self.hidden_size) # shape (batch_size, num_keys, value_vec_size / 2)
            logging.debug('g shape: %s', g.shape)

            # Modeling Layer(2 layers)
            with vs.variable_scope('BiCoattn_modeling_1'):
                (fw_out_1, bw_out_1), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw_1, self.rnn_cell_bw_1, g, dtype=tf.float32) # (batch_size, num_keys, key_vec_size / 2)
                out_1 = tf.concat([fw_out_1, bw_out_1], 2)# (batch_size, num_keys, key_vec_size)
                logging.debug('Bicoattn out_1 shape: %s', out_1.shape)
            with vs.variable_scope('BiCoattn_modeling_2'):
                (fw_out_2, bw_out_2), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw_2, self.rnn_cell_bw_2, out_1, dtype=tf.float32) # (batch_size, num_keys, key_vec_size / 2)
                out_2 = tf.concat([fw_out_2, bw_out_2], 2) # (batch_size, num_keys, key_vec_size)
                logging.debug('Bicoattn out_2 shape: %s', out_2.shape)

            # Output Layer
            g_m = tf.concat([g, out_2], 2) # (batch_size, num_keys, key_vec_size * 1.5)
            logging.debug('Bicoattn g_m shape: %s', g_m.shape)

            with vs.variable_scope('BiCoattn_output_layer2'):
                (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, g_m, dtype=tf.float32) # (batch_size, num_keys, key_vec_size / 2)
                out = tf.concat([fw_out, bw_out], 2) # batch_size, num_keys, key_vec_size
                logging.debug('Bicoattn out shape: %s', out.shape)
            g_m2 = tf.concat([g, out], 2) # (batch_size, num_keys, key_vec_size * 1.5)
            logging.debug('Bicoattn g_m2 shape: %s', g_m2.shape)

        return g_m, g_m2
        
#Reference from https://github.com/unilight/R-NET-in-Tensorflow/blob/master/Models/model_rnet.py 
def mat_weight_mul(mat, weight):
    # [batch_size, n, m] * [m, p] = [batch_size, n, p]
    mat_shape = mat.get_shape().as_list()
    weight_shape = weight.get_shape().as_list()
    mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]]) # [batch_size * n, m]
    mul = tf.matmul(mat_reshape, weight) # [batch_size * n, p]
    return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])
#Discuss with Tianpe
class GatedAttn(object):

    def __init__(self, keep_prob, key_vec_size, value_vec_size, hidden_size):

        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size
        self.hidden_size = hidden_size

    def build_graph(self, values, values_mask, keys):

        with vs.variable_scope("GatedAttn"):

            v_P = []
            W_uQ = tf.get_variable('W_uQ', shape = [self.value_vec_size, self.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            W_uP = tf.get_variable('W_uP', shape = [self.key_vec_size, self.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            W_vP = tf.get_variable('W_vP', shape = [self.hidden_size, self.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            v = tf.get_variable('v', initializer = tf.truncated_normal([self.hidden_size, 1]))
            W_g = tf.get_variable('W_g', shape = [self.key_vec_size + self.value_vec_size, self.key_vec_size + self.value_vec_size], initializer = tf.contrib.layers.xavier_initializer())

            QP_match_cell = rnn_cell.GRUCell(self.hidden_size)
            QP_match_cell = DropoutWrapper(QP_match_cell, input_keep_prob=self.keep_prob)
            QP_match_state = QP_match_cell.zero_state(tf.shape(values)[0], tf.float32)

            W_uQ_u_Q = mat_weight_mul(values, W_uQ) 
            for t in range(keys.shape[1]): # context_len
                W_uP_u_tP = mat_weight_mul(keys[:,t:(t+1),:], W_uP) 
                logging.debug('W_uP_u_tP shape: %s', W_uP_u_tP.shape)
                if t == 0:
                    tanh = tf.tanh(W_uQ_u_Q + W_uP_u_tP)
                else:
                    W_vP_v_t1P = mat_weight_mul(tf.expand_dims(v_P[t-1], 1), W_vP)
                    logging.debug('W_vP_v_t1P shape: %s', W_vP_v_t1P.shape)
                    tanh = tf.tanh(W_uQ_u_Q + W_uP_u_tP + W_vP_v_t1P)
                logging.debug('tanh shape: %s', tanh.shape)
                s_t = tf.squeeze(mat_weight_mul(tanh, v), [2]) 
                logging.debug('s_t shape: %s', s_t.shape)
                _, a_t = masked_softmax(s_t, values_mask,1) 
                logging.debug('a_t shape: %s', a_t.shape)
                c_t = tf.matmul(tf.expand_dims(a_t, 1), values)
                logging.debug('c_t shape: %s', c_t.shape)
                c_t = tf.nn.dropout(c_t, self.keep_prob)
                assert c_t.shape[1:] == [1, self.value_vec_size]
                u_tP_c_t = tf.concat([keys[:,t:(t+1),:], c_t], 2) 
                logging.debug('u_tP_c_t shape: %s', u_tP_c_t.shape)
                assert u_tP_c_t.shape[1:] == [1, self.value_vec_size + self.key_vec_size]
                g_t = tf.sigmoid(mat_weight_mul(u_tP_c_t, W_g)) 
                logging.debug('g_t shape: %s', g_t.shape)
                u_tP_c_t_star = tf.squeeze(u_tP_c_t * g_t, [1]) 
                logging.debug('u_tP_c_t_star shape: %s', u_tP_c_t_star.shape)
                with tf.variable_scope("QP_match"):
                    if t > 0:
                        tf.get_variable_scope().reuse_variables()
                    output, QPmatch_state = QP_match_cell(u_tP_c_t_star, QP_match_state)
                    v_P.append(output) 
            v_P = tf.stack(v_P, 1) 
            logging.debug('v_P shape: %s', v_P.shape)           
        return v_P, a_t

class SelfAttn(object):


    def __init__(self, keep_prob, vec_size, hidden_size):

        self.keep_prob = keep_prob
        self.vec_size = vec_size
        self.hidden_size = hidden_size

    def build_graph(self, values, values_mask):

        with vs.variable_scope("SelfAttn"):
            star = []
            W_vP = tf.get_variable('W_vP', shape = [self.vec_size, self.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            W_vPt = tf.get_variable('W_VPt', shape = [self.vec_size, self.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            v = tf.get_variable('v', initializer = tf.truncated_normal([self.hidden_size, 1]))
            W_g = tf.get_variable('W_g', shape = [self.vec_size * 2, self.vec_size * 2], initializer = tf.contrib.layers.xavier_initializer())

            QP_match_cell = rnn_cell.GRUCell(self.hidden_size)
            QP_match_cell = DropoutWrapper(QP_match_cell, input_keep_prob=self.keep_prob)
            QP_match_state = QP_match_cell.zero_state(tf.shape(values)[0], tf.float32)

            W_vP_v_P = mat_weight_mul(values, W_vP)
            logging.debug('W_vP_v_P shape: %s', W_vP_v_P.shape)
            for t in range(values.shape[1]): 
                W_vPt_v_P =mat_weight_mul(values[:,t:(t+1),:], W_vPt)
                logging.debug('W_vPt_v_P shape: %s', W_vPt_v_P.shape)
                tanh = tf.tanh(W_vP_v_P + W_vPt_v_P)
                logging.debug('tanh shape: %s', tanh.shape)
                s_t = tf.squeeze(mat_weight_mul(tanh, v), [2]) 
                _, a_t = masked_softmax(s_t, values_mask, 1) 
                c_t = tf.matmul(tf.expand_dims(a_t, 1), values)
                logging.debug('c_t shape: %s', c_t.shape)
                c_t = tf.nn.dropout(c_t, self.keep_prob)
                u_tP_c_t = tf.concat([values[:,t:(t+1),:], c_t], 2) 
                logging.debug('u_tP_c_t shape: %s', u_tP_c_t.shape)
                g_t = tf.sigmoid(mat_weight_mul(u_tP_c_t, W_g)) 
                u_tP_c_t_star = tf.squeeze(u_tP_c_t * g_t, [1]) 
                star.append(u_tP_c_t_star)
            star = tf.stack(star, 1)
            logging.debug('star shape: %s', star.shape)
            encoder = RNNEncoder(self.hidden_size, self.keep_prob)
            h_P = encoder.build_graph(star, values_mask) 
            logging.debug('h_P shape: %s', h_P.shape)                   
        return h_P, a_t


class Output_Rnet(object):


    def __init__(self, keep_prob, context_size, question_size, hidden_size):

        self.keep_prob = keep_prob
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.question_size = question_size

    def build_graph(self, context, question, context_mask, question_mask):

        with vs.variable_scope("Output_Rnet"):
            W_hP = tf.get_variable('W_hP', shape = [self.context_size, self.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            W_ha = tf.get_variable('W_ha', shape = [self.question_size, self.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            
            W_uQ = tf.get_variable('W_vP', shape = [self.question_size, 2 * self.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            VrQ = tf.get_variable('VrQ', shape = [question.shape[1], self.hidden_size],  initializer = tf.contrib.layers.xavier_initializer())
            W_vQ = tf.get_variable('W_vQ', shape = [self.hidden_size, 2 * self.hidden_size],  initializer = tf.contrib.layers.xavier_initializer())

            v1 = tf.get_variable('v1', initializer = tf.truncated_normal([2 * self.hidden_size, 1]))
            v2 = tf.get_variable('v2', initializer = tf.truncated_normal([self.hidden_size, 1]))

            ptr_cell = rnn_cell.GRUCell(self.context_size)
            ptr_cell = DropoutWrapper(ptr_cell, input_keep_prob=self.keep_prob)

            W_uQ_u_Q = mat_weight_mul(question, W_uQ) 
            logging.debug('W_uQ_u_Q shape: %s', W_uQ_u_Q.shape)
            W_vQ_V_rQ = tf.expand_dims(tf.matmul(VrQ, W_vQ), 0) 
            logging.debug('W_vQ_V_rQ shape: %s', W_vQ_V_rQ.shape)
            tanh1 = tf.tanh(W_uQ_u_Q + W_vQ_V_rQ) 
            logging.debug('tanh1 shape: %s', tanh1.shape)
            s1 = tf.squeeze(mat_weight_mul(tanh1, v1), [2]) 
            _, a = masked_softmax(s1, question_mask, 1)
            logging.debug('a shape: %s', a.shape)
            r1 = tf.matmul(tf.expand_dims(a, 1), question)
            r1 = tf.nn.dropout(r1, self.keep_prob) 
            logging.debug('r1 shape: %s', r1.shape)

            W_ha_r1 = mat_weight_mul(r1, W_ha) 
            W_hP_h_P1 = mat_weight_mul(context, W_hP) 
            tanh2 = tf.tanh(W_ha_r1 + W_hP_h_P1) 
            s2 = tf.squeeze(mat_weight_mul(tanh2, v2), [2]) 
            logits_start, probdist_start = masked_softmax(s2, context_mask, 1)
            c = tf.squeeze(tf.matmul(tf.expand_dims(probdist_start, 1), context), [1]) # (batch_size, context_size)
            r2, _ = ptr_cell(c, tf.squeeze(r1, [1]))
            r2 = tf.expand_dims(r2, 1)


            W_ha_r2 = mat_weight_mul(r2, W_ha) 
            W_hP_h_P = mat_weight_mul(context, W_hP) 
            tanh3 = tf.tanh(W_ha_r2 + W_hP_h_P) 
            s3 = tf.squeeze(mat_weight_mul(tanh3, v2), [2]) 
            logits_end, probdist_end = masked_softmax(s3, context_mask, 1) 
                    
        return logits_start, probdist_start, logits_end, probdist_end, a

# @zhengyang
class BiDAF(object):

    def __init__(self, hidden_size, keep_prob, key_vec_size, value_vec_size):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

        self.rnn_cell_fw_1 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw_1 = DropoutWrapper(self.rnn_cell_fw_1, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw_1 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw_1 = DropoutWrapper(self.rnn_cell_bw_1, input_keep_prob=self.keep_prob)

        self.rnn_cell_fw_2 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw_2 = DropoutWrapper(self.rnn_cell_fw_2, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw_2 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw_2 = DropoutWrapper(self.rnn_cell_bw_2, input_keep_prob=self.keep_prob)

        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, values, values_mask, keys, keys_mask):

        with vs.variable_scope("BiDAF"):
            num_values = values.shape[1]
            num_keys = keys.shape[1]
            # Start from Attention Flow Layer
            # Calc Similarity Matrix(more efficient way, maybe it's easier to understand from the part of notation below)
            t_keys = tf.tile( tf.expand_dims(keys, 2), [1, 1, num_values, 1] ) # (batch_size, num_keys, num_values, key_vec_size)
            logging.debug('BiDAF tiled keys shape: %s', t_keys.shape)
            t_values = tf.tile( tf.expand_dims(values, 1), [1, num_keys, 1, 1] ) # (batch_size, num_keys, num_values, value_vec_size)
            logging.debug('BiDAF tiled values shape: %s', t_values.shape)
            e_keys = tf.expand_dims(keys, 2) # (batch_size, num_keys, 1, key_vec_size)
            logging.debug('BiDAF expanded keys shape: %s', e_keys.shape)
            e_values = tf.expand_dims(values, 1) # (batch_size, num_keys, num_values, value_vec_size)
            logging.debug('BiDAF expaned values shape: %s', e_values.shape)

            tmp = tf.multiply(e_keys, e_values) # (batch_size, num_keys, num_values, value_vec_size)
            logging.debug('BiDAF tmp shape: %s', tmp.shape)

            with vs.variable_scope('BiDAF_matrix_1'):
                attn_logits_1 = tf.contrib.layers.fully_connected(tf.reshape(tmp, [-1, num_keys * num_values, self.value_vec_size]), 1, activation_fn=None) # (batch_size, num_keys * num_values, 1)
            with vs.variable_scope('BiDAF_matrix_2'):
                attn_logits_2 = tf.contrib.layers.fully_connected(tf.reshape(t_keys, [-1, num_keys * num_values, self.value_vec_size]), 1, activation_fn=None) # (batch_size, num_keys * num_values, 1)
            with vs.variable_scope('BiDAF_matrix_3'):
                attn_logits_3 = tf.contrib.layers.fully_connected(tf.reshape(t_values, [-1, num_keys * num_values, self.value_vec_size]), 1, activation_fn=None) # (batch_size, num_keys * num_values, 1)
            attn_logits = attn_logits_1 + attn_logits_2 + attn_logits_3 # (batch_size, num_keys * num_values, 1)
            logging.debug('BiDAF attn_logits shape: %s', attn_logits.shape)
            attn_logits = tf.reshape(attn_logits, [-1, num_keys, num_values])
            logging.debug('BiDAF reshaped attn_logits shape: %s', attn_logits.shape)

            # Calc Similarity Matrix(less efficient way)
            # intermediate = tf.concat([t_keys, t_values, tf.multiply(t_keys, t_values)], 3) # (batch_size, num_keys, num_values, 3 * value_vec_size)
            # logging.debug('BiDAF intermediate shape: %s', intermediate.shape)
            # intermediate = tf.reshape(intermediate, [-1, num_keys * num_values, 3 * self.value_vec_size]) # (batch_size, num_keys * num_values, 3 * value_vec_size)
            # logging.debug('BiDAF reshaped intermediate shape: %s', intermediate.shape)
            # attn_logits = tf.contrib.layers.fully_connected(intermediate, 1, activation_fn=None) # (batch_size, num_keys * num_values, 1)
            # logging.debug('BiDAF attn_logits shape: %s', attn_logits.shape)
            # attn_logits = tf.reshape(attn_logits, [-1, num_keys, num_values]) # (batch_size, num_keys, num_values)
            # logging.debug('BiDAF reshaped attn_logits shape: %s', attn_logits.shape)

            # Context2Query Attention
            e_values_mask = tf.expand_dims(values_mask, 1) # (batch_size, 1, num_values)
            logging.debug('BiDAF e_values_mask shape: %s', e_values_mask.shape)
            _, c2q_attn_dist = masked_softmax(attn_logits, e_values_mask, 2) # (batch_size, num_keys, num_values)
            logging.debug('BiDAF c2q_attn_dist shape: %s', c2q_attn_dist.shape)
            c2q_output = tf.matmul(c2q_attn_dist, values) # (batch_size, num_keys, value_vec_size)
            logging.debug('BiDAF c2q_output shape: %s', c2q_output.shape)

            # Query2Context Attention
            # take max over rows and transpose that like BasicAttn does
            aggregated_keys = tf.transpose( tf.reduce_max(attn_logits, axis=2, keep_dims=True), perm=[0, 2, 1] ) # (batch_size, 1, num_keys)
            aggregated_keys_mask = tf.expand_dims(keys_mask, 1) # (batch_size, 1, num_keys)
            _, aggregated_keys_dist = masked_softmax(aggregated_keys, aggregated_keys_mask, 2) # (batch_size, 1, num_keys)
            aggregated_keys_dist = tf.tile(aggregated_keys_dist, [1, num_keys, 1]) # (batch_size, num_keys, num_keys)
            logging.debug('BiDAF tile aggregated_keys_dist shape: %s', aggregated_keys_dist.shape)
            q2c_output = tf.matmul( aggregated_keys_dist, keys ) # (batch_size, num_keys, num_keys) dot (batch_size, num_keys, key_vec_size) -> (batch_size, num_keys, key_vec_size)
            logging.debug('BiDAF q2c_output shape: %s', q2c_output.shape)
            # yield G by combining contextual embeddings and attention vectors
            g = tf.concat([keys, c2q_output, tf.multiply(keys, c2q_output), tf.multiply(keys, q2c_output)], 2) # (batch_size, num_keys, 4 * key_vec_size)
            logging.debug('BiDAF g shape: %s', g.shape)

            # Modeling Layer(2 layers)
            with vs.variable_scope('BiDAF_modeling_1'):
                (fw_out_1, bw_out_1), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw_1, self.rnn_cell_bw_1, g, dtype=tf.float32) # (batch_size, num_keys, key_vec_size / 2)
                logging.debug('BiDAF fw_out_1 shape: %s', fw_out_1.shape)
                logging.debug('BiDAF bw_out_1 shape: %s', bw_out_1.shape)
                out_1 = tf.concat([fw_out_1, bw_out_1], 2) # batch_size, num_keys, key_vec_size
                logging.debug('BiDAF out_1 shape: %s', out_1.shape)
            with vs.variable_scope('BiDAF_modeling_2'):
                (fw_out_2, bw_out_2), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw_2, self.rnn_cell_bw_2, out_1, dtype=tf.float32) # (batch_size, num_keys, key_vec_size / 2)
                logging.debug('BiDAF fw_out_2 shape: %s', fw_out_2.shape)
                logging.debug('BiDAF bw_out_2 shape: %s', bw_out_2.shape)
                out_2 = tf.concat([fw_out_2, bw_out_2], 2) # batch_size, num_keys, key_vec_size
                logging.debug('BiDAF out_2 shape: %s', out_2.shape)

            
            g_m = tf.concat([g, out_2], 2) # (batch_size, num_keys, key_vec_size * 5)
            logging.debug('BiDAF g_m shape: %s', g_m.shape)
            with vs.variable_scope('BiDAF_output_layer'):
                (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, g_m, dtype=tf.float32) # (batch_size, num_keys, key_vec_size / 2)
                logging.debug('BiDAF fw_out shape: %s', fw_out.shape)
                logging.debug('BiDAF bw_out shape: %s', bw_out.shape)
                out = tf.concat([fw_out, bw_out], 2) # batch_size, num_keys, key_vec_size
                logging.debug('BiDAF out shape: %s', out.shape)
            g_m2 = tf.concat([g, out], 2) # (batch_size, num_keys, key_vec_size * 5)
            logging.debug('BiDAF g_m2 shape: %s', g_m2.shape)

        return g_m, g_m2

# @zhengyang
class BiCoattn(object):

    def __init__(self, batch_size, context_len, hidden_size, keep_prob, key_vec_size, value_vec_size):
        self.batch_size = batch_size
        self.context_len = context_len
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

        self.rnn_cell_fw_1 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw_1 = DropoutWrapper(self.rnn_cell_fw_1, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw_1 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw_1 = DropoutWrapper(self.rnn_cell_bw_1, input_keep_prob=self.keep_prob)

        self.rnn_cell_fw_2 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw_2 = DropoutWrapper(self.rnn_cell_fw_2, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw_2 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw_2 = DropoutWrapper(self.rnn_cell_bw_2, input_keep_prob=self.keep_prob)

        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, values, values_mask, keys, keys_mask):

        with vs.variable_scope('BiCoattn'):
            # Encoder Module
            n_keys = tf.concat([keys, tf.random_uniform([tf.shape(keys)[0], 1, tf.shape(keys)[2]], minval=-0.25, maxval=0.25, dtype=tf.float32)], 1) # (batch_size, num_keys + 1, key_vec_size)
            logging.debug('Bicoattn n_keys shape: %s', n_keys.shape)
            n_keys_mask = tf.concat([keys_mask, tf.ones([tf.shape(keys_mask)[0], 1], dtype=tf.int32)], 1) # (batch_size, num_keys + 1)
            logging.debug('Bicoattn n_keys_mask shape: %s', n_keys_mask.shape)
            n_values = tf.concat([values, tf.random_uniform([tf.shape(values)[0], 1, tf.shape(values)[2]], minval=-0.25, maxval=0.25, dtype=tf.float32)], 1) # (batch_size, num_values + 1, value_vec_size)
            logging.debug('Bicoattn n_values shape: %s', n_values.shape)
            n_values_mask = tf.concat([values_mask, tf.ones([tf.shape(values_mask)[0], 1], dtype=tf.int32)], 1) # (batch_size, num_values + 1)
            logging.debug('Bicoattn n_values_mask shape: %s', n_values_mask.shape)

            # the num_keys in the comment is the origin keys' shape, which is different from the "num_keys" below; the same as num_values
            num_keys = n_keys.shape[1]
            num_values = n_values.shape[1]

            with vs.variable_scope('BiCoattn_encoder'):
                n_c_values = tf.contrib.layers.fully_connected(n_values, self.value_vec_size, activation_fn=tf.tanh) # (batch_size, num_values + 1, value_vec_size)
                logging.debug('Bicoaatn n_c_values shape: %s', n_c_values.shape)

            # Coattention Matrix Calc
            t_keys = tf.tile( tf.expand_dims(n_keys, 2), [1, 1, num_values, 1] ) # (batch_size, num_keys + 1, num_values + 1, key_vec_size)
            logging.debug('Bicoattn t_keys shape: %s', t_keys.shape)
            t_values = tf.tile( tf.expand_dims(n_c_values, 1), [1, num_keys, 1, 1] ) # (batch_size, num_keys + 1, num_values + 1, value_vec_size)
            logging.debug('Bicoattn t_values shape: %s', t_values.shape)
            e_keys = tf.expand_dims(n_keys, 2) # (batch_size, num_keys + 1, 1, key_vec_size)
            logging.debug('Bicoattn e_keys shape: %s', e_keys.shape)
            e_values = tf.expand_dims(n_c_values, 1) # (batch_size, 1, num_values + 1, value_vec_size)
            logging.debug('Bicoattn e_values shape: %s', e_values.shape)
            tmp = tf.multiply(e_keys, e_values) # (batch_size, num_keys + 1, num_values + 1, key_vec_size)
            logging.debug('Bicoattn tmp shape: %s', tmp.shape)

            with vs.variable_scope('BiCoattn_matrix_1'):
                attn_logits_1 = tf.contrib.layers.fully_connected( tf.reshape(tmp, [-1, num_keys * num_values, self.key_vec_size]), 1, activation_fn=None )
            with vs.variable_scope('BiCoattn_matrix_2'):
                attn_logits_2 = tf.contrib.layers.fully_connected( tf.reshape(t_keys, [-1, num_keys * num_values, self.key_vec_size]), 1, activation_fn=None )
            with vs.variable_scope('BiCoattn_matrix_3'):
                attn_logits_3 = tf.contrib.layers.fully_connected( tf.reshape(t_values, [-1, num_keys * num_values, self.key_vec_size]), 1, activation_fn=None )
            attn_logits = attn_logits_1 + attn_logits_2 + attn_logits_3 # (batch_size, num_keys + 1 * num_values + 1, 1)
            logging.debug('Bicoattn attn_logits shape: %s', attn_logits.shape)
            attn_logits = tf.reshape(attn_logits, [-1, num_keys, num_values]) # (batch_size, num_keys + 1, num_values + 1)
            logging.debug('Bicoattn reshaped attn_logits shape; %s', attn_logits.shape)

            # Context2Query
            e_values_mask = tf.expand_dims(n_values_mask, 1) # (batch_size, 1, num_values + 1)
            logging.debug('Bicoattn e_values_mask shape: %s', e_values_mask.shape)
            _, c2q_attn_dist = masked_softmax(attn_logits, e_values_mask, 2) # (batch_size, num_keys + 1, num_values + 1)
            logging.debug('PointerBicoattn c2q_attn_dist shape: %s', c2q_attn_dist.shape)
            CQ = tf.matmul( tf.transpose(n_keys,perm=[0, 2, 1]), c2q_attn_dist ) # (batch_size, key_vec_size, num_values + 1) 
            logging.debug('Bicoattn CQ shape: %s', CQ.shape)
            # # Context2Query from BiDAF
            # c2q_bidaf_out = tf.matmul(c2q_attn_dist, n_c_values) # (batch_size, num_keys + 1, value_vec_size)
            # logging.debug('Bicoattn c2q_bidaf_out shape: %s', c2q_bidaf_out.shape)

            # Query2Context
            e_keys_mask = tf.expand_dims(n_keys_mask, 2) # (batch_size, num_keys + 1, 1)
            logging.debug('Bicoattn e_keys_mask shape: %s', e_keys_mask.shape)
            _, q2c_attn_dist = masked_softmax(attn_logits, e_keys_mask, 1) # (batch_size, num_keys + 1, num_values + 1)
            logging.debug('Bicoattn q2c_attn_dist shape: %s', q2c_attn_dist.shape)
            # Query2Context from BiDAF
            agg_keys = tf.reduce_max(attn_logits, axis=2, keep_dims=True) # (batch_size, num_keys + 1, 1)
            logging.debug('Bicoattn agg_keys shape: %s', agg_keys.shape)
            _, agg_keys_dist = masked_softmax(agg_keys, e_keys_mask, 1) # (batch_size, num_keys + 1, 1)
            logging.debug('Bicoattn agg_keys_dist shape: %s', agg_keys_dist.shape)
            q2c_bidaf_out = tf.matmul( tf.tile(agg_keys_dist, [1, 1, num_keys]), n_keys ) # (batch_size, num_keys+1, num_keys+1) dot (batch_size, num_keys+1, key_vec_size) -> (batch_size, num_keys+1, key_vec_size)
            logging.debug('Bicoattn q2c_bidaf_out shape: %s', q2c_bidaf_out.shape)

            CD = tf.matmul( tf.concat([tf.transpose(n_c_values, perm=[0, 2, 1]), CQ], 1), tf.transpose(q2c_attn_dist, perm=[0, 2, 1]) ) # (batch_size, 2 * value_vec_size, num_keys + 1)
            logging.debug('Bicoattn CD shape: %s', CD.shape)
            CD = tf.transpose(CD, perm=[0, 2, 1]) # (batch_size, num_keys + 1, 2 * value_vec_size)
            logging.debug('Bicoattn CD.T shape: %s', CD.shape)

            g = tf.concat([n_keys, CD, tf.multiply(tf.concat([n_keys, n_keys], 2), CD), tf.multiply(n_keys, q2c_bidaf_out)], 2)[:, :-1, :] # (batch_size, num_keys, key_vec_size * 6)
            logging.debug('Bicoattn g shape: %s', g.shape)

            # Modeling Layer(2 layers)
            with vs.variable_scope('BiCoattn_modeling_1'):
                (fw_out_1, bw_out_1), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw_1, self.rnn_cell_bw_1, g, dtype=tf.float32) # (batch_size, num_keys, key_vec_size / 2)
                out_1 = tf.concat([fw_out_1, bw_out_1], 2)# (batch_size, num_keys, key_vec_size)
                logging.debug('Bicoattn out_1 shape: %s', out_1.shape)
            with vs.variable_scope('BiCoattn_modeling_2'):
                (fw_out_2, bw_out_2), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw_2, self.rnn_cell_bw_2, out_1, dtype=tf.float32) # (batch_size, num_keys, key_vec_size / 2)
                out_2 = tf.concat([fw_out_2, bw_out_2], 2) # (batch_size, num_keys, key_vec_size)
                logging.debug('Bicoattn out_2 shape: %s', out_2.shape)

            # Output Layer
            g_m = tf.concat([g, out_2], 2) # (batch_size, num_keys, key_vec_size * 7)
            logging.debug('Bicoattn g_m shape: %s', g_m.shape)

            with vs.variable_scope('BiCoattn_output_layer2'):
                (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, g_m, dtype=tf.float32) # (batch_size, num_keys, key_vec_size / 2)
                out = tf.concat([fw_out, bw_out], 2) # batch_size, num_keys, key_vec_size
                logging.debug('Bicoattn out shape: %s', out.shape)
            g_m2 = tf.concat([g, out], 2) # (batch_size, num_keys, key_vec_size * 7)
            logging.debug('Bicoattn g_m2 shape: %s', g_m2.shape)

        return g_m, g_m2, attn_logits



# @zhengyang
def normalize(inputs, epsilon = 1e-8, scope="ln", reuse=None):

    with vs.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        # beta  = tf.get_variable('beta', shape=params_shape, dtype=tf.float32, initializer=tf.constant_initializer(0))
        # gamma  = tf.get_variable('gamma', shape=params_shape, dtype=tf.float32, initializer=tf.constant_initializer(1))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

# @zhengyang
def multihead_attention(queries, query_masks, keys, key_masks, num_units=None, num_heads=8, dropout_rate=0, is_training='yes', scope="multihead_attention", reuse=None):

    with vs.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = (1 - tf.cast(key_masks, 'float')) * (-1e30)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.expand_dims(key_masks, 1) # (h*N, 1, T_k)
        # logging.debug('multihead func outputs shape: %s', outputs.shape)
        # logging.debug('multihead func key_masks shape: %s', key_masks.shape)
        outputs = tf.add(outputs, key_masks)
        outputs = tf.nn.softmax(outputs, 2) # (h*N, T_q, T_k)
        # key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        # paddings = tf.ones_like(outputs)*(-2**32+1)
        # outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
        # Activation
        # outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        # query_masks = tf.cast(query_masks, 'float')
        # query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        # query_masks = tf.expand_dims(query_masks, 2) # (h*N, T_q, 1)
        # outputs = tf.multiply(outputs, query_masks)
          
        # Dropouts
        if is_training == 'yes':
            identify_training = True
        else:
            identify_training = False
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(identify_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection
        outputs = tf.add(outputs, queries) # (N, T_q, C)
              
        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)
 
    return outputs

# @zhengyang
def feedforward(inputs, num_units=[2048, 512], scope="multihead_attention", reuse=None):

    with vs.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params) 
        
        # Readout layer 
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True} 
        outputs = tf.layers.conv1d(**params) 
        
        # Residual connection 
        outputs += inputs 
        
        # Normalize 
        outputs = normalize(outputs) 
    
    return outputs


# @zhengyang
class TransformerNetwork(object):
    def __init__(self, hidden_size, keep_prob, key_vec_size, value_vec_size, num_blocks, num_heads):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        self.rnn_cell_fw_1 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw_1 = DropoutWrapper(self.rnn_cell_fw_1, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw_1 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw_1 = DropoutWrapper(self.rnn_cell_bw_1, input_keep_prob=self.keep_prob)

        self.rnn_cell_fw_2 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw_2 = DropoutWrapper(self.rnn_cell_fw_2, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw_2 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw_2 = DropoutWrapper(self.rnn_cell_bw_2, input_keep_prob=self.keep_prob)

        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, values, values_mask, keys, keys_mask, is_training):
        
        # Encoder
        enc = values # (batch_size, num_values, value_vec_size)
        enc_masks = values_mask # (batch_size, num_values)
        for i in range(self.num_blocks):
            with vs.variable_scope('enc_num_blocks_{}'.format(i)):
                # Multihead Attetion
                enc = multihead_attention(
                        queries=enc, 
                        query_masks=enc_masks, 
                        keys=enc, 
                        key_masks=enc_masks, 
                        num_units=self.value_vec_size, 
                        num_heads=self.num_heads, 
                        dropout_rate=1-self.keep_prob, 
                        is_training=is_training,
                        scope='enc_self_attention'
                        ) #(batch_size, num_values,value_vec_size)
                # Feed Forword
                logging.debug('Encoder Block %s multiattn enc shape: %s', i, enc.shape)
                enc = feedforward(enc, num_units=[4*self.value_vec_size, self.value_vec_size], scope='enc_feedward') # (batch_szie, num_value, value_vec_size)
                logging.debug('Encoder Block %s feedforward enc shape: %s', i, enc.shape)

        # Decoder
        dec = keys # (batch_size, num_keys, key_vec_size)
        dec_masks = keys_mask # (batch_size, num_keys)
        for i in range(self.num_blocks):
            with vs.variable_scope('dec_num_blocks_{}'.format(i)):
                # Multihead Attn(self attetion)
                dec = multihead_attention(
                        queries=dec,
                        query_masks=dec_masks,
                        keys=dec, 
                        key_masks=dec_masks,
                        num_units=self.key_vec_size,
                        num_heads=self.num_heads,
                        dropout_rate=1-self.keep_prob,
                        is_training=is_training,
                        scope='dec_self_attention'
                        ) # (batch_size, num_keys, key_vec_size)
                logging.debug('Decoder Block %s multiattn dec1 shape: %s', i, dec.shape)
                # Multihead attn (vanilla attention) 
                dec = multihead_attention(
                        queries=dec,
                        query_masks=dec_masks,
                        keys=enc,
                        key_masks=enc_masks,
                        num_units=self.key_vec_size,
                        num_heads=self.num_heads,
                        dropout_rate=1-self.keep_prob,
                        is_training=is_training,
                        scope='dec_vanilla_attetion'
                        ) # (batch_size, num_keys, key_vec_size)
                logging.debug('Decoder Block %s multiattn dec2 shape: %s', i, dec.shape)
                # Feed Forward
                dec = feedforward(dec, num_units=[4*self.key_vec_size, self.key_vec_size], scope='dec_feedward') # (batch_size, num_keys, key_vec_size)
                logging.debug('Decoder Block %s feedforward dec3 shape: %s', i, dec.shape)


        # Modeling Layer(2 layers)
        with vs.variable_scope('Transformer_modeling_1'):
            (fw_out_1, bw_out_1), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw_1, self.rnn_cell_bw_1, dec, dtype=tf.float32) # (batch_size, num_keys, key_vec_size / 2)
            out_1 = tf.concat([fw_out_1, bw_out_1], 2)# (batch_size, num_keys, key_vec_size)
            logging.debug('out_1 shape: %s', out_1.shape)
        with vs.variable_scope('Transformer_modeling_2'):
            (fw_out_2, bw_out_2), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw_2, self.rnn_cell_bw_2, out_1, dtype=tf.float32) # (batch_size, num_keys, key_vec_size / 2)
            out_2 = tf.concat([fw_out_2, bw_out_2], 2) # (batch_size, num_keys, key_vec_size)
            logging.debug('out_2 shape: %s', out_2.shape)

        # Output Layer
        g_m = tf.concat([dec, out_2], 2) # (batch_size, num_keys, key_vec_size * 2)
        logging.debug('g_m shape: %s', g_m.shape)

        with vs.variable_scope('Transformer_output_layer2'):
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, g_m, dtype=tf.float32) # (batch_size, num_keys, key_vec_size / 2)
            out = tf.concat([fw_out, bw_out], 2) # batch_size, num_keys, key_vec_size
            logging.debug('out shape: %s', out.shape)
        g_m2 = tf.concat([dec, out], 2) # (batch_size, num_keys, key_vec_size * 2)
        logging.debug('g_m2 shape: %s', g_m2.shape)

        return g_m, g_m2

# @zhengyang
def multihead_bicoattention(queries, query_masks, keys, key_masks, num_units=None, num_heads=8, dropout_rate=0, is_training='yes', is_selfattention=True, scope="multihead_attention", reuse=None):

    with vs.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = (1 - tf.cast(key_masks, 'float')) * (-1e30)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.expand_dims(key_masks, 1) # (h*N, 1, T_k)
        # logging.debug('multihead func outputs shape: %s', outputs.shape)
        # logging.debug('multihead func key_masks shape: %s', key_masks.shape)
        outputs_1 = tf.add(outputs, key_masks)
        outputs_1 = tf.nn.softmax(outputs_1, 2) # (h*N, T_q, T_k)
        
        # Query Masking
        query_masks = (1 - tf.cast(query_masks, 'float')) * (-1e30)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.expand_dims(query_masks, 2) # (h*N, T_q, 1)
        outputs_2 = tf.add(outputs, query_masks) # (h*N, T_q, T_k)
        outputs_2 = tf.nn.softmax(outputs_2, 1) # (h*N, T_q, T_k)
          
        # Dropouts
        if is_training == 'yes':
            identify_training = True
        else:
            identify_training = False
        outputs_1 = tf.layers.dropout(outputs_1, rate=dropout_rate, training=tf.convert_to_tensor(identify_training))
        outputs_2 = tf.layers.dropout(outputs_2, rate=dropout_rate, training=tf.convert_to_tensor(identify_training))
               
        outputs_1_1 = tf.matmul(outputs_1, V_) # ( h*N, T_q, C/h)

        outputs_2_2 = tf.matmul( tf.transpose(outputs_2, perm=[0, 2, 1]), Q_ ) # (h*N, T_k, T_q) dot (h*N, T_q, C/h) => (h*N, T_k, C/h)
        outputs_2_2 = tf.matmul( outputs_1, outputs_2_2 )  # (h*N, T_q, C/h)
        
        outputs_bidaf = tf.reduce_max(outputs_1, 2, keep_dims=True) # (h*N, T_q, 1)
        outputs_bidaf = tf.matmul( tf.transpose( outputs_bidaf, perm=[0, 2, 1] ), Q_ ) # (h*N, 1, C/h)
        outputs_bidaf = tf.multiply( outputs_bidaf, Q_ ) # (h*N, T_q, C/h)

        outputs_3 = tf.concat( [outputs_1_1, outputs_2_2, outputs_bidaf], 2 ) # (h*N, T_q, 3 * C/h)

        # Restore shape
        outputs_4 = tf.concat(tf.split(outputs_3, num_heads, axis=0), axis=2 ) # (N, T_q, 3*C)
        if is_selfattention:
            outputs_4 = tf.layers.dense(outputs_4, num_units, activation=tf.nn.relu) # (N, T_k, C)
              
            # Residual connection
            outputs_4 = tf.add(outputs_4, queries) # (N, T_q, C)
              
        # Normalize
        outputs_4 = normalize(outputs_4) # (N, T_q, ?)
 
    return outputs_4

# @zhengyang
class BCTN(object):
    def __init__(self, hidden_size, keep_prob, key_vec_size, value_vec_size, num_blocks, num_heads):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        self.rnn_cell_fw_1 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw_1 = DropoutWrapper(self.rnn_cell_fw_1, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw_1 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw_1 = DropoutWrapper(self.rnn_cell_bw_1, input_keep_prob=self.keep_prob)

        self.rnn_cell_fw_2 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw_2 = DropoutWrapper(self.rnn_cell_fw_2, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw_2 = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw_2 = DropoutWrapper(self.rnn_cell_bw_2, input_keep_prob=self.keep_prob)

        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, values, values_mask, keys, keys_mask, is_training):
        
        enc = values # (batch_size, num_values, value_vec_size)
        enc_masks = values_mask # (batch_size, num_values)

        dec = keys # (batch_size, num_keys, key_vec_size)
        dec_masks = keys_mask # (batch_size, num_keys)
        for i in range(self.num_blocks):
            with vs.variable_scope('enc_num_blocks_1_{}'.format(i)):
                # Multihead Attetion
                enc = multihead_attention(
                        queries=enc, 
                        query_masks=enc_masks, 
                        keys=enc, 
                        key_masks=enc_masks, 
                        num_units=self.value_vec_size, 
                        num_heads=self.num_heads, 
                        dropout_rate=1-self.keep_prob, 
                        is_training=is_training,
                        scope='enc_self_attention'
                        ) #(batch_size, num_values,value_vec_size)
                logging.debug('Encoder Block %s multiattn enc1 shape: %s', i, enc.shape)

            with vs.variable_scope('dec_num_blocks_1_{}'.format(i)):
                # Multihead Attn(self attetion)
                dec = multihead_attention(
                        queries=dec,
                        query_masks=dec_masks,
                        keys=dec, 
                        key_masks=dec_masks,
                        num_units=self.key_vec_size,
                        num_heads=self.num_heads,
                        dropout_rate=1-self.keep_prob,
                        is_training=is_training,
                        scope='dec_self_attention'
                        ) # (batch_size, num_keys, key_vec_size)
                logging.debug('Decoder Block %s multiattn dec1 shape: %s', i, dec.shape)

            with vs.variable_scope('coattn_enc_num_blocks_{}'.format(i)):
                # Multihead Attetion
                enc2 = multihead_bicoattention(
                        queries=enc, 
                        query_masks=enc_masks, 
                        keys=dec, 
                        key_masks=dec_masks, 
                        num_units=self.value_vec_size, 
                        num_heads=self.num_heads, 
                        dropout_rate=1-self.keep_prob, 
                        is_training=is_training,
                        is_selfattention=True,
                        scope='enc_self_attention'
                        ) #(batch_size, num_values, value_vec_size)
                logging.debug('Encoder Block %s multi_bicoattn enc2 shape: %s', i, enc2.shape)

            with vs.variable_scope('coattn_dec_num_blocks_{}'.format(i)):
                # Multihead Attetion
                dec2 = multihead_bicoattention(
                        queries=dec, 
                        query_masks=dec_masks, 
                        keys=enc, 
                        key_masks=enc_masks, 
                        num_units=self.value_vec_size, 
                        num_heads=self.num_heads, 
                        dropout_rate=1-self.keep_prob, 
                        is_training=is_training,
                        is_selfattention=True,
                        scope='dec_self_attention'
                        ) #(batch_size, num_values, value_vec_size)
                logging.debug('Decoder Block %s multi_bicoattn dec2 shape: %s', i, dec.shape)

            with vs.variable_scope('forward_enc_num_blocks_{}'.format(i)):
                enc = feedforward(enc2, num_units=[4*self.key_vec_size, self.key_vec_size], scope='enc_feedward') # (batch_size, num_keys, key_vec_size)
                logging.debug('Encoder Block %s feedforward enc shape: %s', i, enc.shape)

            with vs.variable_scope('forward_dec_num_blocks_{}'.format(i)):
                dec = feedforward(dec2, num_units=[4*self.key_vec_size, self.key_vec_size], scope='dec_feedward') # (batch_size, num_keys, key_vec_size)
                logging.debug('Encoder Block %s feedforward dec shape: %s', i, dec.shape)

        with vs.variable_scope('final_dec_num_blocks'):
            dec = multihead_bicoattention(
                    queries=dec, 
                    query_masks=dec_masks, 
                    keys=enc, 
                    key_masks=enc_masks, 
                    num_units=self.value_vec_size, 
                    num_heads=self.num_heads, 
                    dropout_rate=1-self.keep_prob, 
                    is_training=is_training,
                    is_selfattention=False,
                    scope='dec_self_attention'
                    ) #(batch_size, num_values,value_vec_size*3)
            logging.debug('Final dec shape: %s', dec.shape)

        # with vs.variable_scope('forward_dec_num_blocks'):
            # dec = feedforward(dec, num_units=[4*self.key_vec_size, self.key_vec_size], scope='dec_feedward') # (batch_size, num_keys, key_vec_size)
            # logging.debug('Final feedforward dec shape: %s', dec.shape)

        # Modeling Layer(2 layers)
        with vs.variable_scope('Transformer_modeling_1'):
            (fw_out_1, bw_out_1), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw_1, self.rnn_cell_bw_1, dec, dtype=tf.float32) # (batch_size, num_keys, key_vec_size / 2)
            out_1 = tf.concat([fw_out_1, bw_out_1], 2)# (batch_size, num_keys, key_vec_size)
            logging.debug('out_1 shape: %s', out_1.shape)
        with vs.variable_scope('Transformer_modeling_2'):
            (fw_out_2, bw_out_2), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw_2, self.rnn_cell_bw_2, out_1, dtype=tf.float32) # (batch_size, num_keys, key_vec_size / 2)
            out_2 = tf.concat([fw_out_2, bw_out_2], 2) # (batch_size, num_keys, key_vec_size)
            logging.debug('out_2 shape: %s', out_2.shape)

        # Output Layer
        g_m = tf.concat([dec, out_2], 2) # (batch_size, num_keys, key_vec_size * 4)
        logging.debug('g_m shape: %s', g_m.shape)

        with vs.variable_scope('Transformer_output_layer2'):
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, g_m, dtype=tf.float32) # (batch_size, num_keys, key_vec_size / 2)
            out = tf.concat([fw_out, bw_out], 2) # batch_size, num_keys, key_vec_size
            logging.debug('out shape: %s', out.shape)
        g_m2 = tf.concat([dec, out], 2) # (batch_size, num_keys, key_vec_size * 4)
        logging.debug('g_m2 shape: %s', g_m2.shape)

        return g_m, g_m2

