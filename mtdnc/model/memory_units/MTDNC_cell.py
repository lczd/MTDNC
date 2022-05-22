# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
import numpy as np

from mtdnc.model.utils import oneplus, layer_norm, unit_simplex_initialization

"""
The memory layer.
"""


class MTDNCMemoryCell:
    
    def __init__(self, input_size, memory_length, memory_width, read_heads, bypass_dropout=False, dnc_norm=False,
                 seed=100, reuse=False, analyse=False, dtype=tf.float32, name='base'):

        self.rng = np.random.RandomState(seed=seed)
        self.seed = seed
        self.dtype = dtype
        self.analyse = analyse

        # dnc parameters
        self.input_size = input_size
        self.h_N = memory_length
        self.h_W = memory_width
        self.h_RH = read_heads
        self.h_B = 0  # batch size, will be set in call

        self.dnc_norm = dnc_norm
        self.bypass_dropout = bypass_dropout

        self.reuse = reuse
        self.name = name

        self.const_memory_ones = None # will be defined with use of batch size in call method
        self.const_batch_memory_range = None # will be defined with use of batch size in call method
        self.const_link_matrix_inv_eye = None # will be defined with use of batch size in call method
    
    @property
    def output_size(self):
        return 2 * self.h_RH * self.h_W + self.input_size

    @property
    def state_size(self):
        init_memory = tf.TensorShape([self.h_N, self.h_W])
        init_memory_sec = tf.TensorShape([self.h_N, self.h_W])
        init_usage_vector = tf.TensorShape([self.h_N])
        init_usage_vector_sec = tf.TensorShape([self.h_N])
        init_write_weighting = tf.TensorShape([self.h_N])
        init_write_weighting_sec = tf.TensorShape([self.h_N])
        init_read_weighting = tf.TensorShape([self.h_RH, self.h_N])
        init_read_weighting_sec = tf.TensorShape([self.h_RH, self.h_N])
        return (init_memory, init_memory_sec, init_usage_vector, init_usage_vector_sec, init_write_weighting,
                init_write_weighting_sec, init_read_weighting, init_read_weighting_sec)

    def zero_state(self, batch_size, dtype=tf.float32):

        init_memory = tf.fill([batch_size, self.h_N, self.h_W], tf.cast(1 / (self.h_N * self.h_W), dtype=dtype))
        init_memory_sec = tf.fill([batch_size, self.h_N, self.h_W], tf.cast(1 / (self.h_N * self.h_W), dtype=dtype))
        init_usage_vector = tf.zeros([batch_size, self.h_N], dtype=dtype)
        init_usage_vector_sec = tf.zeros([batch_size, self.h_N], dtype=dtype)
        init_write_weighting = unit_simplex_initialization(self.rng, batch_size, [self.h_N], dtype=dtype)
        init_write_weighting_sec = unit_simplex_initialization(self.rng, batch_size, [self.h_N], dtype=dtype)
        init_read_weighting = unit_simplex_initialization(self.rng, batch_size, [self.h_RH, self.h_N], dtype=dtype)
        init_read_weighting_sec = unit_simplex_initialization(self.rng, batch_size, [self.h_RH, self.h_N], dtype=dtype)
        zero_states = (init_memory, init_memory_sec, init_usage_vector, init_usage_vector_sec,
                       init_write_weighting, init_write_weighting_sec, init_read_weighting, init_read_weighting_sec)

        return zero_states

    def _weight_input(self, inputs):

        input_size = inputs.get_shape()[1].value
        total_signal_size = (2 * self.h_RH + 6) * self.h_W + 6 + 4 * self.h_RH

        with tf.variable_scope('{}'.format(self.name), reuse=self.reuse):
            w_x = tf.get_variable("mu_w_x", (input_size, total_signal_size),
                                  initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            b_x = tf.get_variable("mu_b_x", (total_signal_size,), initializer=tf.constant_initializer(0.),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            weighted_input = tf.matmul(inputs, w_x) + b_x

            if self.dnc_norm:
                weighted_input = layer_norm(weighted_input, name='layer_norm', dtype=self.dtype)

        return weighted_input

    def __call__(self, inputs, pre_states, scope=None):

        self.h_B = inputs.get_shape()[0].value

        memory_ones, batch_memory_range = self._create_constant_value_tensors(self.h_B, self.dtype)
        self.const_memory_ones = memory_ones
        self.const_batch_memory_range = batch_memory_range

        pre_memory, pre_memory_sec, pre_usage_vector, pre_usage_vector_sec, pre_write_weightings, \
            pre_write_weightings_sec, pre_read_weightings, pre_read_weightings_sec = pre_states

        weighted_input = self._weight_input(inputs)

        control_signals = self.my_create_control_signals(weighted_input)
        alloc_gate, alloc_gate_sec, free_gates, free_gates_sec, write_gate, write_gate_sec, write_keys, \
            write_keys_sec, write_strengths, write_strengths_sec, write_vector, write_vector_sec, erase_vector, \
            erase_vector_sec, read_keys, read_keys_sec, read_strengths, read_strengths_sec = control_signals

        # 工作记忆区处理
        alloc_weightings, usage_vector = self.my_update_alloc_and_usage_vectors(
            pre_write_weightings, pre_read_weightings, pre_usage_vector, free_gates
        )
        write_content_weighting = self.my_calculate_content_weightings(pre_memory, write_keys, write_strengths)
        write_weighting = self.my_update_write_weighting(alloc_weightings, write_content_weighting, write_gate,
                                                         alloc_gate)
        memory = self.my_update_memory(pre_memory, write_weighting, write_vector, erase_vector)
        read_content_weightings = self.my_calculate_content_weightings(memory, read_keys, read_strengths)
        # read_vectors = [batch_size, read_heads, memory_width]
        read_vectors = self.my_read_memory(memory, read_content_weightings)

        # 将read_vectors中的read_heads维度压缩成1
        read_vectors_for_sec = tf.reduce_prod(read_vectors, axis=1, keepdims=True, name='read_vectors_prod')
        
        # 长期记忆区处理
        alloc_weightings_sec, usage_vector_sec = self.my_update_alloc_and_usage_vectors(
            pre_write_weightings_sec, pre_read_weightings_sec, pre_usage_vector_sec, free_gates_sec
        )
        write_content_weighting_sec = self.my_calculate_content_weightings(pre_memory_sec, write_keys_sec,
                                                                           write_strengths_sec)
        write_weighting_sec = self.my_update_write_weighting(alloc_weightings_sec, write_content_weighting_sec,
                                                             write_gate_sec, alloc_gate_sec)
        memory_sec = self.my_update_memory(pre_memory_sec, write_weighting_sec, read_vectors_for_sec, erase_vector_sec)
        read_content_weightings_sec = self.my_calculate_content_weightings(memory_sec, read_keys_sec,
                                                                           read_strengths_sec)
        read_vectors_sec = self.my_read_memory(memory_sec, read_content_weightings_sec)

        # 合并双区读取的内容
        # back_read_vectors = [batch_size, 2 * read_heads, memory_width]
        back_read_vectors = tf.concat([read_vectors, read_vectors_sec], axis=2)

        # back_read_vectors = [batch_size, 2 * read_heads * memory_width]
        back_read_vectors = tf.reshape(back_read_vectors, [self.h_B, 2 * self.h_W * self.h_RH])

        if self.bypass_dropout:
            input_bypass = tf.nn.dropout(inputs, self.bypass_dropout)
        else:
            input_bypass = inputs

        output = tf.concat([back_read_vectors, input_bypass], axis=-1)

        if self.analyse:
            output = (output, control_signals)

        return output, (memory, memory_sec, usage_vector, usage_vector_sec, write_weighting, write_weighting_sec,
                        read_content_weightings, read_content_weightings_sec)

    def _create_constant_value_tensors(self, batch_size, dtype):

        memory_ones = tf.ones([batch_size, self.h_N, self.h_W], dtype=dtype, name="memory_ones")

        batch_range = tf.range(0, batch_size, delta=1, dtype=tf.int32, name="batch_range")
        repeat_memory_length = tf.fill([self.h_N], tf.constant(self.h_N, dtype=tf.int32), name="repeat_memory_length")
        batch_memory_range = tf.matmul(tf.expand_dims(batch_range, -1), tf.expand_dims(repeat_memory_length, 0),
                                       name="batch_memory_range")
        return memory_ones, batch_memory_range

    def my_create_control_signals(self, weighted_input):

        write_keys = weighted_input[:, :         self.h_W]  # W
        write_keys_sec = weighted_input[:, self.h_W:         2 * self.h_W]  # W
        write_strengths = weighted_input[:, 2 * self.h_W:         2 * self.h_W + 1]  # 1
        write_strengths_sec = weighted_input[:, 2 * self.h_W + 1:         2 * self.h_W + 2]  # 1
        erase_vector = weighted_input[:, 2 * self.h_W + 2:       3 * self.h_W + 2]  # W
        erase_vector_sec = weighted_input[:, 3 * self.h_W + 2:       4 * self.h_W + 2]  # W
        write_vector = weighted_input[:, 4 * self.h_W + 2:       5 * self.h_W + 2]  # W
        write_vector_sec = weighted_input[:, 5 * self.h_W + 2:       6 * self.h_W + 2]  # W
        alloc_gates = weighted_input[:, 6 * self.h_W + 2:       6 * self.h_W + 3]  # 1
        alloc_gates_sec = weighted_input[:, 6 * self.h_W + 3:       6 * self.h_W + 4]  # 1
        write_gates = weighted_input[:, 6 * self.h_W + 4:       6 * self.h_W + 5]  # 1
        write_gates_sec = weighted_input[:, 6 * self.h_W + 5:       6 * self.h_W + 6]  # 1
        read_keys = weighted_input[:, 6 * self.h_W + 6: (self.h_RH + 6) * self.h_W + 6]  # R * W
        read_keys_sec = weighted_input[:, (self.h_RH + 6) * self.h_W + 6: (2 * self.h_RH + 6) * self.h_W + 6]  # R * W
        read_strengths = weighted_input[:, (2 * self.h_RH + 6) * self.h_W + 6: (2 * self.h_RH + 6) * self.h_W + 6 + 1 * self.h_RH]  # R
        read_strengths_sec = weighted_input[:, (2 * self.h_RH + 6) * self.h_W + 6 + 1 * self.h_RH: (2 * self.h_RH + 6) * self.h_W + 6 + 2 * self.h_RH]  # R
        free_gates = weighted_input[:, (2 * self.h_RH + 6) * self.h_W + 6 + 2 * self.h_RH: (2 * self.h_RH + 6) * self.h_W + 6 + 3 * self.h_RH]
        free_gates_sec = weighted_input[:, (2 * self.h_RH + 6) * self.h_W + 6 + 3 * self.h_RH: (2 * self.h_RH + 6) * self.h_W + 6 + 4 * self.h_RH]

        alloc_gates = tf.sigmoid(alloc_gates, 'alloc_gates')
        free_gates = tf.sigmoid(free_gates, 'free_gates')
        free_gates = tf.expand_dims(free_gates, 2)
        write_gates = tf.sigmoid(write_gates, 'write_gates')

        alloc_gates_sec = tf.sigmoid(alloc_gates_sec, 'alloc_gates_sec')
        free_gates_sec = tf.sigmoid(free_gates_sec, 'free_gates_sec')
        free_gates_sec = tf.expand_dims(free_gates_sec, 2)
        write_gates_sec = tf.sigmoid(write_gates_sec, 'write_gates_sec')

        write_keys = tf.expand_dims(write_keys, axis=1)
        write_strengths = oneplus(write_strengths)
        write_vector = tf.reshape(write_vector, [self.h_B, 1, self.h_W])
        erase_vector = tf.sigmoid(erase_vector, 'erase_vector')
        erase_vector = tf.reshape(erase_vector, [self.h_B, 1, self.h_W])

        write_keys_sec = tf.expand_dims(write_keys_sec, axis=1)
        write_strengths_sec = oneplus(write_strengths_sec)
        write_vector_sec = tf.reshape(write_vector_sec, [self.h_B, 1, self.h_W])
        erase_vector_sec = tf.sigmoid(erase_vector_sec, 'erase_vector_sec')
        erase_vector_sec = tf.reshape(erase_vector_sec, [self.h_B, 1, self.h_W])

        read_keys = tf.reshape(read_keys, [self.h_B, self.h_RH, self.h_W])
        read_strengths = oneplus(read_strengths)
        read_strengths = tf.expand_dims(read_strengths, axis=2)

        read_keys_sec = tf.reshape(read_keys_sec, [self.h_B, self.h_RH, self.h_W])
        read_strengths_sec = oneplus(read_strengths_sec)
        read_strengths_sec = tf.expand_dims(read_strengths_sec, axis=2)

        return alloc_gates, alloc_gates_sec, free_gates, free_gates_sec, write_gates, write_gates_sec, write_keys, \
            write_keys_sec, write_strengths, write_strengths_sec, write_vector, write_vector_sec, erase_vector, \
            erase_vector_sec, read_keys, read_keys_sec, read_strengths, read_strengths_sec

    def my_update_alloc_and_usage_vectors(self, pre_write_weightings, pre_read_weightings, pre_usage_vector, free_gates):

        retention_vector = tf.reduce_prod(1 - free_gates * pre_read_weightings, axis=1, keepdims=False,
                                          name='retention_prod')
        usage_vector = (
                           pre_usage_vector + pre_write_weightings - pre_usage_vector * pre_write_weightings) * retention_vector

        sorted_usage, free_list = tf.nn.top_k(-1 * usage_vector, self.h_N)
        sorted_usage = -1 * sorted_usage

        cumprod_sorted_usage = tf.cumprod(sorted_usage, axis=1, exclusive=True)
        corrected_free_list = free_list + self.const_batch_memory_range

        cumprod_sorted_usage_re = [tf.reshape(cumprod_sorted_usage, [-1, ]), ]
        corrected_free_list_re = [tf.reshape(corrected_free_list, [-1]), ]

        stitched_usage = tf.dynamic_stitch(corrected_free_list_re, cumprod_sorted_usage_re, name=None)

        stitched_usage = tf.reshape(stitched_usage, [self.h_B, self.h_N])

        alloc_weighting = (1 - usage_vector) * stitched_usage

        return alloc_weighting, usage_vector

    @staticmethod
    def my_update_write_weighting(alloc_weighting, write_content_weighting, write_gate, alloc_gate):

        write_weighting = write_gate * (alloc_gate * alloc_weighting + (1 - alloc_gate) * write_content_weighting)

        return write_weighting

    def my_update_memory(self, pre_memory, write_weighting, write_vector, erase_vector):

        write_w = tf.expand_dims(write_weighting, 2)
        erase_matrix = tf.multiply(pre_memory, (self.const_memory_ones - tf.matmul(write_w, erase_vector)))
        write_matrix = tf.matmul(write_w, write_vector)
        return erase_matrix + write_matrix

    @staticmethod
    def my_calculate_content_weightings(memory, keys, strengths):
    
        similarity_numerator = tf.matmul(keys, memory, adjoint_b=True)
    
        norm_memory = tf.sqrt(tf.reduce_sum(tf.square(memory), axis=2, keepdims=True))
        norm_keys = tf.sqrt(tf.reduce_sum(tf.square(keys), axis=2, keepdims=True))
        similarity_denominator = tf.matmul(norm_keys, norm_memory, adjoint_b=True)
    
        similarity = similarity_numerator / similarity_denominator
        similarity = tf.squeeze(similarity)
        adjusted_similarity = similarity * strengths
    
        softmax_similarity = tf.nn.softmax(adjusted_similarity, axis=-1)
    
        return softmax_similarity

    @staticmethod
    def my_read_memory(memory, read_weightings):
        read_vectors = tf.matmul(read_weightings, memory)
        return read_vectors