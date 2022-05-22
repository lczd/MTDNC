# Copyright 2018 Jörg Franke
#
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

from mtdnc.model.memory_units.MTDNC_cell import MTDNCMemoryCell
from mtdnc.model.memory_units.dnc_cell import DNCMemoryUnitCell


"""
A warpper for the memory units
"""

def get_memory_unit(input_size, config, name='mu', analyse=False, reuse=False, seed=123, dtype=tf.float32):
    memory_length = config['memory_length']
    memory_width = config['memory_width']
    read_heads = config['read_heads']
    dnc_norm = config['dnc_norm']
    bypass_dropout = config['bypass_dropout']

    if 'write_heads' in config:
        write_heads = config['write_heads']

    if config['cell_type'] == 'dnc':
        mu_cell = DNCMemoryUnitCell(input_size, memory_length, memory_width, read_heads, bypass_dropout=bypass_dropout,
                                    dnc_norm=dnc_norm, seed=seed, reuse=reuse, analyse=analyse, dtype=dtype, name=name)
    elif config['cell_type'] == 'MTDNC':
        mu_cell = MTDNCMemoryCell(input_size, memory_length, memory_width, read_heads,
                                             bypass_dropout=bypass_dropout,
                                             dnc_norm=dnc_norm, seed=seed, reuse=reuse, analyse=analyse, dtype=dtype,
                                             name=name)
    else:
        raise UserWarning('Memory Unit: wrong cell type')

    return mu_cell
