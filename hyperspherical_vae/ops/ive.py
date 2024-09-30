# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The exponentially scaled modified Bessel function of the first kind."""

import numpy as np
import scipy.special

from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

import tensorflow.compat.v2 as tf
import tensorflow.keras.backend as K

@tf.custom_gradient
def ive(v, z):
    """Exponentially scaled modified Bessel function of the first kind."""
    output = K.reshape(tf.numpy_function(
        lambda v, z: np.select(condlist=[v == 0, v == 1],
                               choicelist=[scipy.special.i0e(z, dtype=z.dtype),
                                           scipy.special.i1e(z, dtype=z.dtype)],
                               default=scipy.special.ive(v, z, dtype=z.dtype)), [v, z], z.dtype),
        K.shape(z))

    def grad(dy):
        return None, dy * (ive(v - 1, z) - ive(v, z) * (v + z) / z)

    return output, grad
