# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np

# Jax is the most accessible way to get at an xla_client.
# python -m pip install --upgrade pip
# python -m pip install --upgrade jax jaxlib
from jaxlib import xla_client

ops = xla_client.ops

builder = xla_client.XlaBuilder("testbuilder")
in_shape = np.array([4], dtype=np.float32)
in_feed = ops.Parameter(builder, 0, xla_client.shape_from_pyval(in_shape))
result = ops.Add(in_feed, ops.Constant(builder, np.float32(1.0)))
xla_computation = builder.Build(result)

this_dir = os.path.dirname(__file__)
with open(os.path.join(this_dir, "xla_sample.pb"), "wb") as f:
  f.write(xla_computation.as_serialized_hlo_module_proto())
with open(os.path.join(this_dir, "xla_sample.hlo"), "wt") as f:
  f.write(xla_computation.as_hlo_text())
