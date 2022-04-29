# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
