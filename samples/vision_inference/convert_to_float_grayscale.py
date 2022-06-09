#!/usr/bin/env python3

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from PIL import Image
import numpy as np

# Read image from stdin (in any format supported by PIL).
with Image.open(sys.stdin.buffer) as color_img:
  # Convert to grayscale.
  grayscale_img = color_img.convert('L')
  # Rescale to a float32 in range [0.0, 1.0].
  grayscale_arr = np.array(grayscale_img)
  grayscale_arr_f32 = grayscale_arr.astype(np.float32) / 255.0
  # Write bytes back out to stdout.
  sys.stdout.buffer.write(grayscale_arr_f32.tobytes())
