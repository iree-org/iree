## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines the groups of models."""

from . import tflite

MOBILE_MODEL_GROUP = [tflite.MOBILENET_V2_MODEL]
