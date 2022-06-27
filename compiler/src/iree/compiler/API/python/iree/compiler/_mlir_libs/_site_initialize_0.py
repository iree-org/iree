# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Note that this does some load time, once-only init.
from . import _ireecTransforms


def register_dialects(registry):
  # TODO: Convert the below context_init_hook to be DialectRegistry
  # based and use it here.
  pass


def context_init_hook(context):
  _ireecTransforms.register_all_dialects(context)
  