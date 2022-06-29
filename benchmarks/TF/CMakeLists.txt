# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


################################################################################
#                                                                              #
# Benchmark models from Tensorflow                                             #
#                                                                              #
# Each module specification should be a list containing alternating keys and   #
# values. The fields are: NAME, TAGS, SOURCE, ENTRY_FUNCTION, and              #
# FUNCTION_INPUTS. See the iree_benchmark_suite definition for details         #
# about these fields. Note that these must be quoted when used as arguments.   #
#                                                                              #
################################################################################

set(MINILM_L12_H384_UNCASED_INT32_MODULE
  NAME
    "MiniLML12H384Uncased"
  TAGS
    "int32"
  SOURCE
    # Converted from https://huggingface.co/microsoft/MiniLM-L12-H384-uncased/commit/44acabbec0ef496f6dbc93adadea57f376b7c0ec
    "https://storage.googleapis.com/iree-model-artifacts/minilm-l12-h384-uncased-tf-model.tar.gz"
  ENTRY_FUNCTION
    "predict"
  FUNCTION_INPUTS
    "1x512xi32,1x512xi32,1x512xi32"
)

################################################################################
# Add benchmarks for all platforms.                                            #
################################################################################
include(linux-x86_64.cmake)
