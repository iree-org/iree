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

iree_add_all_subdirs()

iree_cc_library(
  NAME
    shim
  HDRS
    "include/tensorflow/lite/c/c_api.h"
    "include/tensorflow/lite/c/c_api_experimental.h"
    "include/tensorflow/lite/c/common.h"
  SRCS
    "interpreter.c"
    "interpreter.h"
    "model.c"
    "model.h"
    "options.c"
    "options.h"
    "shim.c"
    "shim.h"
    "tensor.c"
    "tensor.h"
  DEPS
    iree::base::api
    iree::base::core_headers
    iree::base::internal
    iree::base::tracing
    iree::hal::api
    iree::hal::drivers
    iree::modules::hal
    iree::vm
    iree::vm::bytecode_module
  PUBLIC
)

iree_cc_test(
  NAME
    smoke_test
  SRCS
    "smoke_test.cc"
  DEPS
    ::shim
    bindings::tflite::testdata::add_dynamic_cc
    bindings::tflite::testdata::add_multi_cc
    bindings::tflite::testdata::add_static_cc
    iree::testing::gtest
    iree::testing::gtest_main
)
