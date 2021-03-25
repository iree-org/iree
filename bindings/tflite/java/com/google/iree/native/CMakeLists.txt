# Copyright 2021 Google LLC
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
    interpreter_wrapper
  HDRS
    "interpreter_wrapper.h"
  SRCS
    "interpreter_wrapper.cc"
  DEPS
    ::model_wrapper
    ::options_wrapper
    ::tensor_wrapper
    absl::memory
    bindings::tflite::shim
    iree::base::api
  PUBLIC
)

iree_cc_library(
  NAME
    model_wrapper
  HDRS
    "model_wrapper.h"
  SRCS
    "model_wrapper.cc"
  DEPS
    bindings::tflite::shim
  PUBLIC
)

iree_cc_library(
  NAME
    options_wrapper
  HDRS
    "options_wrapper.h"
  SRCS
    "options_wrapper.cc"
  DEPS
    bindings::tflite::shim
  PUBLIC
)

iree_cc_library(
  NAME
    tensor_wrapper
  HDRS
    "tensor_wrapper.h"
  SRCS
    "tensor_wrapper.cc"
  DEPS
    ::tflite_macros
    bindings::tflite::shim
  PUBLIC
)

iree_cc_library(
  NAME
    tflite_macros
  HDRS
    "tflite_macros.h"
  DEPS
    bindings::tflite::shim
    iree::base::api
  PUBLIC
)

iree_cc_test(
  NAME
    interpreter_wrapper_test
  SRCS
    "interpreter_wrapper_test.cc"
  DEPS
    ::interpreter_wrapper
    ::model_wrapper
    ::options_wrapper
    ::tensor_wrapper
    bindings::tflite::shim
    bindings::tflite::testdata::add_static_cc
    iree::base::api
    iree::testing::gtest
    iree::testing::gtest_main
)
