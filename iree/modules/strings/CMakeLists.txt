# Copyright 2019 Google LLC
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

# TODO(suderman): fix cmake build
# bazel_to_cmake: DO NOT EDIT

add_subdirectory(dialect)

iree_cc_library(
  NAME
    strings
  HDRS
    "strings_module.h"
  SRCS
    "strings_module.cc"
  DEPS
    iree::base::api
    iree::base::api_util
    iree::base::tracing
    iree::hal::api
    iree::hal::command_queue
    iree::hal::device
    iree::vm
    iree::vm::module_abi_cc
    absl::core_headers
    absl::memory
    absl::strings
    absl::span
  PUBLIC
)
