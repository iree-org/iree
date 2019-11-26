# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_subdirectory(debug)

iree_cc_library(
  NAME
    api
  HDRS
    "api.h"
  SRCS
    "api.cc"
  DEPS
    absl::time
    iree::base::api
    iree::base::api_util
    iree::base::tracing
    iree::hal::api
    iree::hal::buffer_view
    iree::hal::driver_registry
    iree::rt
  PUBLIC
)

iree_cc_library(
  NAME
    rt
  SRCS
    "context.cc"
    "function.cc"
    "instance.cc"
    "invocation.cc"
    "module_printer.cc"
    "source_location.cc"
    "stack.cc"
    "stack_frame.cc"
    "stack_trace.cc"
  HDRS
    "context.h"
    "disassembler.h"
    "function.h"
    "function_signature.h"
    "instance.h"
    "invocation.h"
    "module.h"
    "module_printer.h"
    "module_signature.h"
    "policy.h"
    "source_location.h"
    "source_resolver.h"
    "stack.h"
    "stack_frame.h"
    "stack_trace.h"
  DEPS
    absl::core_headers
    absl::inlined_vector
    absl::strings
    absl::synchronization
    absl::time
    absl::optional
    absl::span
    iree::base::bitfield
    iree::base::intrusive_list
    iree::base::ref_ptr
    iree::base::status
    iree::base::tracing
    iree::hal::buffer_view
    iree::hal::device_manager
    iree::rt::debug::debug_server_interface
  PUBLIC
)
