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
    legacy_library_loader
  HDRS
    "legacy_library_loader.h"
  SRCS
    "legacy_library_loader.cc"
  DEPS
    iree::base::api
    iree::base::dynamic_library
    iree::base::flatcc
    iree::base::internal::file_io
    iree::base::internal::file_path
    iree::base::tracing
    iree::hal::api
    iree::hal::local
    iree::schemas::dylib_executable_def_c_fbs
  DEFINES
    "IREE_HAL_HAVE_LEGACY_LIBRARY_LOADER=1"
  PUBLIC
)

iree_cc_library(
  NAME
    system_library_loader
  HDRS
    "system_library_loader.h"
  SRCS
    "system_library_loader.c"
  DEPS
    iree::base::api
    iree::base::flatcc
    iree::base::internal::file_io
    iree::base::tracing
    iree::hal::api
    iree::hal::local
  DEFINES
    "IREE_HAL_HAVE_SYSTEM_LIBRARY_LOADER=1"
  PUBLIC
)

if(${IREE_HAL_DRIVER_VMLA})

iree_cc_library(
  NAME
    vmla_module_loader
  HDRS
    "vmla_module_loader.h"
  SRCS
    "vmla_module_loader.cc"
  DEPS
    iree::base::api
    iree::base::flatcc
    iree::base::tracing
    iree::hal::api
    iree::hal::local
    iree::modules::vmla::op_module
    iree::schemas::vmla_executable_def_c_fbs
    iree::vm
    iree::vm::bytecode_module
  DEFINES
    "IREE_HAL_HAVE_VMLA_MODULE_LOADER=1"
  PUBLIC
)

endif()
