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

iree_add_all_subdirs()

iree_cc_library(
  NAME
    file_handle_win32
  HDRS
    "file_handle_win32.h"
  SRCS
    "file_handle_win32.cc"
  DEPS
    absl::memory
    absl::strings
    iree::base::status
    iree::base::target_platform
  PUBLIC
)

iree_cc_library(
  NAME
    file_io_internal
  SRCS
    "file_io_posix.cc"
    "file_io_win32.cc"
  DEPS
    ::file_handle_win32
    absl::memory
    absl::strings
    iree::base::file_io_hdrs
    iree::base::file_path
    iree::base::status
    iree::base::target_platform
    iree::base::tracing
  PUBLIC
)

iree_cc_library(
  NAME
    file_mapping_internal
  SRCS
    "file_mapping_posix.cc"
    "file_mapping_win32.cc"
  DEPS
    ::file_handle_win32
    absl::memory
    absl::strings
    iree::base::file_mapping_hdrs
    iree::base::target_platform
    iree::base::tracing
  PUBLIC
)

iree_cc_library(
  NAME
    init_internal
  HDRS
    "init_internal.h"
  SRCS
    "init_internal.cc"
  DEPS
    absl::flags_parse
    iree::base::initializer
    iree::base::target_platform
  PUBLIC
)

iree_cc_library(
  NAME
    logging_internal
  HDRS
    "logging.h"
  SRCS
    "logging.cc"
  DEPS
    absl::core_headers
    absl::flags
    iree::base::target_platform
  PUBLIC
)

iree_cc_library(
  NAME
    localfile_internal
  PUBLIC
)

iree_cc_library(
  NAME
    ostringstream
  HDRS
    "ostringstream.h"
  SRCS
    "ostringstream.cc"
  PUBLIC
)

iree_cc_library(
  NAME
    status_internal
  HDRS
    "status.h"
    "status_builder.h"
    "statusor.h"
  SRCS
    "status.cc"
    "status_builder.cc"
    "statusor.cc"
  DEPS
    ::ostringstream
    absl::core_headers
    absl::strings
    iree::base::api
    iree::base::logging
    iree::base::target_platform
  PUBLIC
)
