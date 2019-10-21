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

# TODO(benvanik): re-enable debugger after refactoring.
# iree_cc_library(
#   NAME
#     debug_client
#   SRCS
#     "debug_client.cc"
#   HDRS
#     "debug_client.h"
#   DEPS
#     absl::flat_hash_map
#     absl::optional
#     absl::strings
#     absl::span
#     iree::base::source_location
#     iree::base::status
#     iree::schemas
#     iree::rt::debug::debug_client_interface
#     iree::rt::debug::debug_client_tcp
#   PUBLIC
# )

# iree_cc_library(
#   NAME
#     debug_client_interface
#   HDRS
#     "debug_client.h"
#   DEPS
#     absl::flat_hash_map
#     absl::optional
#     absl::strings
#     absl::span
#     iree::base::status
#     iree::schemas
#   PUBLIC
# )

# iree_cc_library(
#   NAME
#     debug_client_tcp
#   SRCS
#     "debug_client_tcp.cc"
#   DEPS
#     absl::flat_hash_map
#     absl::memory
#     absl::strings
#     absl::span
#     flatbuffers
#     iree::base::flatbuffer_util
#     iree::base::status
#     iree::schemas
#     iree::rt::debug::debug_client_interface
#     iree::rt::debug::debug_tcp_util
#     iree::rt
#   PUBLIC
# )

# if(${IREE_ENABLE_DEBUG})
#   iree_cc_library(
#     NAME
#       debug_server
#     HDRS
#       "debug_server.h"
#     DEPS
#       flatbuffers
#       iree::schemas
#       iree::base::status
#       iree::rt::debug::debug_server_interface
#       iree::rt::debug::debug_server_tcp
#     PUBLIC
#   )
# else()
#   iree_cc_library(
#     NAME
#       debug_server
#     SRCS
#       "debug_server_disabled.cc"
#     HDRS
#       "debug_server.h"
#     DEPS
#       iree::base::status
#     PUBLIC
#   )
# endif()

iree_cc_library(
  NAME
    debug_server
  SRCS
    "debug_server_disabled.cc"
  HDRS
    "debug_server.h"
  DEPS
    iree::base::status
  PUBLIC
)

iree_cc_library(
  NAME
    debug_server_interface
  HDRS
    "debug_server.h"
  DEPS
    iree::base::status
  PUBLIC
)

# TODO(benvanik): re-enable debugger after refactoring.
# iree_cc_library(
#   NAME
#     debug_server_tcp
#   SRCS
#     "debug_server_tcp.cc"
#   DEPS
#     absl::core_headers
#     absl::memory
#     absl::synchronization
#     flatbuffers
#     iree::base::status
#     iree::schemas
#     iree::rt::debug::debug_server_interface
#     iree::rt::debug::debug_service
#     iree::rt::debug::debug_tcp_util
# )
#
# if(${IREE_BUILD_DEBUGGER})
#   iree_cc_library(
#     NAME
#       debug_server_flags
#     SRCS
#       "debug_server_flags.cc"
#     HDRS
#       "debug_server_flags.h"
#     COPTS
#       "-DIREE_DEBUG_EMBEDDED_APP_PRESENT=1"
#     DEPS
#       # TODO(benvanik): EGL/GLESv2
#       absl::flags
#       absl::strings
#       iree::base::memory
#       iree::base::status
#       iree::rt::debug::debug_server
#       iree::tools::debugger::debug_app_embedded
#   )
# else()
#   iree_cc_library(
#     NAME
#       debug_server_flags
#     SRCS
#       "debug_server_flags.cc"
#     HDRS
#       "debug_server_flags.h"
#     DEPS
#       absl::flags
#       absl::strings
#       iree::base::memory
#       iree::base::status
#       iree::rt::debug::debug_server
#   )
# endif()

iree_cc_library(
  NAME
    debug_server_flags
  SRCS
    "debug_server_flags.cc"
  HDRS
    "debug_server_flags.h"
  DEPS
    absl::flags
    absl::strings
    iree::base::memory
    iree::base::status
    iree::rt::debug::debug_server
)

# TODO(benvanik): re-enable debugger after refactoring.
# iree_cc_library(
#   NAME
#     debug_service
#   SRCS
#     "debug_service.cc"
#   HDRS
#     "debug_service.h"
#   DEPS
#     absl::core_headers
#     absl::strings
#     absl::synchronization
#     flatbuffers
#     iree::base::flatbuffer_util
#     iree::base::source_location
#     iree::base::status
#     iree::schemas
#     iree::schemas::reflection_data
#     iree::rt
#     iree::rt::debug::debug_session
#   PUBLIC
# )

# iree_cc_library(
#   NAME
#     debug_session
#   SRCS
#     "debug_session.cc"
#   HDRS
#     "debug_session.h"
#   DEPS
#     absl::core_headers
#     absl::synchronization
#     iree::base::source_location
#     iree::base::status
#     iree::schemas
#     iree::rt
#   PUBLIC
# )

# iree_cc_library(
#   NAME
#     debug_tcp_util
#   HDRS
#     "debug_tcp_util.h"
#   DEPS
#     flatbuffers
#     iree::base::status
#     iree::schemas
#   PUBLIC
# )
