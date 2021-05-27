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

set(LIBURING_ROOT "${IREE_ROOT_DIR}/third_party/liburing/src/")

external_cc_library(
  PACKAGE
    liburing
  NAME
    liburing
  ROOT
    ${LIBURING_ROOT}
  INCLUDES
    ${LIBURING_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/default_config/src/include
  SRCS
    "queue.c"
    "register.c"
    "setup.c"
    "syscall.c"
    "syscall.h"
  HDRS
    "include/liburing.h"
    "include/liburing/barrier.h"
    "include/liburing/io_uring.h"
)
