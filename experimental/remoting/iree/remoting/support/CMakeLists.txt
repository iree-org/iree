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

iree_cc_library(
  NAME
    support
  HDRS
    "channel.h"
    "io_buffer.h"
    "io_loop.h"
    "io_loop_uring.cc.inc"
    "platform.h"
    "socket.h"
  SRCS
    "channel.cc"
    "io_buffer.cc"
    "io_loop.cc"
    "socket.cc"
  DEPS
    ::linux_uring
    iree::base::api
    iree::base::logging
)

iree_cc_library(
  NAME
    linux_uring
  HDRS
    "linux_uring.h"
  DEPS
    $<IF:${IREE_HAVE_IO_URING},liburing_liburing,>
  DEFINES
    $<IF:${IREE_HAVE_IO_URING},IREE_REMOTING_HAVE_URING=1,>
)
