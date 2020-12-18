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

iree_cc_binary(
  NAME
    iree-simple-socket-server
  OUT
    iree-simple-socket-server
  SRCS
    "iree-simple-socket-server-main.cc"
  DEPS
    iree::base::logging
    experimental::remoting::iree::remoting::protocol_v1
    experimental::remoting::iree::remoting::schemas::protocol_v1_c_fbs
    experimental::remoting::iree::remoting::support
)

iree_cc_binary(
  NAME
    iree-test-client
  OUT
    iree-test-client
  SRCS
    "iree-test-client-main.cc"
  DEPS
    iree::base::logging
    experimental::remoting::iree::remoting::protocol_v1
    experimental::remoting::iree::remoting::schemas::protocol_v1_c_fbs
    experimental::remoting::iree::remoting::support
)

iree_cc_binary(
  NAME
    sample-echo-server
  OUT
    sample-echo-server
  SRCS
    "sample-echo-server-main.cc"
  DEPS
    iree::base::logging
    experimental::remoting::iree::remoting::support
)
