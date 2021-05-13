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

set(STBLIB_ROOT "${IREE_ROOT_DIR}/third_party/stblib/")


external_cc_library(
  PACKAGE
    stblib
  NAME
    stb_image
  ROOT
    "${IREE_ROOT_DIR}/build_tools/third_party/stblib/"
  SRCS
    "stb_image.c"
  COPTS
    "-Wno-implicit-fallthrough"
  INCLUDES
    ${STBLIB_ROOT}
)
