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

set(PFFFT_ROOT "${IREE_ROOT_DIR}/third_party/pffft/")

external_cc_library(
  PACKAGE
    pffft
  NAME
    pffft
  ROOT
    ${PFFFT_ROOT}
  SRCS
    "src/pffft.c"
  HDRS
    "src/pffft.h"
  INCLUDES
    "${PFFFT_ROOT}/src"
)

external_cc_library(
  PACKAGE
    pffft
  NAME
    fftpack
  ROOT
    ${PFFFT_ROOT}
  SRCS
    "src/fftpack.c"
  HDRS
    "src/fftpack.h"
  INCLUDES
    "${PFFFT_ROOT}/src"
)

external_cc_library(
  PACKAGE
    pffft
  NAME
    pffft_test
  ROOT
    ${PFFFT_ROOT}
  SRCS
    "src/test_pffft.c"
  DEPS
    pffft::fftpack
    pffft::pffft
)
