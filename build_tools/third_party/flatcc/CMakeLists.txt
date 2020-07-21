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

set(FLATCC_ROOT "${IREE_ROOT_DIR}/third_party/flatcc/")

# NOTE: we exclude JSON parsing/printing to avoid additional dependencies.
external_cc_library(
  PACKAGE
    flatcc
  NAME
    runtime
  ROOT
    ${FLATCC_ROOT}
  INCLUDES
    "${FLATCC_ROOT}/include"
  SRCS
    "src/runtime/builder.c"
    "src/runtime/emitter.c"
    "src/runtime/refmap.c"
    "src/runtime/verifier.c"
  HDRS
    "include/flatcc/flatcc_accessors.h"
    "include/flatcc/flatcc_alloc.h"
    "include/flatcc/flatcc_assert.h"
    "include/flatcc/flatcc_builder.h"
    "include/flatcc/flatcc_emitter.h"
    "include/flatcc/flatcc_endian.h"
    "include/flatcc/flatcc_epilogue.h"
    "include/flatcc/flatcc_flatbuffers.h"
    "include/flatcc/flatcc_identifier.h"
    "include/flatcc/flatcc_iov.h"
    "include/flatcc/flatcc_portable.h"
    "include/flatcc/flatcc_prologue.h"
    "include/flatcc/flatcc_refmap.h"
    "include/flatcc/flatcc_rtconfig.h"
    "include/flatcc/flatcc_types.h"
    "include/flatcc/flatcc_unaligned.h"
    "include/flatcc/flatcc_verifier.h"
    "include/flatcc/reflection/flatbuffers_common_builder.h"
    "include/flatcc/reflection/flatbuffers_common_reader.h"
  PUBLIC
)
