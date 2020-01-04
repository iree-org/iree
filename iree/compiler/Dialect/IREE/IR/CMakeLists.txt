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

iree_cc_library(
  NAME
    IR
  HDRS
    "IREEDialect.h"
    "IREEOps.h"
    "IREEOps.h.inc"
    "IREETraits.h"
    "IREETypes.h"
  SRCS
    "IREEDialect.cpp"
    "IREEOps.cpp"
    "IREEOps.cpp.inc"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRStandardOps
    MLIRSupport
  ALWAYSLINK
  PUBLIC
)


iree_tablegen_library(
  NAME
    IREEOpsGen
  SRCS
    IREEOps.td
  OUTS
    -gen-op-decls IREEOps.h.inc
    -gen-op-defs IREEOps.cpp.inc
)
