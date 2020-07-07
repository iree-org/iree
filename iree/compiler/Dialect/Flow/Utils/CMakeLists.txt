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
    Utils
  HDRS
    "DispatchUtils.h"
    "WorkloadUtils.h"
  SRCS
    "DispatchUtils.cpp"
    "WorkloadUtils.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRStandardOps
    MLIRSupport
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::Shape::IR
    tensorflow::mlir_hlo
  PUBLIC
)
