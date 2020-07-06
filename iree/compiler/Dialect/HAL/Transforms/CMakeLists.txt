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
    Transforms
  HDRS
    "Passes.h"
  SRCS
    "InlineDeviceSwitches.cpp"
    "LinkExecutables.cpp"
    "MaterializeInterfaces.cpp"
    "MaterializeResourceCaches.cpp"
    "MemoizeDeviceQueries.cpp"
    "Passes.cpp"
    "PublicAbiGeneration.cpp"
    "SerializeExecutables.cpp"
    "TranslateExecutables.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRPass
    MLIRStandardOps
    MLIRSupport
    MLIRTransforms
    absl::strings
    iree::base::signature_mangle
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::HAL::Conversion::FlowToHAL
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::HAL::Target
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Dialect::Shape::Transforms
    iree::compiler::Utils
    tensorflow::mlir_hlo
  PUBLIC
)
