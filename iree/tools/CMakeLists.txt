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

if(${IREE_BUILD_COMPILER})

  # TODO(benvanik): fix the ALWAYSLINK property and force MLIR libs.
  set(_ALWAYSLINK_LIBS
    MLIRAffineOps
    MLIRAnalysis
    MLIREDSC
    MLIRParser
    MLIRPass
    MLIRSPIRV
    MLIRSPIRVSerialization
    MLIRStandardOps
    MLIRTransforms
    MLIRTranslation
    MLIRSupport
    MLIRVectorOps
    iree::compiler::IR
    iree::compiler::IR::Sequencer
    iree::compiler::IR::Interpreter
    tensorflow::mlir_xla
  )

  iree_cc_binary(
    NAME
      iree_translate
    OUT
      iree-translate
    SRCS
      "${IREE_ROOT_DIR}/third_party/mlir/tools/mlir-translate/mlir-translate.cpp"
    DEPS
      ${_ALWAYSLINK_LIBS}
      iree::compiler::Translation::Interpreter
      iree::compiler::Translation::Sequencer
      iree::compiler::Translation::SPIRV
      MLIRTranslateClParser
  )

endif()
