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

  # Additional libraries containing statically registered functions/flags, which
  # should always be linked in to binaries.
  #
  # TODO(scotttodd): Make the ALWAYSLINK property apply transitively and force
  #                  MLIR libs to generally be ALWAYSLINK wherever used in IREE.
  set(_ALWAYSLINK_LIBS
    MLIRAffineOps
    MLIRAnalysis
    MLIREDSC
    MLIRParser
    MLIRPass
    MLIRSPIRV
    MLIRSPIRVSerialization
    MLIRSPIRVTransforms
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

  foreach(LIB ${_ALWAYSLINK_LIBS})
    # Aliased targets are always in-project, so we control them and can set
    # ALWAYSLINK on them directly.
    # TODO(scotttodd): add the ALWAYSLINK property to MLIR libraries elsewhere
    get_target_property(_ALIASED_TARGET ${LIB} ALIASED_TARGET)
    if(NOT _ALIASED_TARGET)
      set_property(TARGET ${LIB} PROPERTY ALWAYSLINK 1)
    endif()
  endforeach(LIB)

  iree_cc_binary(
    NAME
      iree_tblgen
    OUT
      iree-tblgen
    SRCS
      "${IREE_ROOT_DIR}/third_party/mlir/tools/mlir-tblgen/mlir-tblgen.cpp"
    DEPS
      iree::compiler::Dialect::VM::Tools
      LLVMMLIRTableGen
      MLIRSupport
      LLVMTableGen
    LINKOPTS
      "-lpthread"
  )
  add_executable(iree-tblgen ALIAS iree_tools_iree_tblgen)

  iree_cc_binary(
    NAME
      iree_translate
    OUT
      iree-translate
    SRCS
      "iree_translate_main.cc"
    DEPS
      ${_ALWAYSLINK_LIBS}
      iree::compiler::Translation::Interpreter
      iree::compiler::Translation::Sequencer
      iree::compiler::Translation::SPIRV
      MLIRTranslateClParser
  )

endif()
