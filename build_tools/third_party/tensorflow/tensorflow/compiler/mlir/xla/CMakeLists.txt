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

set(TF_MLIR_XLA_SRC_ROOT
  "${IREE_ROOT_DIR}/third_party/tensorflow/tensorflow/compiler/mlir/xla/"
)

set(TF_MLIR_XLA_COPTS_BASE
  "-I${IREE_ROOT_DIR}/third_party/tensorflow/"
  "-I${PROJECT_BINARY_DIR}/build_tools/third_party/tensorflow/"
)

# Create directories in the build tree for external_tablegen_library targets to use.
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/build_tools/third_party/tensorflow/tensorflow/compiler/mlir/xla/transforms")

external_cc_library(
  PACKAGE
    tensorflow
  NAME
    mlir_xla
  ROOT
    ${TF_MLIR_XLA_SRC_ROOT}
  SRCS
    "transforms/chlo_legalize_to_hlo.cc"
    "transforms/cycle_detector.cc"
    "transforms/legalize_control_flow.cc"
    "transforms/legalize_to_standard.cc"
    "transforms/lower_general_dot.cc"
    "transforms/materialize_broadcasts.cc"
    "transforms/unfuse_batch_norm.cc"
    "transforms/xla_hlo_fusion.cc"
    "transforms/xla_legalize_to_linalg.cc"
  HDRS
    "transforms/cycle_detector.h"
    "transforms/map_xla_to_scalar_op.h"
    "transforms/passes.h"
    "transforms/rewriters.h"
  COPTS
    ${TF_MLIR_XLA_COPTS_BASE}
  INCLUDES
    ${IREE_ROOT_DIR}/third_party/tensorflow/
    ${CMAKE_BINARY_DIR}/build_tools/third_party/tensorflow/
  DEPS
    absl::core_headers
    absl::flat_hash_set
    absl::memory
    tensorflow_mlir_hlo
    tensorflow_mlir_xla_legalize_to_standard_patterns_gen
    LLVMCore
    LLVMSupport
    MLIRAnalysis
    MLIRDialect
    MLIRIR
    MLIRLinalgOps
    MLIRPass
    MLIRShape
    MLIRStandardOps
    MLIRSupport
    MLIRTransformUtils
    MLIRTransforms
  PUBLIC
)

external_tablegen_library(
  PACKAGE
    tensorflow
  NAME
    mlir_xla_legalize_to_standard_patterns_gen
  TBLGEN
    MLIR
  ROOT
    ${TF_MLIR_XLA_SRC_ROOT}
  SRCS
    "transforms/legalize_to_standard_patterns.td"
  OUTS
    -gen-rewriters transforms/generated_legalize_to_standard.inc
)
