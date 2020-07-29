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

set(TF_MLIR_HLO_SRC_ROOT
  "${IREE_ROOT_DIR}/third_party/tensorflow/tensorflow/compiler/mlir/hlo/"
)

set(TF_MLIR_HLO_COPTS_BASE
  "-I${IREE_ROOT_DIR}/third_party/tensorflow/"
  "-I${PROJECT_BINARY_DIR}/build_tools/third_party/tensorflow/"
)

# Create directories in the build tree for external_tablegen_library targets to use.
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/build_tools/third_party/tensorflow/tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/build_tools/third_party/tensorflow/tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/")

external_cc_library(
  PACKAGE
    tensorflow
  NAME
    mlir_hlo
  ROOT
    ${TF_MLIR_HLO_SRC_ROOT}
  SRCS
    "lib/Dialect/mhlo/IR/chlo_ops.cc"
    "lib/Dialect/mhlo/IR/dialect_registration.cc"
    "lib/Dialect/mhlo/IR/hlo_ops.cc"
    "lib/Dialect/mhlo/IR/infer_fusibility_op_interface.cc"
    "lib/Dialect/mhlo/IR/lhlo_ops.cc"
    "lib/Dialect/mhlo/transforms/chlo_legalize_to_hlo.cc"
    "lib/Dialect/mhlo/transforms/legalize_control_flow.cc"
    "lib/Dialect/mhlo/transforms/legalize_gather_to_torch_index_select.cc"
    "lib/Dialect/mhlo/transforms/legalize_to_standard.cc"
    "lib/Dialect/mhlo/transforms/lower_complex.cc"
    "lib/Dialect/mhlo/transforms/lower_general_dot.cc"
    "lib/Dialect/mhlo/transforms/materialize_broadcasts.cc"
    "lib/Dialect/mhlo/transforms/unfuse_batch_norm.cc"
    "lib/Dialect/mhlo/transforms/mhlo_fusion.cc"
    "lib/Dialect/mhlo/transforms/legalize_to_linalg.cc"
    "lib/utils/broadcast_utils.cc"
    "lib/utils/convert_op_folder.cc"
    "lib/utils/cycle_detector.cc"
    "lib/utils/hlo_utils.cc"
  HDRS
    "include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
    "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
    "include/mlir-hlo/Dialect/mhlo/IR/infer_fusibility_op_interface.h"
    "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
    "include/mlir-hlo/Dialect/mhlo/transforms/map_lmhlo_to_scalar_op.h"
    "include/mlir-hlo/Dialect/mhlo/transforms/passes.h"
    "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
    "include/mlir-hlo/utils/broadcast_utils.h"
    "include/mlir-hlo/utils/convert_op_folder.h"
    "include/mlir-hlo/utils/cycle_detector.h"
    "include/mlir-hlo/utils/hlo_utils.h"
  COPTS
    ${TF_MLIR_HLO_COPTS_BASE}
  INCLUDES
    ${IREE_ROOT_DIR}/third_party/tensorflow/
    ${CMAKE_BINARY_DIR}/build_tools/third_party/tensorflow/
    ${CMAKE_BINARY_DIR}/build_tools/third_party/tensorflow/tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR
    ${CMAKE_BINARY_DIR}/build_tools/third_party/tensorflow/tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/
  DEPS
    absl::core_headers
    LLVMSupport
    MLIRAnalysis
    MLIRInferTypeOpInterface
    MLIRIR
    MLIRPass
    MLIRShape
    MLIRSideEffectInterfaces
    MLIRStandardOps
    MLIRSupport
    MLIRTransforms
    MLIRTransformUtils
    MLIRViewLikeInterface
    tensorflow_mlir_hlo_chlo_ops_gen
    tensorflow_mlir_hlo_hlo_ops_base_gen
    tensorflow_mlir_hlo_hlo_ops_gen
    tensorflow_mlir_hlo_hlo_ops_pattern_gen
    tensorflow_mlir_hlo_infer_fusibility_op_interface_gen
    tensorflow_mlir_hlo_legalize_to_standard_patterns_gen
    tensorflow_mlir_hlo_lower_to_complex_inc_gen
    tensorflow_mlir_hlo_lhlo_ops_gen
    tensorflow_mlir_hlo_xla_canonicalize_gen
  PUBLIC
)

external_tablegen_library(
  PACKAGE
    tensorflow
  NAME
    mlir_hlo_chlo_ops_gen
  TBLGEN
    MLIR
  ROOT
    ${TF_MLIR_HLO_SRC_ROOT}
  SRCS
    "include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.td"
  OUTS
    -gen-op-decls include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h.inc
    -gen-op-defs include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.cc.inc
)

external_tablegen_library(
  PACKAGE
    tensorflow
  NAME
    mlir_hlo_hlo_ops_gen
  TBLGEN
    MLIR
  ROOT
    ${TF_MLIR_HLO_SRC_ROOT}
  SRCS
    "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"
  OUTS
    -gen-op-decls include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h.inc
    -gen-op-defs include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.cc.inc
    -gen-struct-attr-decls include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_structs.h.inc
    -gen-struct-attr-defs include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_structs.cc.inc
)

external_tablegen_library(
  PACKAGE
    tensorflow
  NAME
    mlir_hlo_hlo_ops_base_gen
  TBLGEN
    MLIR
  ROOT
    ${TF_MLIR_HLO_SRC_ROOT}
  SRCS
    "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.td"
  OUTS
    -gen-op-decls include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.h.inc
    -gen-op-defs include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.cc.inc
)

external_tablegen_library(
  PACKAGE
    tensorflow
  NAME
    mlir_hlo_hlo_ops_pattern_gen
  TBLGEN
    MLIR
  ROOT
    ${TF_MLIR_HLO_SRC_ROOT}
  SRCS
    "lib/Dialect/mhlo/IR/hlo_patterns.td"
  OUTS
    -gen-rewriters lib/Dialect/mhlo/IR/hlo_patterns.cc.inc
)

external_tablegen_library(
  PACKAGE
    tensorflow
  NAME
    mlir_hlo_lhlo_ops_gen
  TBLGEN
    MLIR
  ROOT
    ${TF_MLIR_HLO_SRC_ROOT}
  SRCS
    "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.td"
  OUTS
    -gen-op-decls include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h.inc
    -gen-op-defs include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.cc.inc
    -gen-struct-attr-decls include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops_structs.h.inc
    -gen-struct-attr-defs include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops_structs.cc.inc
)

external_tablegen_library(
  PACKAGE
    tensorflow
  NAME
    mlir_hlo_xla_canonicalize_gen
  TBLGEN
    MLIR
  ROOT
    ${TF_MLIR_HLO_SRC_ROOT}
  SRCS
  "lib/Dialect/mhlo/IR/mhlo_canonicalize.td"
  OUTS
    -gen-rewriters lib/Dialect/mhlo/IR/mhlo_canonicalize.inc
)

external_tablegen_library(
  PACKAGE
    tensorflow
  NAME
    mlir_hlo_infer_fusibility_op_interface_gen
  TBLGEN
    MLIR
  ROOT
    ${TF_MLIR_HLO_SRC_ROOT}
  SRCS
    "include/mlir-hlo/Dialect/mhlo/IR/infer_fusibility_op_interface.td"
  OUTS
    -gen-op-interface-decls include/mlir-hlo/Dialect/mhlo/IR/infer_fusibility_op_interface.h.inc
    -gen-op-interface-defs include/mlir-hlo/Dialect/mhlo/IR/infer_fusibility_op_interface.cpp.inc
)

external_tablegen_library(
  PACKAGE
    tensorflow
  NAME
    mlir_hlo_legalize_to_standard_patterns_gen
  TBLGEN
    MLIR
  ROOT
    ${TF_MLIR_HLO_SRC_ROOT}
  SRCS
    "lib/Dialect/mhlo/transforms/legalize_to_standard_patterns.td"
  OUTS
    -gen-rewriters lib/Dialect/mhlo/transforms/generated_legalize_to_standard.inc
)

external_tablegen_library(
  PACKAGE
    tensorflow
  NAME
    mlir_hlo_lower_to_complex_inc_gen
  TBLGEN
    MLIR
  ROOT
    ${TF_MLIR_HLO_SRC_ROOT}
  SRCS
    "lib/Dialect/mhlo/transforms/lower_complex_patterns.td"
  OUTS
    -gen-rewriters lib/Dialect/mhlo/transforms/generated_lower_complex.inc
)
