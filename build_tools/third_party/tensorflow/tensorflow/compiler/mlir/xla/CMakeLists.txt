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
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/build_tools/third_party/tensorflow/tensorflow/compiler/mlir/xla/ir")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/build_tools/third_party/tensorflow/tensorflow/compiler/mlir/xla/transforms")

external_cc_library(
  PACKAGE
    tensorflow
  NAME
    mlir_xla
  ROOT
    ${TF_MLIR_XLA_SRC_ROOT}
  SRCS
    "convert_op_folder.cc"
    "ir/broadcast_utils.cc"
    "ir/chlo_ops.cc"
    "ir/dialect_registration.cc"
    "ir/hlo_ops.cc"
    "ir/hlo_utils.cc"
    "ir/lhlo_ops.cc"
    "transforms/legalize_control_flow.cc"
    "transforms/legalize_to_standard.cc"
    "transforms/lower_general_dot.cc"
    "transforms/materialize_broadcasts.cc"
    "transforms/unfuse_batch_norm.cc"
    "transforms/xla_legalize_to_linalg.cc"
  HDRS
    "convert_op_folder.h"
    "ir/chlo_ops.h"
    "ir/hlo_ops.h"
    "ir/hlo_utils.h"
    "ir/lhlo_ops.h"
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
    tensorflow_mlir_xla_canonicalize_gen
    tensorflow_mlir_xla_chlo_ops_gen
    tensorflow_mlir_xla_hlo_ops_gen
    tensorflow_mlir_xla_hlo_ops_base_gen
    tensorflow_mlir_xla_legalize_to_standard_patterns_gen
    tensorflow_mlir_xla_lhlo_ops_gen
    LLVMSupport
    MLIRAnalysis
    MLIRIR
    MLIRLinalgOps
    MLIRPass
    MLIRShape
    MLIRTransformUtils
  PUBLIC
)

external_tablegen_library(
  PACKAGE
    tensorflow
  NAME
    mlir_xla_canonicalize_gen
  TBLGEN
    MLIR
  ROOT
    ${TF_MLIR_XLA_SRC_ROOT}
  SRCS
    "transforms/canonicalize.td"
  OUTS
    -gen-rewriters transforms/generated_canonicalize.inc
)

external_tablegen_library(
  PACKAGE
    tensorflow
  NAME
    mlir_xla_hlo_ops_gen
  TBLGEN
    MLIR
  ROOT
    ${TF_MLIR_XLA_SRC_ROOT}
  SRCS
    "ir/hlo_ops.td"
  OUTS
    -gen-op-decls ir/hlo_ops.h.inc
    -gen-op-defs ir/hlo_ops.cc.inc
    -gen-struct-attr-decls ir/hlo_structs.h.inc
    -gen-struct-attr-defs ir/hlo_structs.cc.inc
)

external_tablegen_library(
  PACKAGE
    tensorflow
  NAME
    mlir_xla_chlo_ops_gen
  TBLGEN
    MLIR
  ROOT
    ${TF_MLIR_XLA_SRC_ROOT}
  SRCS
    "ir/chlo_ops.td"
  OUTS
    -gen-op-decls ir/chlo_ops.h.inc
    -gen-op-defs ir/chlo_ops.cc.inc
)

external_tablegen_library(
  PACKAGE
    tensorflow
  NAME
    mlir_xla_hlo_ops_base_gen
  TBLGEN
    MLIR
  ROOT
    ${TF_MLIR_XLA_SRC_ROOT}
  SRCS
    "ir/hlo_ops_base.td"
  OUTS
    -gen-op-decls ir/hlo_ops_base.h.inc
    -gen-op-defs ir/hlo_ops_base.cc.inc
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

external_tablegen_library(
  PACKAGE
    tensorflow
  NAME
    mlir_xla_lhlo_ops_gen
  TBLGEN
    MLIR
  ROOT
    ${TF_MLIR_XLA_SRC_ROOT}
  SRCS
    "ir/lhlo_ops.td"
  OUTS
    -gen-op-decls ir/lhlo_ops.h.inc
    -gen-op-defs ir/lhlo_ops.cc.inc
)
