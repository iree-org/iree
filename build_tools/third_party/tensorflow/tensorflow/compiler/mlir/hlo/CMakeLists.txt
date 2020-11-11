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

external_cc_library(
  PACKAGE
    tensorflow
  NAME
    mlir_hlo
  ROOT
    ${TF_MLIR_HLO_SRC_ROOT}
  DEPS
    MhloDialect
    MhloInferFusibilityOpInterface
    LmhloDialect
    ChloPasses
    MhloToStandard
    MhloPasses
    MhloLhloToLinalg
    MLIRMhloUtils
  PUBLIC
)
