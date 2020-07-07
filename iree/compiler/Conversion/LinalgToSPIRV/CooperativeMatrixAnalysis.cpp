// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "iree/compiler/Conversion/LinalgToSPIRV/CooperativeMatrixAnalysis.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/Vector/VectorOps.h"

using namespace mlir;

namespace {
bool isLegalVectorContract(vector::ContractionOp contract) {
  if (llvm::size(contract.masks()) != 0) return false;
  VectorType lhsType = contract.lhs().getType().cast<VectorType>();
  VectorType rhsType = contract.rhs().getType().cast<VectorType>();
  VectorType accType = contract.acc().getType().cast<VectorType>();

  std::tuple<int, int, int> dim(lhsType.getDimSize(0), rhsType.getDimSize(1),
                                lhsType.getDimSize(1));
  bool supportedType = false;
  // Check if the mattrix type can be supported as a cooperative matrix.
  // Currently we have hardcoded checks for what Turing hardware supports.
  // TODO(thomasraoux): Add device information to be able to query what the
  // device supports.
  if (lhsType.getElementType().isInteger(8) &&
      rhsType.getElementType().isInteger(8) &&
      accType.getElementType().isInteger(32) &&
      (dim == std::make_tuple(8, 8, 32) || dim == std::make_tuple(16, 16, 32) ||
       dim == std::make_tuple(16, 8, 32)))
    supportedType = true;

  if (lhsType.getElementType().isF16() && rhsType.getElementType().isF16() &&
      (accType.getElementType().isF16() || accType.getElementType().isF32()) &&
      (dim == std::make_tuple(8, 8, 16) || dim == std::make_tuple(16, 16, 16) ||
       dim == std::make_tuple(16, 8, 16)))
    supportedType = true;

  return supportedType;
}

bool supportsCooperativeMatrix(Operation* op) {
  if (isa<vector::TransferReadOp>(op) || isa<vector::TransferWriteOp>(op))
    return true;
  if (isa<vector::ContractionOp>(op) &&
      isLegalVectorContract(cast<vector::ContractionOp>(op)))
    return true;
  // We only support minimal set of operations right now. We can trivially
  // extend to ALU instructions supporting Cooperative Matrix in SPIR-V spec.
  // We also need to extend to control flow operations, Alloca, etc...
  // TODO(thomasraoux): extend support to more complex chain of instructions.
  return false;
}
}  // namespace
namespace mlir {
namespace iree_compiler {

CooperativeMatrixAnalysis::CooperativeMatrixAnalysis(mlir::Operation* op) {
  auto targetEnv = spirv::TargetEnv(spirv::lookupTargetEnv(op));
  if (!targetEnv.allows(spirv::Capability::CooperativeMatrixNV) ||
      !targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix))
    return;

  op->walk([&](Operation* op) {
    auto contract = dyn_cast<vector::ContractionOp>(op);
    if (contract == nullptr) return;
    auto hasVectorDest = [](Operation* op) {
      for (auto resultType : op->getResultTypes()) {
        if (resultType.isa<VectorType>()) return true;
      }
      return false;
    };
    auto dependentOps = getSlice(op, hasVectorDest);
    for (auto* dependeOp : dependentOps) {
      // If any instruction cannot use cooperative matrix drop the whole chaine.
      // In the future we can introduce "bitcast" type of conversion to allow
      // the same value to be used as both cooperative matrix as well as an
      // array.
      if (!supportsCooperativeMatrix(dependeOp)) {
        return;
      }
    }
    // All the dependent instruction can use cooperative matrix type. We can
    // mark the whole chain of operations as using cooperative matrix.
    usesCooperativeMatrix.insert(op);
    usesCooperativeMatrix.insert(dependentOps.begin(), dependentOps.end());
  });
}
}  // namespace iree_compiler
}  // namespace mlir
