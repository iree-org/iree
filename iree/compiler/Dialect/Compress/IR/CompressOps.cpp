// Copyright 2019 Google LLC
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

#include "iree/compiler/Dialect/Compress/IR/CompressOps.h"

#include "mlir/Dialect/QuantOps/QuantOps.h"
#include "mlir/Dialect/QuantOps/QuantTypes.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {

#include "iree/compiler/Dialect/Compress/IR/CompressStructs.cpp.inc"

namespace IREE {
namespace Compress {

using quant::QuantizedType;

//===----------------------------------------------------------------------===//
// QuantRegionOp
//===----------------------------------------------------------------------===//

static bool isValidWideningConversion(Type fromType, Type toType) {
  if (fromType == toType) return true;
  auto fromExpressed = QuantizedType::castToExpressedType(fromType);
  if (fromExpressed && fromExpressed == toType) {
    return true;
  }
  return false;
}

static LogicalResult verifyQuantRegionOp(QuantRegionOp op) {
  auto& bodyBlock = op.body().front();
  if (op.operands().size() != bodyBlock.getNumArguments()) {
    return op.emitOpError("different arity between op and body block");
  }

  // Verify that operands and block arguments are widened.
  for (auto arg : llvm::zip(op.getOperands(), bodyBlock.getArguments())) {
    auto fromType = std::get<0>(arg).getType();
    auto toType = std::get<1>(arg).getType();
    if (!isValidWideningConversion(fromType, toType)) {
      return op.emitOpError()
             << "incompatible operand to block argument type conversion: "
             << fromType << " does not implicitly widen to " << toType;
    }
  }

  // Verify that block results to op results are narrowed.
  if (op.getResults().size() !=
      bodyBlock.getTerminator()->getOperands().size()) {
    return op.emitOpError("different arity between body block and results");
  }
  for (auto resultType : llvm::zip(bodyBlock.getTerminator()->getOperandTypes(),
                                   op.getResultTypes())) {
    auto fromType = std::get<0>(resultType);
    auto toType = std::get<1>(resultType);
    if (!isValidWideningConversion(toType, fromType)) {
      return op.emitOpError()
             << "incompatible block to result type conversion: " << fromType
             << " does not implicitly narrow to " << toType;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Compress/IR/CompressOps.cpp.inc"

}  // namespace Compress
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
