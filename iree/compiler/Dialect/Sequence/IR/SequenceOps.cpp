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

#include "iree/compiler/Dialect/Sequence/IR/SequenceOps.h"

#include <iostream>

#include "llvm/Support/raw_os_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Sequence {

//===----------------------------------------------------------------------===//
// sequence.map
//===----------------------------------------------------------------------===//

static LogicalResult verifyMapOp(MapOp &op) {
  auto inputElementType =
      op.input_sequence().getType().cast<SequenceType>().getTargetType();
  auto outputElementType =
      op.output_sequence().getType().cast<SequenceType>().getTargetType();
  auto func_name = op.mapping_functionAttr().getValue();
  auto *symbolOp = SymbolTable::lookupNearestSymbolFrom(op, func_name);
  if (!symbolOp) {
    return op.emitOpError() << "mapping function " << func_name << " not found";
  }
  auto mappingOp = dyn_cast<FuncOp>(symbolOp);
  if (!mappingOp) {
    return op.emitOpError()
           << "mapping function " << func_name << " not a function";
  }
  if (mappingOp.getNumArguments() != 1) {
    return op.emitOpError()
           << "mapping function must take exactly one argument; "
           << mappingOp.getName() << " is " << mappingOp.getType();
  }
  if (mappingOp.getNumResults() != 1) {
    return op.emitOpError()
           << "mapping function must return exactly one result; "
           << mappingOp.getName() << " is " << mappingOp.getType();
  }
  auto mappingArgType = mappingOp.getType().getInput(0);
  if (mappingArgType != inputElementType) {
    return op.emitOpError()
           << "mapping function expects argument of type " << mappingArgType
           << ", but the input sequence has elements of type "
           << inputElementType;
  }
  auto mappingResultType = mappingOp.getType().getResult(0);
  if (mappingResultType != outputElementType) {
    return op.emitOpError()
           << "mapping function returns result of type " << mappingResultType
           << ", but the output sequence has elements of type "
           << outputElementType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Sequence/IR/SequenceOps.cpp.inc"

}  // namespace Sequence
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
