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

//===- IndexComputation.cpp ------------------------------------*- C++//-*-===//
//
// For an IREE dispatch function, compute the map from workitem ID to index of
// tensor computed within that workitem.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Translation/SPIRV/IndexComputation.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<bool> doAffineExprSimplify(
    "simplify-spirv-affine-exprs",
    llvm::cl::desc("Simplify affine expressions during code-generation."),
    llvm::cl::init(true));

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Reshape Utility Functions
//===----------------------------------------------------------------------===//

namespace {
/// Handles shapes for scalars. Shape of scalars are represented as empty vetor,
/// i.e. {}. Its easier to do index propogation to handle the scalar as vector
/// of size 1.
inline SmallVector<int64_t, 4> handleIfScalar(ArrayRef<int64_t> shape) {
  SmallVector<int64_t, 4> resultShape;
  if (shape.empty()) {
    return {1};
  }
  return SmallVector<int64_t, 4>(shape.begin(), shape.end());
}

/// Reshapes are often used to either add a dimension of size 1 or remove a
/// dimension of size 1. Recognizing such cases can make the code-generation
/// easier. The AffineMap needs to either add a constant 0 in the range for such
/// added dimensions or drop those dimensions.
inline LogicalResult getAffineExprForAddOrRemoveDimension(
    Builder &builder, ArrayRef<AffineExpr> resultExprs,
    ArrayRef<int64_t> resultShape, ArrayRef<int64_t> operandShape,
    SmallVectorImpl<AffineExpr> &operandExprs) {
  auto resultIndex = resultShape.size();
  auto operandIndex = operandShape.size();
  operandExprs.resize(operandShape.size());
  // Try to match up the dimensions of the operand and result by ignoring any
  // dimensions of size of 1 that are introduced.
  while (resultIndex > 0 && operandIndex > 0) {
    if (resultShape[resultIndex - 1] == -1 ||
        operandShape[operandIndex - 1] == -1) {
      return failure();
    }
    if (resultShape[resultIndex - 1] == operandShape[operandIndex - 1]) {
      operandExprs[operandIndex - 1] = resultExprs[resultIndex - 1];
      resultIndex--;
      operandIndex--;
      continue;
    }
    if (resultShape[resultIndex - 1] == 1) {
      // This is a dimension that is added on the operand. This affine
      // expression corresponding to this dimension is dropped.
      resultIndex--;
      continue;
    }
    if (operandShape[operandIndex - 1] == 1) {
      // This is a dimension of size 1 of the operand that is dropped. Add a
      // constant expr 0.
      operandExprs[operandIndex - 1] = builder.getAffineConstantExpr(0);
      operandIndex--;
      continue;
    }
    return failure();
  }
  // Any remaining dimensions should be 1.
  while (resultIndex > 0) {
    if (resultShape[resultIndex - 1] != 1) {
      return failure();
    }
    resultIndex--;
  }
  while (operandIndex > 0) {
    if (operandShape[operandIndex - 1] != 1) {
      return failure();
    }
    // This is a dimension of size 1 that is dropped. Add a constant expression
    // 0.
    operandExprs[operandIndex - 1] = builder.getAffineConstantExpr(0);
    operandIndex--;
  }
  return success();
}

/// Constructs the strides of an array assuming a row-major packed layout.
// TODO(ravishankarm): This assumes the shape are static. When using dynamic
// shapes, parameters of each dimension can be used to construct AffineExpr for
// strides along each dimension. Note that multiplying two symbolic constants is
// technically not affine, but you could use another symbol to represent the
// product, so it should be still representable as affine exprs.
inline LogicalResult getRowMajorPackedStrides(
    Builder &builder, ArrayRef<int64_t> shape,
    SmallVectorImpl<AffineExpr> &strides) {
  strides.resize(shape.size());
  int64_t stride = 1;
  for (auto dim : enumerate(reverse(shape))) {
    if (dim.value() < 0) {
      // TODO(ravishankarm) : Better error message.
      return failure();
    }
    strides[shape.size() - 1 - dim.index()] =
        builder.getAffineConstantExpr(stride);
    stride *= dim.value();
  }
  return success();
}

/// Linearizes the index of the result position accessed using the shape of the
/// result tensor and delinearizes it to get the position of the operand.
inline LogicalResult getAffineExprForReshape(
    Builder &builder, unsigned numDims, unsigned numSymbols,
    ArrayRef<AffineExpr> resultExprs, ArrayRef<int64_t> resultShape,
    ArrayRef<int64_t> operandShape, SmallVectorImpl<AffineExpr> &operandExprs) {
  // To linearize the index, assume that the memory is laid out in
  // packed-row-major layout based on the shape.
  // TODO(ravishankarm) : When there is stride information, use that to map from
  // index to memory location.
  SmallVector<AffineExpr, 4> resultStrides;
  if (failed(getRowMajorPackedStrides(builder, resultShape, resultStrides))) {
    return failure();
  }
  AffineExpr linearizedExpr;
  for (auto index : enumerate(resultExprs)) {
    auto val = getAffineBinaryOpExpr(AffineExprKind::Mul, index.value(),
                                     resultStrides[index.index()]);
    if (doAffineExprSimplify) {
      val = simplifyAffineExpr(val, numDims, numSymbols);
    }
    linearizedExpr = (index.index() ? getAffineBinaryOpExpr(AffineExprKind::Add,
                                                            linearizedExpr, val)
                                    : val);
    if (doAffineExprSimplify) {
      linearizedExpr = simplifyAffineExpr(val, numDims, numSymbols);
    }
  }

  // Unlinearize the index, assuming row-major-packed layout.
  // TODO(ravishankarm) : When there is stride information, use that to map from
  // memory location to index.
  SmallVector<AffineExpr, 4> operandStrides;
  if (failed(getRowMajorPackedStrides(builder, operandShape, operandStrides))) {
    return failure();
  }
  operandExprs.resize(operandStrides.size());
  for (auto stride : enumerate(operandStrides)) {
    if (stride.index() == operandStrides.size() - 1) {
      operandExprs[stride.index()] = linearizedExpr;
      break;
    }
    auto expr = getAffineBinaryOpExpr(AffineExprKind::FloorDiv, linearizedExpr,
                                      stride.value());
    operandExprs[stride.index()] =
        (doAffineExprSimplify ? simplifyAffineExpr(expr, numDims, numSymbols)
                              : expr);

    linearizedExpr = getAffineBinaryOpExpr(AffineExprKind::Mod, linearizedExpr,
                                           stride.value());
    if (doAffineExprSimplify) {
      linearizedExpr = simplifyAffineExpr(linearizedExpr, numDims, numSymbols);
    }
  }
  return success();
}
}  // namespace

LogicalResult getReshapeOperandMap(Builder &builder, AffineMap resultIndexMap,
                                   ArrayRef<int64_t> resultShapeRef,
                                   ArrayRef<int64_t> operandShapeRef,
                                   AffineMap &operandIndexMap) {
  auto resultShape = handleIfScalar(resultShapeRef);
  auto operandShape = handleIfScalar(operandShapeRef);
  auto resultExprs = resultIndexMap.getResults();
  assert(resultShape.size() == resultExprs.size() &&
         "Ranks of the Domain of index map and result must be the same");
  SmallVector<AffineExpr, 4> operandExprs;
  if (failed(getAffineExprForAddOrRemoveDimension(
          builder, resultExprs, resultShape, operandShape, operandExprs)) &&
      failed(getAffineExprForReshape(
          builder, resultIndexMap.getNumDims(), resultIndexMap.getNumSymbols(),
          resultExprs, resultShape, operandShape, operandExprs))) {
    return failure();
  }
  assert(operandExprs.size() == operandShape.size() &&
         "expected as many exprs for the operand as the rank of the operand");
  operandIndexMap =
      builder.getAffineMap(resultIndexMap.getNumDims(),
                           resultIndexMap.getNumSymbols(), operandExprs);

  return success();
}

LogicalResult IndexPropagation::propagateIndexMap(
    Operation *op, IndexComputationCache &indexMap) const {
  if (op->getNumResults() == 0) {
    // Nothing to do for this op.
    return success();
  }
  if (op->getNumResults() != 1) {
    return op->emitError(
        "default index propagation handles case with a single-return value");
  }
  // Initialize the storage for all the operands.
  for (auto arg : op->getOperands()) {
    indexMap[arg];
  }
  for (auto &resultIndexMap : indexMap[op->getResult(0)]) {
    SmallVector<AffineMap, 4> operandIndices;
    if (failed(this->propagateIndexMap(op, resultIndexMap.first,
                                       operandIndices))) {
      return failure();
    }
    assert(operandIndices.size() == op->getNumOperands() &&
           "Expected as many indices as operands");
    for (auto arg : enumerate(op->getOperands())) {
      indexMap[arg.value()][operandIndices[arg.index()]];
      resultIndexMap.second.push_back(operandIndices[arg.index()]);
    }
  }
  return success();
}

void dumpIndexCache(IndexComputationCache &indexMap) {
  for (auto &el : indexMap) {
    // llvm::errs() << "Value : " << *(el.first);
    // llvm::errs().flush();
    if (isa<OpResult>(el.first)) {
      llvm::errs() << "Operation : " << el.first->getDefiningOp()->getName();
    } else if (isa<BlockArgument>(el.first)) {
      llvm::errs() << "BlockArgument";
    }
    for (auto &used : el.second) {
      llvm::errs() << "\n\t" << used.first << " : [";
      std::string sep = "";
      for (auto &operand : used.second) {
        llvm::errs() << sep << operand;
        sep = ", ";
      }
      llvm::errs() << "]";
    }
    llvm::errs() << "\n";
  }
  llvm::errs() << "\n";
}

}  // namespace iree_compiler
}  // namespace mlir
