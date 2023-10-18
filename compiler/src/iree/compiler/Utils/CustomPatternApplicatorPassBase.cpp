// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_CUSTOMPATTERNAPPLICATORPASSBASE_H_
#define IREE_COMPILER_UTILS_CUSTOMPATTERNAPPLICATORPASSBASE_H_

#include "iree/compiler/Utils/CustomPatternApplicatorPassBase.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace detail {

/// Helper to get a list of sizes from the given RankedTensorType value.
static SmallVector<Value> getValueRangeTensorSizes(PatternRewriter &rewriter,
                                                   ValueRange vals) {
  SmallVector<Value> flatTensorSizes;
  for (auto val : vals) {
    if (auto tensorType = dyn_cast<RankedTensorType>(val.getType())) {
      for (int64_t i = 0; i < tensorType.getRank(); ++i) {
        flatTensorSizes.push_back(
            rewriter.create<tensor::DimOp>(val.getLoc(), val, i).getResult());
      }
    }
  }
  return flatTensorSizes;
}

static SmallVector<Value> getI32TensorSizes(PatternRewriter &rewriter,
                                            ValueRange vals) {
  SmallVector<Value> flatI32TensorSizes;
  for (auto val : vals) {
    if (isa<IndexType>(val.getType())) {
      flatI32TensorSizes.push_back(
          rewriter
              .create<arith::IndexCastOp>(val.getLoc(),
                                          rewriter.getIntegerType(32), val)
              .getResult());
    }
  }
  return flatI32TensorSizes;
}

static FailureOr<Value> extractValueFromRange(PatternRewriter &rewriter,
                                              ValueRange vals, Attribute attr) {
  IntegerAttr index = dyn_cast<IntegerAttr>(attr);
  if (!index || index.getInt() >= vals.size())
    return failure();
  return vals[index.getInt()];
}

void populateCommonNativeRewriteHelpers(RewritePatternSet &patterns) {
  mlir::registerConversionPDLFunctions(patterns);
  patterns.getPDLPatterns().registerRewriteFunction("extract_value",
                                                    extractValueFromRange);
  patterns.getPDLPatterns().registerRewriteFunction("get_tensor_sizes",
                                                    getValueRangeTensorSizes);
  patterns.getPDLPatterns().registerRewriteFunction("convert_index_to_i32",
                                                    getI32TensorSizes);
}

LogicalResult populatePDLModuleFromFileName(MLIRContext *context,
                                            RewritePatternSet &patterns,
                                            llvm::StringRef pdlModuleFileName) {
  std::string errorMessage;
  auto memoryBuffer = mlir::openInputFile(pdlModuleFileName, &errorMessage);
  if (!memoryBuffer) {
    return emitError(FileLineColLoc::get(
               StringAttr::get(context, pdlModuleFileName), 0, 0))
           << "failed to open pattern module file: " << errorMessage;
  }
  // Tell sourceMgr about this buffer, the parser will pick it up.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
  PDLPatternModule pdlModule =
      OwningOpRef<ModuleOp>(parseSourceFile<ModuleOp>(sourceMgr, context));
  patterns.insert(std::move(pdlModule));
  return success();
}

} // namespace detail
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_UTILS_CUSTOMPATTERNAPPLICATORPASSBASE_H_
