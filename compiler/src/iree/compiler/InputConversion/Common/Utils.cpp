// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/Utils.h"

#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

namespace mlir {
namespace iree_compiler {

// Reduce the input value along the reduction dimensions
Value sumReduceDimensionSubset(ImplicitLocOpBuilder &rewriter, Value val,
                               Type accETy, ArrayRef<bool> is_reduction) {
  auto context = val.getContext();
  RankedTensorType ty = val.getType().cast<RankedTensorType>();

  llvm::SmallVector<int64_t> staticSizes;
  SmallVector<Value> dynSizes;
  for (int i = 0, s = is_reduction.size(); i < s; i++) {
    if (is_reduction[i]) continue;

    staticSizes.push_back(ty.getDimSize(i));
    if (ty.isDynamicDim(i)) {
      dynSizes.push_back(rewriter.create<tensor::DimOp>(val, i));
    }
  }

  // Create a zero-filled accumulator.
  Value initAcc =
      rewriter.create<tensor::EmptyOp>(staticSizes, accETy, dynSizes);
  Value zeroInt = rewriter.create<arith::ConstantIntOp>(0, accETy).getResult();
  Value zeroAcc =
      rewriter.create<linalg::FillOp>(zeroInt, initAcc).getResult(0);

  SmallVector<AffineExpr> filterExprs(ty.getRank());
  SmallVector<AffineExpr> outputExprs;
  SmallVector<utils::IteratorType> iterators;

  for (int i = 0, s = ty.getRank(); i < s; i++) {
    if (!is_reduction[i]) {
      auto expr = rewriter.getAffineDimExpr(iterators.size());
      iterators.push_back(utils::IteratorType::parallel);

      outputExprs.push_back(expr);
      filterExprs[i] = expr;
    }
  }

  for (int i = 0, s = filterExprs.size(); i < s; i++) {
    if (!filterExprs[i]) {
      auto expr = mlir::getAffineDimExpr(iterators.size(), context);
      iterators.push_back(utils::IteratorType::reduction);
      filterExprs[i] = expr;
    }
  }

  SmallVector<AffineMap> affineMaps{
      AffineMap::get(ty.getRank(), 0, filterExprs, context),
      AffineMap::get(ty.getRank(), 0, outputExprs, context)};

  return rewriter
      .create<linalg::GenericOp>(
          zeroAcc.getType(), ValueRange{val}, ValueRange{zeroAcc}, affineMaps,
          iterators,
          [=](OpBuilder &b, Location loc, ValueRange args) {
            Value ext = b.create<arith::ExtSIOp>(loc, accETy, args[0]);
            Value sum = b.create<arith::AddIOp>(loc, ext, args[1]);
            b.create<linalg::YieldOp>(loc, sum);
          })
      .getResult(0);
}

}  // namespace iree_compiler
}  // namespace mlir
