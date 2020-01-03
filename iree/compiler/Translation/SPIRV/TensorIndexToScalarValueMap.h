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

//===- TensorIndexToScalarValueMap.h ----------------------------*- C++ -*-===//
//
// Maintains mapping from Tensor value at an index to the scalar value during
// SPIR-V lowering.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_TRANSLATION_SPIRV_TENSORINDEXTOSCALARVALUEMAP_H
#define IREE_COMPILER_TRANSLATION_SPIRV_TENSORINDEXTOSCALARVALUEMAP_H

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {

class TensorIndexToScalarValueMap {
 public:
  explicit TensorIndexToScalarValueMap(MLIRContext *context)
      : affineExprCodegen(context) {}

  /// Returns the buffer associated with an argument in the dispatch function.
  Value getBufferForArgument(Value arg) { return argToBufferMap.lookup(arg); }

  /// Returns the value in the lowered function that represents the scalar value
  /// of the `tensor` in the original function at a given `index`
  Value getValueAtIndex(Value tensor, AffineMap index) {
    auto tensorIt = tensorIndexToScalarValueMap.find(tensor);
    if (tensorIt == tensorIndexToScalarValueMap.end()) {
      return nullptr;
    }
    auto scalarValIt = tensorIt->second.find(index);
    if (scalarValIt == tensorIt->second.end()) {
      return nullptr;
    }
    return scalarValIt->second;
  }

  /// Returns the value in the lowered function (or generates it if it hasn't
  /// been generated already), the scalar value that stores the value
  /// corresponding to `expr`
  Value getAffineExprValue(OpBuilder::InsertPoint ip, Location loc,
                           AffineExpr expr) {
    return affineExprCodegen.getExprValue(ip, loc, expr);
  }

  /// Records the `buffer` to use for an `argument` in the dispatch function.
  void setBufferForArgument(Value argument, Value buffer) {
    argToBufferMap[argument] = buffer;
  }

  /// Records the `scalar` value in the lowered function that represents the
  /// value of the `tensor` in the original function at a given `index`
  void setValueAtIndex(Value tensor, AffineMap index, Value scalar) {
    tensorIndexToScalarValueMap[tensor][index] = scalar;
  }

  /// Records the `value` to use for an AffineDimExpr while generating code for
  /// AffineExpr trees.
  void setDimValue(unsigned dim, Value value) {
    return affineExprCodegen.setDimValue(dim, value);
  }

  /// Records the `value` to use for an AffineSymbolExpr while generating code
  /// for AffineExpr trees.
  void setSymbolValue(unsigned pos, Value value) {
    return affineExprCodegen.setSymbolValue(pos, value);
  }

 private:
  /// Class to walk AffineExpr trees and generate scalar code that evaluates the
  /// tree.
  class AffineExprCodegen : public AffineExprVisitor<AffineExprCodegen, Value> {
   public:
    explicit AffineExprCodegen(MLIRContext *context)
        : builder(context), location(UnknownLoc::get(context)) {}
    Value getExprValue(OpBuilder::InsertPoint ip, Location loc,
                       AffineExpr expr) {
      Value &val = affineExprToValueMap[expr];
      if (!val) {
        builder.restoreInsertionPoint(ip);
        location = loc;
        val = visit(expr);
      }
      return val;
    }
    Value visitAddExpr(AffineBinaryOpExpr expr) {
      auto operand1 = getValue(expr.getLHS());
      auto operand2 = getValue(expr.getRHS());
      return builder.create<spirv::IAddOp>(location, operand1, operand2);
    }
    Value visitCeilDivExpr(AffineBinaryOpExpr expr) {
      // TODO(ravishankarm): Implement ceil div expr codegen.
      llvm_unreachable("Unimplemented affine AffineCeilDivExpr codegen");
      return nullptr;
    }
    Value visitConstantExpr(AffineConstantExpr expr) {
      return builder.create<spirv::ConstantOp>(
          location, builder.getIntegerType(32),
          builder.getI32IntegerAttr(expr.getValue()));
    }
    Value visitDimExpr(AffineDimExpr expr) {
      return threadDimToValue.lookup(expr.getPosition());
    }
    Value visitFloorDivExpr(AffineBinaryOpExpr expr) {
      auto operand1 = getValue(expr.getLHS());
      auto operand2 = getValue(expr.getRHS());
      return builder.create<spirv::SDivOp>(location, operand1, operand2);
    }
    Value visitModExpr(AffineBinaryOpExpr expr) {
      auto operand1 = getValue(expr.getLHS());
      auto operand2 = getValue(expr.getRHS());
      return builder.create<spirv::SModOp>(location, operand1, operand2);
    }
    Value visitMulExpr(AffineBinaryOpExpr expr) {
      auto operand1 = getValue(expr.getLHS());
      auto operand2 = getValue(expr.getRHS());
      return builder.create<spirv::IMulOp>(location, operand1, operand2);
    }
    Value visitSymbolExpr(AffineSymbolExpr expr) {
      return symbolPosToValue.lookup(expr.getPosition());
    }
    void setDimValue(unsigned dim, Value value) {
      threadDimToValue[dim] = value;
    }
    void setSymbolValue(unsigned pos, Value value) {
      assert(!symbolPosToValue.count(pos));
      symbolPosToValue[pos] = value;
    }

   private:
    Value getValue(AffineExpr expr) {
      Value &val = affineExprToValueMap[expr];
      if (!val) {
        val = visit(expr);
      }
      return val;
    }

    /// Cache of AffineExpr to scalar Value to avoid regeneration.
    DenseMap<AffineExpr, Value> affineExprToValueMap;

    /// Mapping from AffineDimExpr to Value to use in generating scalar code to
    /// compute an AffineExpr tree.
    DenseMap<unsigned, Value> threadDimToValue;

    /// Mapping from AffineSymbolExpr to Value to use in generating scalar code
    /// to compute an AffineExpr tree.
    DenseMap<unsigned, Value> symbolPosToValue;

    OpBuilder builder;
    Location location;
  };

  AffineExprCodegen affineExprCodegen;
  DenseMap<Value, DenseMap<AffineMap, Value>> tensorIndexToScalarValueMap;
  DenseMap<Value, Value> argToBufferMap;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_TENSORINDEXTOSCALARVALUEMAP_H
