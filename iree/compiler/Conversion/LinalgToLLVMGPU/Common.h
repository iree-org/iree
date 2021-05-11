// Copyright 2021 Nod Labs
// Copyright 2021 Google LLC
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

#ifndef IREE_COMPILER_CONVERSION_LINALGTOLLVMGPU_COMMON_H_
#define IREE_COMPILER_CONVERSION_LINALGTOLLVMGPU_COMMON_H_

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

class ConvertFunc : public ConvertToLLVMPattern {
 public:
  explicit ConvertFunc(MLIRContext *context, LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(mlir::FuncOp::getOperationName(), context,
                             converter, 100) {}
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

class ConvertIREEBindingOp : public ConvertToLLVMPattern {
 public:
  explicit ConvertIREEBindingOp(MLIRContext *context,
                                LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(
            IREE::HAL::InterfaceBindingSubspanOp::getOperationName(), context,
            converter) {}
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

/// A pattern to convert hal.interface.workgroup.id/count/size into
/// corresponding NVVM/ROCDL ops.
template <typename InterfaceOpTy, typename XOp, typename YOp, typename ZOp>
struct HALInterfaceWorkgroupOpsConverter final
    : public OpConversionPattern<InterfaceOpTy> {
  using OpConversionPattern<InterfaceOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      InterfaceOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type i32Type = rewriter.getI32Type();
    Value newOp;
    int32_t index = static_cast<int32_t>(op.dimension().getSExtValue());
    switch (index) {
      case 0:
        newOp = rewriter.create<XOp>(loc, i32Type);
        break;
      case 1:
        newOp = rewriter.create<YOp>(loc, i32Type);
        break;
      case 2:
        newOp = rewriter.create<ZOp>(loc, i32Type);
        break;
      default:
        return failure();
    }

    newOp =
        rewriter.create<LLVM::SExtOp>(loc, rewriter.getIntegerType(64), newOp);
    rewriter.replaceOp(op, {newOp});
    return success();
  }
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_LINALGTOLLVMGPU_COMMON_H_
