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

#include "iree/compiler/Dialect/HAL/Conversion/IREEToHAL/ConvertIREEToHAL.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {
class DynamicShapeConstantOpConversion
    : public OpConversionPattern<IREE::DynamicShapeConstantOp> {
 public:
  using OpConversionPattern<IREE::DynamicShapeConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::DynamicShapeConstantOp constantOp,
      llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    assert(newOperands.empty() && "dynamic_shape_constant takes no operands");
    auto device =
        rewriter.createOrFold<IREE::HAL::ExSharedDeviceOp>(constantOp.getLoc());
    auto allocator = rewriter.createOrFold<IREE::HAL::DeviceAllocatorOp>(
        constantOp.getLoc(), device);

    // TODO(benvanik): compute from SSA use-def chain uses.
    IREE::HAL::MemoryTypeBitfield memoryTypes =
        IREE::HAL::MemoryTypeBitfield::DeviceLocal |
        IREE::HAL::MemoryTypeBitfield::HostVisible;
    IREE::HAL::BufferUsageBitfield bufferUsage =
        IREE::HAL::BufferUsageBitfield::All |
        IREE::HAL::BufferUsageBitfield::Constant;

    auto view = rewriter.createOrFold<IREE::HAL::BufferViewConstOp>(
        constantOp.getLoc(), allocator, memoryTypes, bufferUsage,
        constantOp.value());

    rewriter.replaceOpWithNewOp<IREE::DoNotOptimizeOp>(constantOp, view);
    return success();
  }
};

}  // namespace

// Appends all patterns for lowering IREE ops to HAL buffer ops.
void populateIREEToHALPatterns(MLIRContext *context,
                               OwningRewritePatternList &patterns) {
  patterns.insert<DynamicShapeConstantOpConversion>(context);
}

void setupIREEToHALLegality(MLIRContext *context, ConversionTarget &target) {
  target.addIllegalOp<IREE::DynamicShapeConstantOp>();
}

}  // namespace iree_compiler
}  // namespace mlir
