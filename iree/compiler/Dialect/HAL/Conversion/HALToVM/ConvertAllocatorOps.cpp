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

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class AllocatorAllocateConstOpConversion
    : public OpConversionPattern<IREE::HAL::AllocatorAllocateConstOp> {
 public:
  AllocatorAllocateConstOpConversion(MLIRContext *context,
                                     SymbolTable &importSymbols,
                                     TypeConverter &typeConverter,
                                     StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult matchAndRewrite(
      IREE::HAL::AllocatorAllocateConstOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Encode constant data into a rodata segment. These will eventually get
    // deduped and combined.
    auto ip = rewriter.saveInsertionPoint();
    auto parentFuncOp = op.getParentOfType<IREE::VM::FuncOp>();
    rewriter.setInsertionPoint(parentFuncOp);
    auto constName = (parentFuncOp.getName() + "_const_" +
                      std::to_string(allocateUniqueId(parentFuncOp)))
                         .str();
    auto rodataOp =
        rewriter.create<IREE::VM::RodataOp>(op.getLoc(), constName, op.value());
    rewriter.restoreInsertionPoint(ip);
    auto loadRodataOp =
        rewriter.create<IREE::VM::ConstRefRodataOp>(op.getLoc(), rodataOp);

    IREE::HAL::AllocatorAllocateConstOp::Adaptor opAdaptor(operands);
    auto shape = IREE::HAL::getStaticShapeDims(op.getLoc(),
                                               op.value().getType(), rewriter);
    SmallVector<Value, 8> callOperands = {
        opAdaptor.allocator(),
        rewriter.create<mlir::ConstantOp>(
            op.getLoc(), rewriter.getI32IntegerAttr(
                             static_cast<int32_t>(op.memory_types()))),
        rewriter.create<mlir::ConstantOp>(
            op.getLoc(), rewriter.getI32IntegerAttr(
                             static_cast<int32_t>(op.buffer_usage()))),
    };
    callOperands.append(shape.begin(), shape.end());
    callOperands.push_back(rewriter.create<mlir::ConstantOp>(
        op.getLoc(),
        IREE::HAL::getElementTypeAttr(op.value().getType().getElementType())));
    callOperands.push_back(loadRodataOp.getResult());
    SmallVector<int16_t, 6> segmentSizes = {
        /*allocator=*/-1,
        /*memory_types=*/-1,
        /*buffer_usage=*/-1,
        /*shape=*/static_cast<int16_t>(shape.size()),
        /*element_type=*/-1,
        /*value=*/-1,
    };

    auto importType = importOp.getType();
    rewriter.replaceOpWithNewOp<IREE::VM::CallVariadicOp>(
        op, rewriter.getSymbolRefAttr(importOp), importType.getResults(),
        segmentSizes, importType.getInputs(), callOperands);
    return success();
  }

 private:
  // TODO(b/145839814): find a name that's unique or make the rewriter support
  // assigning unique names.
  int allocateUniqueId(Operation *context) const {
    if (uniqueContext != context) {
      uniqueContext = context;
      uniqueCounter = 0;
    }
    return uniqueCounter++;
  }
  mutable Operation *uniqueContext = nullptr;
  mutable int uniqueCounter = 0;

  mutable IREE::VM::ImportOp importOp;
};

}  // namespace

void populateHALAllocatorToVMPatterns(MLIRContext *context,
                                      SymbolTable &importSymbols,
                                      TypeConverter &typeConverter,
                                      OwningRewritePatternList &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::AllocatorComputeSizeOp>>(
      context, importSymbols, typeConverter, "hal.allocator.compute_size");
  patterns.insert<VMImportOpConversion<IREE::HAL::AllocatorComputeOffsetOp>>(
      context, importSymbols, typeConverter, "hal.allocator.compute_offset");
  patterns.insert<VMImportOpConversion<IREE::HAL::AllocatorComputeRangeOp>>(
      context, importSymbols, typeConverter, "hal.allocator.compute_range");
  patterns.insert<VMImportOpConversion<IREE::HAL::AllocatorAllocateOp>>(
      context, importSymbols, typeConverter, "hal.allocator.allocate");
  patterns.insert<AllocatorAllocateConstOpConversion>(
      context, importSymbols, typeConverter, "hal.allocator.allocate.const");
}

}  // namespace iree_compiler
}  // namespace mlir
