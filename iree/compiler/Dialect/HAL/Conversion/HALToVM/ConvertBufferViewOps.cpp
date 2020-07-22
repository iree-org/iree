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
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Converts the hal.buffer_view.dims op with variadic results to one of the
// non-variadic import functions (hal.buffer_view.dims.N). We include N=1-4 as
// that's the common case. If N>4 we use the N=4 import and then get the
// remaining dims with hal.buffer_view.dim as that's pretty rare.
class BufferViewDimsOpConversion
    : public OpConversionPattern<IREE::HAL::BufferViewDimsOp> {
 public:
  BufferViewDimsOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                             TypeConverter &typeConverter,
                             StringRef importNamePrefix)
      : OpConversionPattern(context) {
    importOp1 = importSymbols.lookup<IREE::VM::ImportOp>(
        (importNamePrefix + ".1").str());
    importOp2 = importSymbols.lookup<IREE::VM::ImportOp>(
        (importNamePrefix + ".2").str());
    importOp3 = importSymbols.lookup<IREE::VM::ImportOp>(
        (importNamePrefix + ".3").str());
    importOp4 = importSymbols.lookup<IREE::VM::ImportOp>(
        (importNamePrefix + ".4").str());
    assert(importOp1 && importOp2 && importOp3 && importOp4);
  }

  LogicalResult matchAndRewrite(
      IREE::HAL::BufferViewDimsOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    switch (op.getNumResults()) {
      case 1:
        rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
            op, rewriter.getSymbolRefAttr(importOp1),
            importOp1.getType().getResults(), operands);
        break;
      case 2:
        rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
            op, rewriter.getSymbolRefAttr(importOp2),
            importOp2.getType().getResults(), operands);
        break;
      case 3:
        rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
            op, rewriter.getSymbolRefAttr(importOp3),
            importOp3.getType().getResults(), operands);
        break;
      case 4:
        rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
            op, rewriter.getSymbolRefAttr(importOp4),
            importOp4.getType().getResults(), operands);
        break;
      default:
        SmallVector<Value, 8> dimensions =
            rewriter
                .create<IREE::VM::CallOp>(
                    op.getLoc(), rewriter.getSymbolRefAttr(importOp4),
                    importOp4.getType().getResults(), operands)
                .getResults();
        for (int i = 4; i < op.getNumResults(); ++i) {
          dimensions.push_back(
              rewriter.createOrFold<IREE::HAL::BufferViewDimOp>(
                  op.getLoc(), rewriter.getIndexType(), operands[0],
                  rewriter.getIntegerAttr(rewriter.getIndexType(), i)));
        }
        rewriter.replaceOp(op, dimensions);
        break;
    }
    return success();
  }

 private:
  mutable IREE::VM::ImportOp importOp1;
  mutable IREE::VM::ImportOp importOp2;
  mutable IREE::VM::ImportOp importOp3;
  mutable IREE::VM::ImportOp importOp4;
};

}  // namespace

void populateHALBufferViewToVMPatterns(MLIRContext *context,
                                       SymbolTable &importSymbols,
                                       TypeConverter &typeConverter,
                                       OwningRewritePatternList &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferViewCreateOp>>(
      context, importSymbols, typeConverter, "hal.buffer_view.create");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferViewSubviewOp>>(
      context, importSymbols, typeConverter, "hal.buffer_view.subview");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferViewBufferOp>>(
      context, importSymbols, typeConverter, "hal.buffer_view.buffer");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferViewByteLengthOp>>(
      context, importSymbols, typeConverter, "hal.buffer_view.byte_length");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferViewComputeOffsetOp>>(
      context, importSymbols, typeConverter, "hal.buffer_view.compute_offset");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferViewComputeRangeOp>>(
      context, importSymbols, typeConverter, "hal.buffer_view.compute_range");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferViewRankOp>>(
      context, importSymbols, typeConverter, "hal.buffer_view.rank");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferViewDimOp>>(
      context, importSymbols, typeConverter, "hal.buffer_view.dim");
  patterns.insert<BufferViewDimsOpConversion>(
      context, importSymbols, typeConverter, "hal.buffer_view.dims");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferViewTraceOp>>(
      context, importSymbols, typeConverter, "hal.buffer_view.trace");
}

}  // namespace iree_compiler
}  // namespace mlir
