// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/HAL/Loader/Conversion/StreamToHALLoader/Patterns.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Modules/HAL/Inline/IR/HALInlineOps.h"
#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderDialect.h"
#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

// Returns the !util.buffer from the given converted resource, which may be
// either a !util.buffer or an external !hal.buffer.
static Value getResourceBuffer(Location loc, Value resource,
                               OpBuilder &builder) {
  if (llvm::isa<IREE::HAL::BufferType>(resource.getType())) {
    // Get the storage of the buffer; the returned buffer is already a subspan.
    return builder.createOrFold<IREE::HAL::Inline::BufferStorageOp>(loc,
                                                                    resource);
  }
  return resource;
}

// Converts a dispatch command into an inline executable dispatch.
struct CmdDispatchOpPattern
    : public OpConversionPattern<IREE::Stream::CmdDispatchOp> {
  CmdDispatchOpPattern(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern(typeConverter, context, PatternBenefit(10000)) {}
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdDispatchOp dispatchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = dispatchOp.getLoc();

    // TODO(benvanik): support a lightweight switch builder for picking variants
    // that doesn't pull in the full HAL dialect. We could make the match
    // expressions take a callback that performs the query, for example.
    // For now we bail if there's multiple.
    auto entryPointAttrs = dispatchOp.getEntryPoints().getValue();
    if (entryPointAttrs.size() != 1) {
      return rewriter.notifyMatchFailure(dispatchOp,
                                         "multiple variant targets not yet "
                                         "supported in the inline HAL loader");
    }
    auto entryPointAttr = llvm::cast<SymbolRefAttr>(entryPointAttrs.front());

    // Get the handle to the executable that is compatible with our device.
    auto executableOp =
        cast<IREE::HAL::ExecutableOp>(SymbolTable::lookupNearestSymbolFrom(
            dispatchOp, entryPointAttr.getRootReference()));
    assert(executableOp && "dispatch target executable op not found");

    // For now we aren't doing loader support checks. We should, though.
    auto variantOps = executableOp.getOps<IREE::HAL::ExecutableVariantOp>();
    if (std::distance(variantOps.begin(), variantOps.end()) > 1) {
      return rewriter.notifyMatchFailure(dispatchOp,
                                         "only one variant is supported today");
    }

    // Lookup executable reference.
    auto lookupOp = rewriter.create<IREE::HAL::Loader::ExecutableLookupOp>(
        loc, rewriter.getType<IREE::HAL::ExecutableType>(),
        executableOp.getName());

    // TODO(benvanik): use scf.index_switch as with the full HAL.
    for (auto variantOp : variantOps) {
      auto exportOps = variantOp.getExportOps();
      auto exportIt =
          llvm::find_if(exportOps, [&](IREE::HAL::ExecutableExportOp op) {
            return op.getNameAttr() == entryPointAttr.getLeafReference();
          });
      if (exportIt == exportOps.end()) {
        return variantOp.emitError()
               << "hal.executable.variant is missing the entry point for "
               << entryPointAttr;
      }
      auto exportOp = *exportIt;
      dispatchVariant(dispatchOp, adaptor, executableOp, variantOp, exportOp,
                      lookupOp.getResult(), rewriter);
    }

    rewriter.eraseOp(dispatchOp);
    return success();
  }

  void dispatchVariant(IREE::Stream::CmdDispatchOp dispatchOp,
                       OpAdaptor adaptor, IREE::HAL::ExecutableOp executableOp,
                       IREE::HAL::ExecutableVariantOp variantOp,
                       IREE::HAL::ExecutableExportOp exportOp, Value executable,
                       OpBuilder &builder) const {
    auto loc = dispatchOp.getLoc();

    // Push constant values.
    // TODO(#5322): symbolic push constant names on the hal.interface so we can
    // sparsely pack these.
    SmallVector<Value> pushConstants;
    for (auto operand : adaptor.getUniformOperands()) {
      assert(operand.getType().isInteger(32) &&
             "expected only i32 values after iree-hal-pack-dispatch-operands");
      pushConstants.push_back(operand);
    }

    // Push descriptor bindings.
    SmallVector<Value> bindingBuffers;
    SmallVector<Value> bindingOffsets;
    SmallVector<Value> bindingLengths;
    for (unsigned i = 0; i < adaptor.getResources().size(); ++i) {
      auto buffer = getResourceBuffer(loc, adaptor.getResources()[i], builder);
      bindingBuffers.push_back(buffer);
      bindingOffsets.push_back(adaptor.getResourceOffsets()[i]);
      bindingLengths.push_back(adaptor.getResourceLengths()[i]);
    }

    // Dispatch with a target-specific workgroup count.
    auto exportSymRef =
        SymbolRefAttr::get(builder.getContext(), executableOp.getName(),
                           {SymbolRefAttr::get(exportOp->getParentOp()),
                            SymbolRefAttr::get(exportOp)});
    auto workgroupCount = exportOp.calculateWorkgroupCount(
        loc, /*device=*/nullptr, adaptor.getWorkload(), builder);
    builder.create<IREE::HAL::Loader::ExecutableDispatchSymbolOp>(
        loc, executable, exportSymRef, workgroupCount[0], workgroupCount[1],
        workgroupCount[2], pushConstants, bindingBuffers, bindingOffsets,
        bindingLengths);
  }
};

} // namespace

void populateStreamToHALLoaderPatterns(MLIRContext *context,
                                       ConversionTarget &conversionTarget,
                                       TypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  // Executables are taken care of after serialization by the
  // MaterializeExecutables pass. We allow them to pass through for now.
  conversionTarget.addLegalOp<IREE::HAL::ExecutableOp>();
  conversionTarget.markOpRecursivelyLegal<IREE::HAL::ExecutableOp>();

  patterns.insert<CmdDispatchOpPattern>(typeConverter, context);
}

} // namespace mlir::iree_compiler
