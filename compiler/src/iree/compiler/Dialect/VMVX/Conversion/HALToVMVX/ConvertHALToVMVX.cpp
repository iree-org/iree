// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VMVX/Conversion/HALToVMVX/ConvertHALToVMVX.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXOps.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXTypes.h"
#include "iree/compiler/Utils/IndexSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// Ordered indices of arguments to the entry point function.
// This is what the VM will receive at runtime from the HAL.
enum EntryArgOrdinals {
  kEntryArgLocalMemory,
  kEntryArgConstants,
  kEntryArgBindings,
  kEntryArgWorkgroupX,
  kEntryArgWorkgroupY,
  kEntryArgWorkgroupZ,
  kEntryArgWorkgroupSizeX,
  kEntryArgWorkgroupSizeY,
  kEntryArgWorkgroupSizeZ,
  kEntryArgWorkgroupCountX,
  kEntryArgWorkgroupCountY,
  kEntryArgWorkgroupCountZ,
};

/// Rewrites entry functions to have a vmvx.interface, local memory, and an XYZ
/// workgroup ID. The runtime will provide these values during invocation.
///
/// Source:
///   func.func @entry()
///
/// Target:
///   func.func @entry(
///       %local_memory: !vmvx.buffer,
///       %constants: !vmvx.buffer,
///       %bindings: !util.list<!vmvx.buffer>,
///       %workgroup_id_x: i32,
///       %workgroup_id_y: i32,
///       %workgroup_id_z: i32,
///       %workgroup_size_x: i32,
///       %workgroup_size_y: i32,
///       %workgroup_size_z: i32,
///       %workgroup_count_x: i32,
///       %workgroup_count_y: i32,
///       %workgroup_count_z: i32
///   )
LogicalResult updateHALToVMVXEntryFuncOp(func::FuncOp funcOp,
                                         TypeConverter &typeConverter) {
  auto originalType = funcOp.getFunctionType();
  if (originalType.getNumInputs() != 0 || originalType.getNumResults() != 0) {
    return funcOp.emitError() << "exported functions must have no I/O";
  }

  auto bufferType = IREE::Util::BufferType::get(funcOp.getContext());
  auto bindingsType = IREE::Util::ListType::get(bufferType); // of i8
  auto i32Type = IntegerType::get(funcOp.getContext(), 32);
  auto newType = FunctionType::get(funcOp.getContext(),
                                   {
                                       /*local_memory=*/bufferType, // of i8
                                       /*constants=*/bufferType,    // of i32
                                       /*bindings=*/bindingsType,
                                       /*workgroup_id_x=*/i32Type,
                                       /*workgroup_id_y=*/i32Type,
                                       /*workgroup_id_z=*/i32Type,
                                       /*workgroup_size_x=*/i32Type,
                                       /*workgroup_size_y=*/i32Type,
                                       /*workgroup_size_z=*/i32Type,
                                       /*workgroup_count_x=*/i32Type,
                                       /*workgroup_count_y=*/i32Type,
                                       /*workgroup_count_z=*/i32Type,
                                   },
                                   {});

  funcOp.setType(newType);
  SmallVector<Location> locs(newType.getNumInputs(), funcOp.getLoc());
  funcOp.front().addArguments(newType.getInputs(), locs);

  return success();
}

namespace {

/// Rewrites hal.interface.workgroup.id to use the arguments injected onto the
/// function.
struct ConvertHALInterfaceWorkgroupIDOp
    : public OpConversionPattern<IREE::HAL::InterfaceWorkgroupIDOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceWorkgroupIDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    uint64_t dim = op.getDimension().getZExtValue();
    if (dim >= 3) {
      return op.emitOpError() << "out of bounds workgroup ID dimension";
    }

    // Get the argument to the function corresponding to the workgroup dim.
    Value workgroupDimI32 =
        op->getParentOfType<mlir::func::FuncOp>().getArgument(
            kEntryArgWorkgroupX + dim);
    Value workgroupDim = rewriter.create<arith::IndexCastOp>(
        op.getLoc(), rewriter.getIndexType(), workgroupDimI32);
    rewriter.replaceOp(op, workgroupDim);
    return success();
  }
};

/// Rewrites hal.interface.workgroup.size to use the arguments injected onto the
/// function.
struct ConvertHALInterfaceWorkgroupSizeOp
    : public OpConversionPattern<IREE::HAL::InterfaceWorkgroupSizeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceWorkgroupSizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    uint64_t dim = op.getDimension().getZExtValue();
    if (dim >= 3) {
      return op.emitOpError() << "out of bounds workgroup size dimension";
    }

    // Get the argument to the function corresponding to the workgroup dim.
    Value workgroupDimI32 =
        op->getParentOfType<mlir::func::FuncOp>().getArgument(
            kEntryArgWorkgroupSizeX + dim);
    Value workgroupDim = rewriter.create<arith::IndexCastOp>(
        op.getLoc(), rewriter.getIndexType(), workgroupDimI32);
    rewriter.replaceOp(op, workgroupDim);
    return success();
  }
};

/// Rewrites hal.interface.workgroup.count to use the arguments injected onto
/// the function.
struct ConvertHALInterfaceWorkgroupCountOp
    : public OpConversionPattern<IREE::HAL::InterfaceWorkgroupCountOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceWorkgroupCountOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    uint64_t dim = op.getDimension().getZExtValue();
    if (dim >= 3) {
      return op.emitOpError() << "out of bounds workgroup count dimension";
    }

    // Get the argument to the function corresponding to the workgroup dim.
    Value workgroupDimI32 =
        op->getParentOfType<mlir::func::FuncOp>().getArgument(
            kEntryArgWorkgroupCountX + dim);
    Value workgroupDim = rewriter.create<arith::IndexCastOp>(
        op.getLoc(), rewriter.getIndexType(), workgroupDimI32);
    rewriter.replaceOp(op, workgroupDim);
    return success();
  }
};

/// Rewrites hal.interface.constant.load to ops loading from the ABI structs.
struct ConvertHALInterfaceConstantLoadOp
    : public OpConversionPattern<IREE::HAL::InterfaceConstantLoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceConstantLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Find the vmvx.interface argument to the function.
    auto constantsArg = op->getParentOfType<mlir::func::FuncOp>().getArgument(
        kEntryArgConstants);
    assert(constantsArg && "entry point not conforming to requirements");
    // HACK: we could find the total push constant count and avoid this size op
    // but it'd require walking all the way up to the hal.executable export.
    auto constantsSize =
        rewriter.create<IREE::Util::BufferSizeOp>(op.getLoc(), constantsArg);
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());

    // Index -> byte offset.
    auto constantIndex = rewriter.createOrFold<arith::ConstantIndexOp>(
        op.getLoc(), op.getIndex().getZExtValue());
    auto elementSize =
        rewriter.createOrFold<IREE::Util::SizeOfOp>(op.getLoc(), resultType);
    auto byteOffset = rewriter.createOrFold<arith::MulIOp>(
        op.getLoc(), elementSize, constantIndex);
    rewriter.replaceOpWithNewOp<IREE::Util::BufferLoadOp>(
        op, resultType, constantsArg, constantsSize, byteOffset, elementSize);
    return success();
  }
};

struct ConvertGetRawInterfaceBindingBufferOp
    : public OpConversionPattern<IREE::VMVX::GetRawInterfaceBindingBufferOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::VMVX::GetRawInterfaceBindingBufferOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Find the vmvx.interface argument to the function.
    auto bindingsArg = op->getParentOfType<mlir::func::FuncOp>().getArgument(
        kEntryArgBindings);
    assert(bindingsArg && bindingsArg.getType().isa<IREE::Util::ListType>() &&
           "entry point not conforming to requirements");

    // TODO(benvanik): compact the indices - the bindings we have on the ABI
    // interface are dense.
    if (op.getSet().getZExtValue() != 0) {
      return op.emitOpError() << "sparse binding sets not yet implemented";
    }

    IndexSet indexSet(op.getLoc(), rewriter);
    auto bindingType = llvm::cast<IREE::Util::ListType>(bindingsArg.getType())
                           .getElementType();
    rewriter
        .replaceOpWithNewOp<IREE::Util::ListGetOp>(
            op, bindingType, bindingsArg,
            rewriter.createOrFold<arith::ConstantIndexOp>(
                op.getLoc(), op.getBinding().getZExtValue()))
        .getResult();
    return success();
  }
};

/// Rewrites hal.interface.binding.subspan to ops loading from the ABI structs.
struct ConvertHALInterfaceBindingSubspanOp
    : public OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Find the vmvx.interface argument to the function.
    auto bindingsArg = op->getParentOfType<mlir::func::FuncOp>().getArgument(
        kEntryArgBindings);
    assert(bindingsArg && bindingsArg.getType().isa<IREE::Util::ListType>() &&
           "entry point not conforming to requirements");

    // TODO(benvanik): compact the indices - the bindings we have on the ABI
    // interface are dense.
    if (op.getSet().getZExtValue() != 0) {
      return op.emitOpError() << "sparse binding sets not yet implemented";
    }

    IndexSet indexSet(op.getLoc(), rewriter);
    auto bindingType = llvm::cast<IREE::Util::ListType>(bindingsArg.getType())
                           .getElementType();
    auto sourceBuffer =
        rewriter
            .create<IREE::Util::ListGetOp>(
                op.getLoc(), bindingType, bindingsArg,
                rewriter.createOrFold<arith::ConstantIndexOp>(
                    op.getLoc(), op.getBinding().getZExtValue()))
            .getResult();

    if (op.getByteOffset() && !matchPattern(op.getByteOffset(), m_Zero())) {
      // Offsetted binding: replace with a BufferSubspanOp.
      Value sourceSize = rewriter.createOrFold<IREE::Util::BufferSizeOp>(
          op.getLoc(), sourceBuffer);

      // Compute the dest size by multiplying the element size by all extents
      // (static and dynamic).
      auto memRefType = llvm::cast<MemRefType>(op.getResult().getType());
      Value destSize = rewriter.createOrFold<IREE::Util::SizeOfOp>(
          op.getLoc(), memRefType.getElementType());
      auto dynamicExtentIt = adaptor.getDynamicDims().begin();
      for (int i = 0; i < memRefType.getRank(); ++i) {
        Value extent;
        if (memRefType.isDynamicDim(i)) {
          extent = *dynamicExtentIt;
          dynamicExtentIt++;
        } else {
          extent = indexSet.get(memRefType.getDimSize(i));
        }
        destSize =
            rewriter.createOrFold<arith::MulIOp>(op.getLoc(), destSize, extent);
      }

      rewriter.replaceOpWithNewOp<IREE::Util::BufferSubspanOp>(
          op, sourceBuffer, sourceSize, adaptor.getByteOffset(), destSize);
    } else {
      // Zero offset. Just return the source buffer.
      rewriter.replaceOp(op, sourceBuffer);
    }
    return success();
  }
};

} // namespace

void populateHALToVMVXPatterns(MLIRContext *context,
                               ConversionTarget &conversionTarget,
                               RewritePatternSet &patterns,
                               TypeConverter &typeConverter) {
  conversionTarget.addIllegalDialect<IREE::HAL::HALDialect>();
  conversionTarget.addIllegalOp<IREE::VMVX::GetRawInterfaceBindingBufferOp>();

  patterns.insert<ConvertGetRawInterfaceBindingBufferOp>(typeConverter,
                                                         context);
  patterns.insert<ConvertHALInterfaceWorkgroupIDOp>(typeConverter, context);
  patterns.insert<ConvertHALInterfaceWorkgroupSizeOp>(typeConverter, context);
  patterns.insert<ConvertHALInterfaceWorkgroupCountOp>(typeConverter, context);
  patterns.insert<ConvertHALInterfaceConstantLoadOp>(typeConverter, context);
  patterns.insert<ConvertHALInterfaceBindingSubspanOp>(typeConverter, context);
}

} // namespace mlir::iree_compiler
