// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Modules/VMVX/Conversion/HALToVMVX/ConvertHALToVMVX.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXOps.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

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
///   func @entry()
///
/// Target:
///   func @entry(
///       %local_memory: !vmvx.buffer,
///       %constants: !vmvx.buffer,
///       %bindings: !iree.list<!vmvx.buffer>,
///       %workgroup_x: index,
///       %workgroup_y: index,
///       %workgroup_z: index,
///       %workgroup_size_x: index,
///       %workgroup_size_y: index,
///       %workgroup_size_z: index,
///       %workgroup_count_x: index,
///       %workgroup_count_y: index,
///       %workgroup_count_z: index
///   )
LogicalResult updateHALToVMVXEntryFuncOp(FuncOp funcOp,
                                         TypeConverter &typeConverter) {
  auto originalType = funcOp.getType();
  if (originalType.getNumInputs() != 0 || originalType.getNumResults() != 0) {
    return funcOp.emitError() << "exported functions must have no I/O";
  }

  auto i8Type = IntegerType::get(funcOp.getContext(), 8);
  auto i32Type = IntegerType::get(funcOp.getContext(), 32);
  auto memRefI8Type = MemRefType::get({-1}, i8Type);
  auto memRefI32Type = MemRefType::get({-1}, i32Type);
  auto bindingsType = IREE::ListType::get(memRefI8Type);
  auto indexType = IndexType::get(funcOp.getContext());
  auto newType = FunctionType::get(funcOp.getContext(),
                                   {
                                       /*local_memory=*/memRefI8Type,
                                       /*constants=*/memRefI32Type,
                                       /*bindings=*/bindingsType,
                                       /*workgroup_x=*/indexType,
                                       /*workgroup_y=*/indexType,
                                       /*workgroup_z=*/indexType,
                                       /*workgroup_size_x=*/indexType,
                                       /*workgroup_size_y=*/indexType,
                                       /*workgroup_size_z=*/indexType,
                                       /*workgroup_count_x=*/indexType,
                                       /*workgroup_count_y=*/indexType,
                                       /*workgroup_count_z=*/indexType,
                                   },
                                   {});

  funcOp.setType(newType);
  funcOp.front().addArguments(newType.getInputs());

  return success();
}

namespace {

/// Rewrites hal.interface.workgroup.id to use the arguments injected onto the
/// function.
class ConvertHALInterfaceWorkgroupIDOp
    : public OpConversionPattern<IREE::HAL::InterfaceWorkgroupIDOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceWorkgroupIDOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    uint64_t dim = op.dimension().getZExtValue();
    if (dim >= 3) {
      return op.emitOpError() << "out of bounds workgroup ID dimension";
    }

    // Get the argument to the function corresponding to the workgroup dim.
    auto workgroupDim = op->getParentOfType<mlir::FuncOp>().getArgument(
        kEntryArgWorkgroupX + dim);
    rewriter.replaceOp(op, workgroupDim);
    return success();
  }
};

/// Rewrites hal.interface.workgroup.size to use the arguments injected onto the
/// function.
class ConvertHALInterfaceWorkgroupSizeOp
    : public OpConversionPattern<IREE::HAL::InterfaceWorkgroupSizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceWorkgroupSizeOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    uint64_t dim = op.dimension().getZExtValue();
    if (dim >= 3) {
      return op.emitOpError() << "out of bounds workgroup size dimension";
    }

    // Get the argument to the function corresponding to the workgroup dim.
    auto workgroupDim = op->getParentOfType<mlir::FuncOp>().getArgument(
        kEntryArgWorkgroupSizeX + dim);
    rewriter.replaceOp(op, workgroupDim);
    return success();
  }
};

/// Rewrites hal.interface.workgroup.count to use the arguments injected onto
/// the function.
class ConvertHALInterfaceWorkgroupCountOp
    : public OpConversionPattern<IREE::HAL::InterfaceWorkgroupCountOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceWorkgroupCountOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    uint64_t dim = op.dimension().getZExtValue();
    if (dim >= 3) {
      return op.emitOpError() << "out of bounds workgroup count dimension";
    }

    // Get the argument to the function corresponding to the workgroup dim.
    auto workgroupDim = op->getParentOfType<mlir::FuncOp>().getArgument(
        kEntryArgWorkgroupCountX + dim);
    rewriter.replaceOp(op, workgroupDim);
    return success();
  }
};

/// Rewrites hal.interface.load.constant to ops loading from the ABI structs.
class ConvertHALInterfaceLoadConstantOp
    : public OpConversionPattern<IREE::HAL::InterfaceLoadConstantOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceLoadConstantOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Find the vmvx.interface argument to the function.
    auto constantsArg =
        op->getParentOfType<mlir::FuncOp>().getArgument(kEntryArgConstants);
    assert(constantsArg && "entry point not conforming to requirements");
    auto constantType =
        constantsArg.getType().cast<MemRefType>().getElementType();

    auto resultType = getTypeConverter()->convertType(op.result().getType());

    auto constantOrdinal = rewriter.createOrFold<ConstantIndexOp>(
        op.getLoc(), op.offset().getZExtValue());
    auto loadedValue = rewriter.createOrFold<memref::LoadOp>(
        op.getLoc(), constantType, constantsArg, ValueRange{constantOrdinal});
    rewriter.replaceOpWithNewOp<IndexCastOp>(op, loadedValue, resultType);
    return success();
  }
};

/// Rewrites hal.interface.binding.subspan to ops loading from the ABI structs.
class ConvertHALInterfaceBindingSubspanOp
    : public OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceBindingSubspanOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // NOTE: we expect FoldSubViewOps to have run and zeroed out the offset.
    APInt byteOffset;
    if (!matchPattern(op.byte_offset(), m_ConstantInt(&byteOffset))) {
      return op.emitOpError()
             << "FoldSubViewOps must have ran prior to conversion";
    }

    // Find the vmvx.interface argument to the function.
    auto bindingsArg =
        op->getParentOfType<mlir::FuncOp>().getArgument(kEntryArgBindings);
    assert(bindingsArg && bindingsArg.getType().isa<IREE::ListType>() &&
           "entry point not conforming to requirements");

    // Lookup the source interface binding.
    auto interfaceBindingOp = op.queryBindingOp();

    // TODO(benvanik): compact the indices - the bindings we have on the ABI
    // interface are dense.
    if (interfaceBindingOp.set().getZExtValue() != 0) {
      return op.emitOpError() << "sparse binding sets not yet implemented";
    }

    auto bindingType =
        bindingsArg.getType().cast<IREE::ListType>().getElementType();
    auto getOp = rewriter.create<IREE::ListGetOp>(
        op.getLoc(), bindingType, bindingsArg,
        rewriter.createOrFold<ConstantIndexOp>(
            op.getLoc(), interfaceBindingOp.binding().getZExtValue()));
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op,
        getTypeConverter()
            ->convertType(op.result().getType())
            .cast<MemRefType>(),
        getOp.result());
    return success();
  }
};

/// Removes the hal.interface from the IR - it's not used after conversion.
class RemoveHALInterfaceOpPattern
    : public OpConversionPattern<IREE::HAL::InterfaceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void populateHALToVMVXPatterns(MLIRContext *context,
                               OwningRewritePatternList &patterns,
                               TypeConverter &typeConverter) {
  patterns.insert<ConvertHALInterfaceWorkgroupIDOp>(typeConverter, context);
  patterns.insert<ConvertHALInterfaceWorkgroupSizeOp>(typeConverter, context);
  patterns.insert<ConvertHALInterfaceWorkgroupCountOp>(typeConverter, context);
  patterns.insert<ConvertHALInterfaceLoadConstantOp>(typeConverter, context);
  patterns.insert<ConvertHALInterfaceBindingSubspanOp>(typeConverter, context);
  patterns.insert<RemoveHALInterfaceOpPattern>(typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
