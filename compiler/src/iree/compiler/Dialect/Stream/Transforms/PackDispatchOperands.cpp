// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-pack-dispatch-operands"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_PACKDISPATCHOPERANDSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Type conversion/expansion
//===----------------------------------------------------------------------===//
//
// TODO(#8037): ┏༼ ◉╭╮◉༽┓ this "packing" is bad and needs to be reworked.
//
// This could be dramatically improved and a benefit of the HAL interface being
// an implementation detail is that we can do that at any time without runtime
// changes. This currently optimizes for correctness for all types and not for
// optimal layout for minimizing storage requirements (push constants are a
// very finite resource - think <= 32 total i32 values) or access performance
// (loading 64-bit values are not aligned and may require split loads).
//
// If we wanted to support an open type conversion process we could have a
// method on HALConversionDialectInterface for getting a TypeConverter (or
// something). For now our verifiers on ops earlier in the compiler don't allow
// anything but int-or-float-or-index so it's not possible.

// TODO(benvanik): handle known zero/unused values; the stream annotations will
// tell us the potential values for each operand (when knowable) and we could
// for example take an i64 with values [0, 1, 2] and turn that into an i8.
// There may be another round of cleanup we want to do for deduplication after
// that but so far in most programs that's rare. Lots of dynamic shapes or
// detensorized parameters may change that.

// TODO(#8043): handle host/device size bit width mismatches; if the host is
// 32 and device is 64 we don't need to pass the known-zero hi word. We need to
// check for this specifically as during this pass we won't know that the index
// type is range limited otherwise.

// TODO(benvanik): rework this to actually pack. We could pack multiple operands
// together (i16 + i8 + i8 -> i32). We'd probably also want to apply some struct
// packing rules to ensure alignment. This current implementation just looks for
// any i64 (or equivalent) operand and splits it in to two i32s that get
// stitched back together.

// Converts |operand| to a sequence of one or more i32 values.
static void convertAndDecomposeToI32s(
    Location loc, Value operand, SmallVector<Value> &newOperands,
    IREE::Stream::ResourceConfigAttr resourceConfig, OpBuilder &builder) {
  // NOTE: we do these in sequence so that we can reuse type expansion; we
  // want f64 to become i64 so that i64 can become i32+i32 etc.

  // Complex types are decomposed into real/imaginary components and processed
  // as if they were independent.
  if (auto complexType = dyn_cast<ComplexType>(operand.getType())) {
    auto real = builder.create<complex::ReOp>(loc, operand);
    auto imag = builder.create<complex::ImOp>(loc, operand);
    convertAndDecomposeToI32s(loc, real, newOperands, resourceConfig, builder);
    convertAndDecomposeToI32s(loc, imag, newOperands, resourceConfig, builder);
    return;
  }

  // If the value complex from a complex::BitcastOp we should grab the
  // real / complex values instead.
  if (auto bitcast =
          dyn_cast_or_null<complex::BitcastOp>(operand.getDefiningOp())) {
    auto complexOperand = bitcast.getOperand();
    auto complexTy = cast<ComplexType>(complexOperand.getType());
    auto real = builder.createOrFold<complex::ReOp>(
        loc, complexTy.getElementType(), complexOperand);
    auto imag = builder.createOrFold<complex::ImOp>(
        loc, complexTy.getElementType(), complexOperand);
    convertAndDecomposeToI32s(loc, real, newOperands, resourceConfig, builder);
    convertAndDecomposeToI32s(loc, imag, newOperands, resourceConfig, builder);
    return;
  }

  // index -> i32 or i64
  if (operand.getType().isIndex()) {
    // Convert index types to a concrete bit width. 'index' can mean different
    // things on the host and device as well as varying across devices.
    // Today we just hardcode to i32 as we are working with smallish data and
    // i64 uses 2x as much of our limited push constant space and is much
    // slower to work with on mobile GPUs. In the future we will want to flag
    // this as a global setting as well as have some heuristics for deriving
    // from target devices.
    operand = builder.createOrFold<arith::IndexCastUIOp>(
        loc, builder.getIntegerType(resourceConfig.getIndexBits()), operand);
  }

  // bf16 -> i16, f32 -> i32, f64 -> i64 ...
  if (auto floatType = dyn_cast<FloatType>(operand.getType())) {
    // Floats need to be bitcast into opaque integers so that our handling
    // below can deal with simple fixed width ints (bf16->i16, etc).
    operand = builder.createOrFold<arith::BitcastOp>(
        loc, builder.getIntegerType(floatType.getIntOrFloatBitWidth()),
        operand);
  }

  // i1-i31 -> i32 and i33-i63 -> i64
  // TODO(benvanik): don't extend here but instead pack as we can fit 4 i8's
  // into a single i32 and save 4x our push constant capacity
  unsigned bitWidth = IREE::Util::getTypeBitWidth(operand.getType());
  if (bitWidth < 31) {
    operand = builder.createOrFold<arith::ExtUIOp>(loc, builder.getI32Type(),
                                                   operand);
  } else if (bitWidth > 32 && bitWidth < 64) {
    operand = builder.createOrFold<arith::ExtUIOp>(loc, builder.getI64Type(),
                                                   operand);
  }

  // i64 -> i32 + i32
  if (operand.getType().isInteger(64)) {
    // TODO(benvanik): reorder operands to preserve natural alignment; right
    // now this can split i64 across 4 byte boundaries which sucks.
    // TODO(benvanik): use something like IndexSet for the shift amount to
    // reduce the amount of IR we emit when passing multiple operands.
    // lo = i32(operand)
    // hi = i32(operand >> 32)
    auto lo = builder.createOrFold<arith::TruncIOp>(loc, builder.getI32Type(),
                                                    operand);
    auto hi = builder.createOrFold<arith::TruncIOp>(
        loc, builder.getI32Type(),
        builder.createOrFold<arith::ShRUIOp>(
            loc, builder.getI64Type(), operand,
            builder.create<arith::ConstantIntOp>(loc, 32, 64)));
    newOperands.push_back(lo);
    newOperands.push_back(hi);
  } else {
    newOperands.push_back(operand);
  }
}

// Converts stream.cmd.dispatch operands into their packed representation. This
// will add/remove/reorder operands and must mirrored in a consistent manner
// with argument changes in the executable function (handled below).
static void updateDispatchOp(IREE::Stream::CmdDispatchOp dispatchOp,
                             IREE::Stream::ExecutableExportOp exportOp) {
  // Insert ops outside of the execution region.
  auto parentOp = dispatchOp->getParentOfType<IREE::Stream::CmdExecuteOp>();
  assert(parentOp && "dispatch ops must be within an execution region");
  OpBuilder builder(parentOp);

  auto resourceConfig = IREE::Stream::ResourceConfigAttr::lookup(exportOp);

  auto loc = dispatchOp.getLoc();
  SmallVector<Value> newOperands;
  for (auto operand : dispatchOp.getUniformOperands()) {
    // Decompose the operand to a sequence of i32 values.
    // This must be consistent with recomposeFromI32s.
    convertAndDecomposeToI32s(loc, operand, newOperands, resourceConfig,
                              builder);
  }
  dispatchOp.getUniformOperandsMutable().assign(newOperands);
}

// Recompose integer values from multiple i64 values. The values remain in their
// integer types and need to be reconverted.
static Value recomposeFromI32sAndConvert(
    Block *entryBlock, Location loc, Type oldArgType, DictionaryAttr oldArgAttr,
    SmallVector<Type> &newArgTypes, SmallVector<DictionaryAttr> &newArgAttrs,
    IREE::Stream::ResourceConfigAttr resourceConfig, OpBuilder &builder) {
  // NOTE: we do the packing in a reverse sequence so that we can reuse type
  // expansion; if we did f64->i64->i32+i32 in convertAndDecomposeToI32s we need
  // to do i32+i32->i64->f64 here.

  // Complex types need to be reconstructed from their elemental parts.
  if (auto complexType = dyn_cast<ComplexType>(oldArgType)) {
    Value real = recomposeFromI32sAndConvert(
        entryBlock, loc, complexType.getElementType(), oldArgAttr, newArgTypes,
        newArgAttrs, resourceConfig, builder);
    Value imag = recomposeFromI32sAndConvert(
        entryBlock, loc, complexType.getElementType(), oldArgAttr, newArgTypes,
        newArgAttrs, resourceConfig, builder);
    return builder.create<complex::CreateOp>(loc, oldArgType, real, imag);
  }

  // Pass through/no change other types (!stream.binding, probably).
  if (!oldArgType.isIntOrIndexOrFloat()) {
    newArgTypes.push_back(oldArgType);
    newArgAttrs.push_back(oldArgAttr);
    return entryBlock->addArgument(oldArgType, loc);
  }

  // If the arg was decomposed into i32s then first recompose it into an i64.
  Value value;
  bool wasDecomposed =
      (oldArgType.isIndex() && resourceConfig.getIndexBits() > 32) ||
      (oldArgType.isIntOrFloat() && oldArgType.getIntOrFloatBitWidth() > 32);
  if (wasDecomposed) {
    // The existing arg becomes the lo word and we need to insert the hi word.
    auto loArg = entryBlock->addArgument(builder.getI32Type(), loc);
    newArgTypes.push_back(loArg.getType());
    auto hiArg = entryBlock->addArgument(builder.getI32Type(), loc);
    newArgTypes.push_back(hiArg.getType());
    // i64(lo) | (i64(hi) << 32)
    value = builder.create<arith::OrIOp>(
        loc, builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), loArg),
        builder.create<arith::ShLIOp>(
            loc,
            builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), hiArg),
            builder.create<arith::ConstantIntOp>(loc, 32, 64)));
  } else {
    // Forced bitcast.
    value = entryBlock->addArgument(builder.getI32Type(), loc);
    newArgTypes.push_back(value.getType());
  }

  // i32 or i64 -> index
  if (oldArgType.isIndex()) {
    value = builder.create<arith::IndexCastUIOp>(loc, builder.getIndexType(),
                                                 value);
  }

  // Truncate back to original bit width.
  // i32 -> i16, i64 -> i48, ...
  if (oldArgType.isIntOrFloat() && oldArgType.getIntOrFloatBitWidth() < 32) {
    value = builder.create<arith::TruncIOp>(
        loc, builder.getIntegerType(oldArgType.getIntOrFloatBitWidth()), value);
  }

  // i16 -> bf16, i32 -> f32, i64 -> f64 ...
  if (auto floatType = llvm::dyn_cast<FloatType>(oldArgType)) {
    value = builder.create<arith::BitcastOp>(loc, oldArgType, value);
  }

  // Preserve the arg attrs on either the final op or the function argument
  // if none was required.
  if (auto definingOp = value.getDefiningOp()) {
    if (oldArgAttr)
      definingOp->setAttrs(oldArgAttr);
    newArgAttrs.push_back(nullptr);
  } else {
    newArgAttrs.push_back(oldArgAttr);
  }
  // Note that if we had decomposed the arg we'll expect that there are two attr
  // dicts for the two new args.
  if (wasDecomposed)
    newArgAttrs.push_back(nullptr);

  return value;
}

// Updates an exported function in a stream.executable to match the packing
// that was applied to dispatch ops above.
//
// This is a mirror of updateDispatchOp; see that for more information.
static void updateExportFuncOp(mlir::func::FuncOp funcOp) {
  assert(!funcOp.empty() && "can't have empty exported functions");
  auto &entryBlock = funcOp.getFunctionBody().front();
  auto builder = OpBuilder::atBlockBegin(&entryBlock);

  auto resourceConfig = IREE::Stream::ResourceConfigAttr::lookup(funcOp);

  // Recompose i32 args into i64s and (if needed) convert them.
  // This appends new arguments and then we clean up the old ones below.
  SmallVector<Type> newArgTypes;
  SmallVector<DictionaryAttr> newArgAttrs;
  auto oldArgs = llvm::to_vector(entryBlock.getArguments());
  for (auto oldArg : oldArgs) {
    auto oldArgAttr = funcOp.getArgAttrDict(oldArg.getArgNumber());
    auto newArg = recomposeFromI32sAndConvert(
        &entryBlock, oldArg.getLoc(), oldArg.getType(), oldArgAttr, newArgTypes,
        newArgAttrs, resourceConfig, builder);
    oldArg.replaceAllUsesWith(newArg);
  }

  // Remove all the original arguments from the entry block.
  entryBlock.eraseArguments(0, oldArgs.size());

  // Update the function signature and arg attrs that may have changed.
  funcOp.setType(builder.getFunctionType(
      newArgTypes, funcOp.getFunctionType().getResults()));
  funcOp.setAllArgAttrs(newArgAttrs);
}

//===----------------------------------------------------------------------===//
// --iree-hal-pack-dispatch-operands
//===----------------------------------------------------------------------===//

struct PackDispatchOperandsPass
    : public IREE::Stream::impl::PackDispatchOperandsPassBase<
          PackDispatchOperandsPass> {
  void runOnOperation() override {
    SymbolTable symbolTable(getOperation());

    // Convert all public function signatures and manipulate the arguments.
    for (auto executableOp :
         getOperation().getOps<IREE::Stream::ExecutableOp>()) {
      auto innerModuleOp = executableOp.getInnerModule();
      if (!innerModuleOp)
        continue;
      for (auto funcOp : innerModuleOp.getOps<mlir::func::FuncOp>()) {
        if (funcOp.isPublic()) {
          updateExportFuncOp(funcOp);
        }
      }
    }

    // Walk the module and update all dispatch operands.
    getOperation()->walk([&](IREE::Stream::CmdDispatchOp dispatchOp) {
      dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPointAttr) {
        auto exportOp =
            symbolTable
                .lookupNearestSymbolFrom<IREE::Stream::ExecutableExportOp>(
                    dispatchOp, entryPointAttr);
        if (exportOp) {
          updateDispatchOp(dispatchOp, exportOp);
        }
      });
      return WalkResult::advance();
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
