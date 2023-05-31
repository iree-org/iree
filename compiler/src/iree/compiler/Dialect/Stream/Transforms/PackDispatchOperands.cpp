// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-pack-dispatch-operands"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {
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

// TODO(benvanik): handle index types here based on target devices. For now
// we assume 32-bit device sizes and cast all index operands to i32.

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
    // NOTE: we do these in sequence so that we can reuse type expansion; we
    // want f64 to become i64 so that i64 can become i32+i32 etc.

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
    if (auto floatType = llvm::dyn_cast<FloatType>(operand.getType())) {
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
  dispatchOp.getUniformOperandsMutable().assign(newOperands);
}

// Updates an exported function in a stream.executable to match the packing
// that was applied to dispatch ops above.
//
// This is a mirror of updateDispatchOp; see that for more information.
static void updateExportFuncOp(mlir::func::FuncOp funcOp) {
  assert(!funcOp.empty() && "can't have empty exported functions");
  auto &entryBlock = funcOp.getFunctionBody().front();
  auto builder = OpBuilder::atBlockBegin(&entryBlock);
  auto streamAlignmentAttr = builder.getStringAttr("stream.alignment");
  auto streamValuesAttr = builder.getStringAttr("stream.values");

  auto resourceConfig = IREE::Stream::ResourceConfigAttr::lookup(funcOp);

  // NOTE: we have !stream.binding mixed in here; we only want to look at
  // primitives.

  // NOTE: we do the packing in a reverse sequence so that we can reuse type
  // expansion; if we did f64->i64->i32+i32 above we need to do
  // i32+i32->i64->f64 here.

  // i32+i32->i64 is the only case that adds arguments today so to keep things
  // simple we do that first. This ensures we have single SSA values for arg
  // to operand mapping.
  SmallVector<Type> newArgTypes;
  SmallVector<DictionaryAttr> newArgAttrs;
  auto oldArgAttrs = funcOp.getAllArgAttrs();
  for (auto it :
       llvm::enumerate(llvm::to_vector<4>(funcOp.getArgumentTypes()))) {
    auto targetType = it.value();
    if (!targetType.isIntOrIndexOrFloat() ||
        (targetType.isIndex() && resourceConfig.getIndexBits() <= 32) ||
        (targetType.isIntOrFloat() &&
         targetType.getIntOrFloatBitWidth() <= 32)) {
      // Pass through/no change.
      if (oldArgAttrs) {
        newArgAttrs.push_back(
            llvm::dyn_cast_if_present<DictionaryAttr>(oldArgAttrs[it.index()]));
      } else {
        newArgAttrs.push_back(nullptr);
      }
      newArgTypes.push_back(targetType);
      continue;
    }

    // The existing arg becomes the lo word and we need to insert the hi word.
    auto loArg = entryBlock.getArgument(newArgTypes.size());
    newArgTypes.push_back(builder.getI32Type());
    loArg.setType(builder.getI32Type());
    auto hiArg = entryBlock.insertArgument(
        newArgTypes.size(), builder.getI32Type(), funcOp.getLoc());
    newArgTypes.push_back(builder.getI32Type());

    // i64(lo) | (i64(hi) << 32)
    auto loc = loArg.getLoc();
    auto loArgI64 =
        builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), loArg);
    Operation *argOp = builder.create<arith::OrIOp>(
        loc, loArgI64,
        builder.create<arith::ShLIOp>(
            loc,
            builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), hiArg),
            builder.create<arith::ConstantIntOp>(loc, 32, 64)));

    // If going to index we need to insert a cast as the type changes.
    if (targetType.isIndex()) {
      argOp = builder.create<arith::IndexCastUIOp>(loc, targetType,
                                                   argOp->getResult(0));
    }

    // Replace all subsequent uses with the new reconstituted value.
    loArg.replaceAllUsesExcept(argOp->getResult(0), loArgI64);

    // Take the existing potential values, if present, and fix them up.
    // If we had a `stream.values = [0xAA...BB... : i64]` then we can
    // put a `stream.values = [0xBB... : i32]` on the lo and `[0xAA... : i32]`
    // on the hi.
    //
    // TODO(benvanik): find a nicer way/write some utilities - manipulating
    // these attributes is really annoying.
    if (oldArgAttrs) {
      auto oldArgAttr =
          llvm::dyn_cast_if_present<DictionaryAttr>(oldArgAttrs[it.index()]);
      if (auto alignmentAttr =
              oldArgAttr.getAs<IntegerAttr>(streamAlignmentAttr)) {
        argOp->setAttr(streamAlignmentAttr, alignmentAttr);
      }
      if (auto valuesAttr = oldArgAttr.getAs<ArrayAttr>(streamValuesAttr)) {
        // Attach the original potential value set to the new arg value.
        // This allows any further analysis walking the SSA values to find the
        // easily understandable i64 types instead of having to reconstruct it
        // from the i32 values and the combining operations (which MLIR doesn't
        // do today).
        argOp->setAttr(streamValuesAttr, valuesAttr);
      }
    }
    newArgAttrs.push_back(nullptr);
    newArgAttrs.push_back(nullptr);
  }
  if (newArgTypes.size() != funcOp.getNumArguments()) {
    // Changed argument count; update signature.
    funcOp.setType(builder.getFunctionType(
        newArgTypes, funcOp.getFunctionType().getResults()));
    funcOp.setAllArgAttrs(newArgAttrs);
  }

  // -- now the function has all <= 32-bit operands --

  newArgTypes.clear();
  for (auto arg : entryBlock.getArguments()) {
    auto alignmentAttr = funcOp.getArgAttrOfType<IntegerAttr>(
        arg.getArgNumber(), streamAlignmentAttr);
    auto oldValuesAttr = funcOp.getArgAttrOfType<ArrayAttr>(arg.getArgNumber(),
                                                            streamValuesAttr);
    auto targetType = arg.getType();
    if (!targetType.isIntOrIndexOrFloat()) {
      // Pass-through !stream.bindings.
      newArgTypes.push_back(targetType);
      continue;
    }

    auto loc = arg.getLoc();
    Value value = arg;
    arg.setType(builder.getI32Type());

    // i32 or i64 -> index
    if (targetType.isIndex()) {
      auto castOp = builder.create<arith::IndexCastUIOp>(
          loc, builder.getIndexType(), value);
      value.replaceAllUsesExcept(castOp.getResult(), castOp);
      value = castOp.getResult();
    }

    // Truncate back to original bit width.
    // i32 -> i16, i64 -> i48, ...
    if (targetType.isIntOrFloat() && targetType.getIntOrFloatBitWidth() < 32) {
      auto truncOp = builder.create<arith::TruncIOp>(
          loc, builder.getIntegerType(targetType.getIntOrFloatBitWidth()),
          value);
      value.replaceAllUsesExcept(truncOp.getResult(), truncOp);
      value = truncOp.getResult();
    }

    // i16 -> bf16, i32 -> f32, i64 -> f64 ...
    if (auto floatType = llvm::dyn_cast<FloatType>(targetType)) {
      auto bitcastOp = builder.create<arith::BitcastOp>(loc, targetType, value);
      value.replaceAllUsesExcept(bitcastOp.getResult(), bitcastOp);
      value = bitcastOp.getResult();
    }

    // Set the original streams.values attribute on the op with the type in the
    // original form. This allows subsequent analysis to easily find the value.
    if (auto definingOp = value.getDefiningOp()) {
      if (alignmentAttr) {
        definingOp->setAttr(streamAlignmentAttr, alignmentAttr);
        funcOp.removeArgAttr(arg.getArgNumber(), streamAlignmentAttr);
      }
      if (oldValuesAttr) {
        definingOp->setAttr(streamValuesAttr, oldValuesAttr);
        funcOp.removeArgAttr(arg.getArgNumber(), streamValuesAttr);
      }
    }

    newArgTypes.push_back(builder.getI32Type());
  }
  funcOp.setType(builder.getFunctionType(
      newArgTypes, funcOp.getFunctionType().getResults()));
}

//===----------------------------------------------------------------------===//
// -iree-hal-pack-dispatch-operands
//===----------------------------------------------------------------------===//

class PackDispatchOperandsPass
    : public PackDispatchOperandsBase<PackDispatchOperandsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
  }

  void runOnOperation() override {
    SymbolTable symbolTable(getOperation());

    // Convert all public function signatures and manipulate the arguments.
    for (auto executableOp :
         getOperation().getOps<IREE::Stream::ExecutableOp>()) {
      for (auto funcOp :
           executableOp.getInnerModule().getOps<mlir::func::FuncOp>()) {
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

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createPackDispatchOperandsPass() {
  return std::make_unique<PackDispatchOperandsPass>();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
