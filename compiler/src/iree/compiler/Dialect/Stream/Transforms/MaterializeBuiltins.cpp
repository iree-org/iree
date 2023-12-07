// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Stream/Builtins/Builtins.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Utils/ElementPackingUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

// We could overengineer this with custom DSLs and python generation and all
// that kind of stuff - but that stuff really belongs closer to the frontend
// (linalg/etc). Once we are at this point we are just patching over things for
// compatibility and not handling arbitrary programs. A linalg.fill of an i64
// that gets tiled and fused with other operations is always going to be several
// of orders of magnitude faster than this approach and we should spend our
// effort improving things at that layer instead of leaning too much on this.
//
// Consider these as replacements for the blobs we'd have to ship with every
// deployment of the runtime (for all target platforms/HAL backends/etc) - in
// that sense this is a dramatically more scalable approach even if not perfect.

// TODO(#13984): memset emulation of all types is required for CUDA graphs due
// to a driver bug. Once it's fixed we can remove this global flag.
static llvm::cl::opt<bool>
    clEmulateMemset("iree-stream-emulate-memset",
                    llvm::cl::desc("Emulate all memset types with dispatches."),
                    llvm::cl::init(false));

namespace mlir::iree_compiler::IREE::Stream {

//===----------------------------------------------------------------------===//
// Type utilities
//===----------------------------------------------------------------------===//

enum class TypeSupport {
  // Natively supported by the stream ops (stream.async.fill, etc).
  Native,
  // Emulated with a builtin (stream.builtin.fill, etc).
  Builtin,
  // Entirely unsupported and compilation cannot continue.
  Unsupported,
};

// Returns the support for the given type based on the resource configuration.
static TypeSupport
queryTypeSupport(IREE::Stream::ResourceConfigAttr resourceConfig, Type type) {
  unsigned bitWidth = IREE::Util::getTypeBitWidth(type);

  if (bitWidth > 32) {
    // 64-bit fills are currently always handled by builtins because the HAL
    // does not support them as CUDA, Vulkan, and Metal don't. If we start to
    // get APIs with better support we can make this conditional based on the
    // resource config.
    return TypeSupport::Builtin;
  }

  // TODO(benvanik): use the resource config instead of command line flags.
  // For now this is mostly done as a workaround or fixed attribute of the
  // HAL (no i64/complex support, etc) so we don't.
  if (clEmulateMemset) {
    return TypeSupport::Builtin;
  }

  // We can canonicalize into something we support natively.
  return TypeSupport::Native;
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Merges a builtin module from iree/compiler/Dialect/Stream/Builtins/*.mlir
// into the user module; this allows for host functions and multiple
// executables.
//
// Fails if there's a name conflict; we have a __ prefix and things outside the
// compiler shouldn't use it.
static LogicalResult mergeBuiltinModuleSource(Location loc, StringRef fileName,
                                              Operation *targetOp,
                                              OpBuilder &targetBuilder) {
  // Find the file in the embedded data.
  const iree_file_toc_t *toc = iree_compiler_Stream_Builtins_create();
  const iree_file_toc_t *file = nullptr;
  for (size_t i = 0; i < iree_compiler_Stream_Builtins_size(); ++i) {
    if (fileName == toc[i].name) {
      file = &toc[i];
      break;
    }
  }
  if (!file) {
    return mlir::emitError(
        loc, "unable to merge builtin module; file not found " + fileName);
  }
  return mergeSourceModuleInto(loc, StringRef(file->data, file->size), targetOp,
                               targetBuilder);
}

// Tracks all builtin modules required by the program and the locations of the
// ops that require them.
class RequiredModules {
public:
  // Inserts a module into the required module set for use by the op at |loc|.
  void insert(Location loc, StringRef moduleFile) {
    modules[moduleFile.str()].push_back(loc);
  }

  // Merges all required modules into |targetOp| using |targetBuilder| as an
  // insertion point.
  LogicalResult merge(Operation *targetOp, OpBuilder &targetBuilder) {
    for (auto &[moduleFile, locs] : modules) {
      auto fusedLoc = targetBuilder.getFusedLoc(locs);
      if (failed(mergeBuiltinModuleSource(fusedLoc, moduleFile, targetOp,
                                          targetBuilder))) {
        return failure();
      }
    }
    return success();
  }

private:
  // A map of builtin module filename to the locations using it.
  std::map<std::string, SmallVector<Location>> modules; // ordered
};

// Returns an OpBuilder at a point safe to insert arith/etc ops.
// When |nestedOp| is in an execution region this returns a builder inserting
// immediately prior to the region.
static OpBuilder getParentBuilder(Operation *nestedOp) {
  auto executeOp = nestedOp->getParentOfType<IREE::Stream::AsyncExecuteOp>();
  return OpBuilder(executeOp ? executeOp : nestedOp);
}

//===----------------------------------------------------------------------===//
// Builtins
//===----------------------------------------------------------------------===//

static LogicalResult replaceBuiltinSplatOp(IREE::Stream::AsyncSplatOp splatOp,
                                           Value pattern,
                                           RequiredModules &requiredModules) {
  auto loc = splatOp.getLoc();
  unsigned bitWidth = pattern.getType().getIntOrFloatBitWidth();
  StringRef builtinName;
  switch (bitWidth) {
  case 8:
    builtinName = "__builtin_splat_i8";
    requiredModules.insert(loc, "splat_i8.mlir");
    break;
  case 16:
    builtinName = "__builtin_splat_i16";
    requiredModules.insert(loc, "splat_i16.mlir");
    break;
  case 32:
    builtinName = "__builtin_splat_i32";
    requiredModules.insert(loc, "splat_i32.mlir");
    break;
  case 64:
    builtinName = "__builtin_splat_i64";
    requiredModules.insert(loc, "splat_i64.mlir");
    break;
  default:
    return splatOp.emitOpError() << "has no builtin for bit width " << bitWidth;
  }

  auto arithBuilder = getParentBuilder(splatOp);
  auto byteWidth =
      arithBuilder.createOrFold<arith::ConstantIndexOp>(loc, bitWidth / 8);
  auto elementCount = arithBuilder.createOrFold<arith::DivUIOp>(
      loc, splatOp.getResultSize(), byteWidth);

  Value workload[] = {elementCount};
  SmallVector<Value> operands = {
      pattern,
      elementCount,
  };
  SmallVector<Value> operandSizes;
  SmallVector<Value> operandOffsets;
  SmallVector<Value> operandEnds;
  SmallVector<Value> operandLengths;
  SmallVector<int64_t> tiedOperands = {
      -1,
  };
  SmallVector<Value> resultSizes = {
      splatOp.getResultSize(),
  };
  SmallVector<Type> resultTypes{
      splatOp.getResult().getType(),
  };
  OpBuilder builder(splatOp);
  auto dispatchOp = builder.create<IREE::Stream::AsyncDispatchOp>(
      loc, resultTypes, workload,
      builder.getArrayAttr({SymbolRefAttr::get(
          builder.getStringAttr(builtinName),
          FlatSymbolRefAttr::get(builder.getContext(), builtinName))}),
      operands, operandSizes, operandOffsets, operandEnds, operandLengths,
      resultSizes, builder.getIndexArrayAttr(tiedOperands),
      splatOp.getAffinityAttr());
  splatOp.getResult().replaceAllUsesWith(dispatchOp.getResults().front());

  splatOp.erase();
  return success();
}

static LogicalResult processSplatOp(IREE::Stream::AsyncSplatOp splatOp,
                                    RequiredModules &requiredModules) {
  // Update the existing op or emit a new one.
  auto resourceConfig = IREE::Stream::ResourceConfigAttr::lookup(splatOp);
  auto pattern = splatOp.getValue();
  switch (queryTypeSupport(resourceConfig, pattern.getType())) {
  case TypeSupport::Native:
    // Already ok!
    return success();
  case TypeSupport::Builtin:
    return replaceBuiltinSplatOp(splatOp, pattern, requiredModules);
  default:
  case TypeSupport::Unsupported:
    return splatOp.emitOpError()
           << "has unsupported fill pattern type "
           << splatOp.getValue().getType() << " (tried converting to "
           << pattern.getType() << ")";
  }
}

static LogicalResult replaceBuiltinFillOp(IREE::Stream::AsyncFillOp fillOp,
                                          Value pattern,
                                          RequiredModules &requiredModules) {
  auto loc = fillOp.getLoc();
  unsigned bitWidth = pattern.getType().getIntOrFloatBitWidth();
  StringRef builtinName;
  switch (bitWidth) {
  case 8:
    builtinName = "__builtin_fill_i8";
    requiredModules.insert(loc, "fill_i8.mlir");
    break;
  case 16:
    builtinName = "__builtin_fill_i16";
    requiredModules.insert(loc, "fill_i16.mlir");
    break;
  case 32:
    builtinName = "__builtin_fill_i32";
    requiredModules.insert(loc, "fill_i32.mlir");
    break;
  case 64:
    builtinName = "__builtin_fill_i64";
    requiredModules.insert(loc, "fill_i64.mlir");
    break;
  default:
    return fillOp.emitOpError() << "has no builtin for bit width " << bitWidth;
  }

  auto arithBuilder = getParentBuilder(fillOp);
  auto byteWidth =
      arithBuilder.createOrFold<arith::ConstantIndexOp>(loc, bitWidth / 8);
  auto elementCount = arithBuilder.createOrFold<arith::DivUIOp>(
      loc, fillOp.getTargetLength(), byteWidth);

  Value workload[] = {elementCount};
  SmallVector<Value> operands = {
      fillOp.getTarget(),
      pattern,
      fillOp.getTargetOffset(),
      elementCount,
  };
  SmallVector<Value> operandSizes = {
      fillOp.getTargetSize(),
  };
  SmallVector<Value> operandOffsets = {
      fillOp.getTargetOffset(),
  };
  SmallVector<Value> operandEnds = {
      fillOp.getTargetEnd(),
  };
  SmallVector<Value> operandLengths = {
      fillOp.getTargetLength(),
  };
  SmallVector<int64_t> tiedOperands = {
      0,
  };
  SmallVector<Value> resultSizes = {
      fillOp.getTargetSize(),
  };
  SmallVector<Type> resultTypes{
      fillOp.getResult().getType(),
  };
  OpBuilder builder(fillOp);
  auto dispatchOp = builder.create<IREE::Stream::AsyncDispatchOp>(
      loc, resultTypes, workload,
      builder.getArrayAttr({SymbolRefAttr::get(
          builder.getStringAttr(builtinName),
          FlatSymbolRefAttr::get(builder.getContext(), builtinName))}),
      operands, operandSizes, operandOffsets, operandEnds, operandLengths,
      resultSizes, builder.getIndexArrayAttr(tiedOperands),
      fillOp.getAffinityAttr());
  fillOp.getResult().replaceAllUsesWith(dispatchOp.getResults().front());

  fillOp.erase();
  return success();
}

static LogicalResult processFillOp(IREE::Stream::AsyncFillOp fillOp,
                                   RequiredModules &requiredModules) {
  // Update the existing op or emit a new one.
  auto resourceConfig = IREE::Stream::ResourceConfigAttr::lookup(fillOp);
  auto pattern = fillOp.getValue();
  switch (queryTypeSupport(resourceConfig, pattern.getType())) {
  case TypeSupport::Native:
    // Already ok!
    return success();
  case TypeSupport::Builtin:
    return replaceBuiltinFillOp(fillOp, pattern, requiredModules);
  default:
  case TypeSupport::Unsupported:
    return fillOp.emitOpError()
           << "has unsupported fill pattern type "
           << fillOp.getValue().getType() << " (tried converting to "
           << pattern.getType() << ")";
  }
}

//===----------------------------------------------------------------------===//
// -iree-stream-materialize-builtins
//===----------------------------------------------------------------------===//

class MaterializeBuiltinsPass
    : public MaterializeBuiltinsBase<MaterializeBuiltinsPass> {
public:
  MaterializeBuiltinsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    // We need to include all dialects that the builtin modules use.
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::vector::VectorDialect>();
    registry.insert<IREE::Flow::FlowDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty())
      return;

    // Find and replace (if needed) ops that we want to turn into builtins
    // across the entire program.
    RequiredModules requiredModules;
    auto walkResult = getOperation()->walk(
        [&](IREE::Stream::StreamableOpInterface streamableOp) {
          return TypeSwitch<Operation *, WalkResult>(streamableOp)
              .Case<IREE::Stream::AsyncSplatOp>([&](auto splatOp) {
                return succeeded(processSplatOp(splatOp, requiredModules))
                           ? WalkResult::advance()
                           : WalkResult::interrupt();
              })
              .Case<IREE::Stream::AsyncFillOp>([&](auto fillOp) {
                return succeeded(processFillOp(fillOp, requiredModules))
                           ? WalkResult::advance()
                           : WalkResult::interrupt();
              })
              .Default([&](auto op) { return WalkResult::advance(); });
        });
    if (walkResult.wasInterrupted()) {
      return signalPassFailure();
    }

    // Merge required builtin modules into the main program module.
    OpBuilder moduleBuilder(&moduleOp.getBody()->front());
    if (failed(requiredModules.merge(moduleOp, moduleBuilder))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>> createMaterializeBuiltinsPass() {
  return std::make_unique<MaterializeBuiltinsPass>();
}

} // namespace mlir::iree_compiler::IREE::Stream
