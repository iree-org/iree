// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_MATERIALIZEENCODINGSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {
struct MaterializeEncodingsPass
    : public IREE::Stream::impl::MaterializeEncodingsPassBase<
          MaterializeEncodingsPass> {
  void runOnOperation() override;
};
} // namespace

/// Returns a pretty function name based on the `encodeOp` source and result
/// types. Note that the encodings are dropped in the name because it could be
/// too verbose.
static std::string getDispatchFuncName(IREE::Stream::TensorEncodeOp encodeOp) {
  std::string str;
  llvm::raw_string_ostream os(str);
  auto printShape = [&](RankedTensorType type) {
    for (auto dimSize : type.getShape()) {
      if (ShapedType::isDynamic(dimSize)) {
        os << "D";
      } else {
        os << std::to_string(dimSize);
      }
      os << "x";
    }
    type.getElementType().print(os);
  };

  os << "encode_";
  printShape(dyn_cast<RankedTensorType>(encodeOp.getSourceEncoding()));
  os << "_to_";
  printShape(dyn_cast<RankedTensorType>(encodeOp.getResultEncoding()));
  return str;
}

/// Creates a workgroup function for the `encodedOp`. The order of function
/// arguments is
///   - source_binding
///   - dynamic dimension sizes of the source type
///   - dynamic dimension sizes of the destination type
///   - destination binding
static func::FuncOp createWorkgroupFunc(IREE::Stream::TensorEncodeOp encodeOp,
                                        StringRef functionName) {
  Location loc = encodeOp.getLoc();
  MLIRContext *ctx = encodeOp.getContext();
  SmallVector<Type> argumentTypes;
  SmallVector<Location> argumentLocs;
  auto bindingType = IREE::Stream::BindingType::get(ctx);
  int ordinalCount = 0;

  // Add the block argument for the source resource and corresponding dynamic
  // dimension sizes.
  argumentTypes.push_back(bindingType);
  argumentLocs.push_back(loc);
  for (auto argument : encodeOp.getSourceEncodingDims()) {
    argumentTypes.push_back(argument.getType());
    argumentLocs.push_back(argument.getLoc());
  }

  // Add the block argument for the result resource and corresponding dynamic
  // dimension sizes.
  for (auto argument : encodeOp.getResultEncodingDims()) {
    argumentTypes.push_back(argument.getType());
    argumentLocs.push_back(argument.getLoc());
  }
  argumentTypes.push_back(bindingType);
  argumentLocs.push_back(loc);

  // Build function type matching the region signature.
  auto functionType = FunctionType::get(ctx, argumentTypes, /*results=*/{});
  auto funcOp = mlir::func::FuncOp::create(loc, functionName, functionType);
  Block &block = funcOp.getBody().emplaceBlock();
  block.addArguments(argumentTypes, argumentLocs);
  OpBuilder builder(funcOp.getBody());

  // Build operations to handle load/store from/to the bindings.
  SmallVector<Value> sourceDynamicDims;
  SmallVector<Value> destinationDynamicDims;
  for (auto argument : block.getArguments().drop_front(1).take_front(
           encodeOp.getSourceEncodingDims().size())) {
    sourceDynamicDims.push_back(
        builder.create<IREE::TensorExt::DispatchWorkloadOrdinalOp>(
            loc, argument, builder.getIndexAttr(ordinalCount++)));
  }
  for (auto argument : block.getArguments().drop_back(1).take_back(
           encodeOp.getResultEncodingDims().size())) {
    destinationDynamicDims.push_back(
        builder.create<IREE::TensorExt::DispatchWorkloadOrdinalOp>(
            loc, argument, builder.getIndexAttr(ordinalCount++)));
  }

  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto sourceDispatchType = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::ReadOnly, encodeOp.getSourceEncoding());
  Value source = builder.create<IREE::Stream::BindingSubspanOp>(
      loc, sourceDispatchType, block.getArgument(0), zero, sourceDynamicDims);
  auto destinationDispatchType = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::WriteOnly, encodeOp.getResultEncoding());
  Value destination = builder.create<IREE::Stream::BindingSubspanOp>(
      loc, destinationDispatchType, block.getArguments().back(), zero,
      destinationDynamicDims);

  // Load the value from the source binding.
  RankedTensorType sourceType = sourceDispatchType.asRankedTensorType();
  Value value = builder.create<IREE::TensorExt::DispatchTensorLoadOp>(
      loc, sourceType, source, sourceDynamicDims);

  // We can only add/remove encodings using set_encoding/unset_encoding ops
  // today. Thus, we firstly need to bring the tensor encodings to pure tensor
  // types, and then encode the tensor types when needed.
  RankedTensorType destinationType =
      destinationDispatchType.asRankedTensorType();
  if (sourceType != destinationType) {
    if (sourceType.getEncoding()) {
      value = builder.create<IREE::Encoding::UnsetEncodingOp>(
          loc, sourceType.dropEncoding(), value, sourceDynamicDims);
    }
    if (destinationType.getEncoding()) {
      value = builder.create<IREE::Encoding::SetEncodingOp>(
          loc, destinationType, value);
    }
  }

  // Store the value to the destination binding.
  builder.create<IREE::TensorExt::DispatchTensorStoreOp>(
      loc, value, destination, destinationDynamicDims);
  builder.create<func::ReturnOp>(loc);

  return funcOp;
}

/// Creates an export op pointing at the `funcOp` function.
static IREE::Stream::ExecutableExportOp
createExportOp(RewriterBase &rewriter, Location loc,
               IREE::Stream::TensorEncodeOp encodeOp,
               IREE::Stream::ExecutableOp executableOp, func::FuncOp funcOp) {
  SmallVector<Type> workloadTypes;
  SmallVector<Location> workloadLocs;
  for (auto argument : encodeOp.getSourceEncodingDims()) {
    Type argumentType = argument.getType();
    if (!llvm::isa<IndexType>(argumentType)) {
      continue;
    }
    workloadTypes.push_back(argumentType);
    workloadLocs.push_back(argument.getLoc());
  }
  for (auto argument : encodeOp.getResultEncodingDims()) {
    Type argumentType = argument.getType();
    if (!llvm::isa<IndexType>(argumentType)) {
      continue;
    }
    workloadTypes.push_back(argumentType);
    workloadLocs.push_back(argument.getLoc());
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&executableOp.getBody().front());
  auto exportOp = rewriter.create<IREE::Stream::ExecutableExportOp>(
      loc, funcOp.getName(), SymbolRefAttr::get(funcOp));
  Block *block = rewriter.createBlock(&exportOp.getWorkgroupCount(),
                                      exportOp.getWorkgroupCount().end(),
                                      workloadTypes, workloadLocs);
  rewriter.setInsertionPointToStart(block);
  auto defaultCountOp =
      rewriter.create<IREE::TensorExt::DispatchWorkgroupCountFromSliceOp>(
          loc, block->getArguments());
  rewriter.create<IREE::Stream::ReturnOp>(loc, defaultCountOp.getResults());
  return exportOp;
}

/// Creates the executable op and build the content for `encodeOp`. The
/// executable op and the export op are returned for further lowering.
static std::pair<IREE::Stream::ExecutableOp, IREE::Stream::ExecutableExportOp>
createExecutableAndExport(RewriterBase &rewriter,
                          IREE::Stream::TensorEncodeOp encodeOp,
                          int executableId) {
  OpBuilder::InsertionGuard guard(rewriter);

  auto parentFuncOp = encodeOp->getParentOfType<FunctionOpInterface>();
  ModuleOp parentModuleOp = parentFuncOp->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(&parentModuleOp.getBody()->back());
  Location loc = encodeOp.getLoc();
  std::string executableName = "_encoding_" + std::to_string(executableId);
  auto executableOp =
      rewriter.create<IREE::Stream::ExecutableOp>(loc, executableName);
  executableOp.getOperation()->moveBefore(parentFuncOp);
  executableOp.setPrivate();

  // Build the inner module and func op, and an export op pointing at the
  // function.
  std::string funcName = executableName + "_" + getDispatchFuncName(encodeOp);
  auto funcOp = createWorkgroupFunc(encodeOp, funcName);
  rewriter.setInsertionPointToStart(&executableOp.getBody().front());
  auto innerModule = rewriter.create<mlir::ModuleOp>(loc);
  innerModule.push_back(funcOp);
  IREE::Stream::ExecutableExportOp exportOp =
      createExportOp(rewriter, loc, encodeOp, executableOp, funcOp);
  return std::make_pair(executableOp, exportOp);
}

/// Returns the encoding signature for dispatch as ArrayAttr form. Currently,
/// only the source encoding and the result encoding matter.
static ArrayAttr getEncodingSignature(Builder &builder,
                                      IREE::Stream::TensorEncodeOp encodeOp) {
  return builder.getArrayAttr(
      {encodeOp.getSourceEncodingAttr(), encodeOp.getResultEncodingAttr()});
}

/// Replaces the `encodeOp` with a `stream.async.dispatch` op on the given
/// `executableOp` and `exportOp`.
static void
replaceEncodeOpWithDispatchOp(RewriterBase &rewriter,
                              IREE::Stream::TensorEncodeOp encodeOp,
                              IREE::Stream::ExecutableOp executableOp,
                              IREE::Stream::ExecutableExportOp exportOp) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(encodeOp);
  Value zero = rewriter.create<arith::ConstantIndexOp>(encodeOp.getLoc(), 0);
  SmallVector<Value> operandOffsets = {zero};
  SmallVector<Value> operandEnds = {encodeOp.getSourceSize()};
  SmallVector<Value> operandLengths = {encodeOp.getSourceSize()};
  SmallVector<Value> operands = {encodeOp.getSource()};
  for (auto argument : encodeOp.getSourceEncodingDims()) {
    operands.push_back(argument);
  }
  for (auto argument : encodeOp.getResultEncodingDims()) {
    operands.push_back(argument);
  }

  SmallVector<int64_t> tiedArguments = {
      IREE::Util::TiedOpInterface::kUntiedIndex};
  SmallVector<Value> dynamicDims;
  for (Value argument : encodeOp.getSourceEncodingDims()) {
    dynamicDims.push_back(argument);
  }
  for (Value argument : encodeOp.getResultEncodingDims()) {
    dynamicDims.push_back(argument);
  }
  rewriter.replaceOpWithNewOp<IREE::Stream::AsyncDispatchOp>(
      encodeOp, exportOp,
      /*workload=*/dynamicDims, encodeOp.getResult().getType(), operands,
      encodeOp.getSourceSize(), operandOffsets, operandEnds, operandLengths,
      encodeOp.getResultSize(), tiedArguments, encodeOp.getAffinityAttr());
}

void MaterializeEncodingsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ModuleOp moduleOp = getOperation();

  RewritePatternSet patterns(ctx);
  IREE::Stream::TensorEncodeOp::getCanonicalizationPatterns(patterns, ctx);
  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    return signalPassFailure();
  }

  SmallVector<IREE::Stream::TensorEncodeOp> encodeOps;
  for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
    funcOp.walk(
        [&](IREE::Stream::TensorEncodeOp op) { encodeOps.push_back(op); });
  }
  if (encodeOps.empty()) {
    return;
  }

  // Mapping from (sourceEncoding, resultEncoding) to the executable op and the
  // export op. The encoding changes are described by the encoding pairs and the
  // executables can be reused by stream.tensor.encode ops materialization.
  DenseMap<ArrayAttr, std::pair<IREE::Stream::ExecutableOp,
                                IREE::Stream::ExecutableExportOp>>
      cachedExecutables;

  IRRewriter rewriter(ctx);
  int executableId = 0;
  for (auto encodeOp : encodeOps) {
    ArrayAttr encodingSignature = getEncodingSignature(rewriter, encodeOp);
    if (!cachedExecutables.contains(encodingSignature)) {
      cachedExecutables[encodingSignature] =
          createExecutableAndExport(rewriter, encodeOp, executableId++);
    }
    auto [executableOp, exportOp] = cachedExecutables[encodingSignature];
    replaceEncodeOpWithDispatchOp(rewriter, encodeOp, executableOp, exportOp);
  }
}

} // namespace mlir::iree_compiler::IREE::Stream
