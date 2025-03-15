// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

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

static std::string getDispatchFuncName(IREE::Stream::TensorEncodeOp encodeOp) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << "encode_";
  auto resultType = dyn_cast<RankedTensorType>(encodeOp.getResultEncoding());
  for (auto dimSize : resultType.getShape()) {
    if (ShapedType::isDynamic(dimSize)) {
      os << "D";
    } else {
      os << std::to_string(dimSize);
    }
    os << "x";
  }
  resultType.getElementType().print(os);
  return str;
}

static func::FuncOp createWorkgroupFunc(IREE::Stream::TensorEncodeOp encodeOp,
                                        StringRef functionName) {
  Location loc = encodeOp.getLoc();
  MLIRContext *ctx = encodeOp.getContext();
  SmallVector<Type> argumentTypes;
  SmallVector<Location> argumentLocs;
  auto bindingType = IREE::Stream::BindingType::get(ctx);
  // Source
  argumentTypes.push_back(bindingType);
  argumentLocs.push_back(loc);
  for (auto argument : encodeOp.getSourceEncodingDims()) {
    Type argumentType = argument.getType();
    if (!llvm::isa<IndexType>(argumentType)) {
      argumentTypes.push_back(bindingType);
      argumentLocs.push_back(loc);
    } else {
      argumentTypes.push_back(argumentType);
      argumentLocs.push_back(argument.getLoc());
    }
  }
  // Destination
  argumentTypes.push_back(bindingType);
  argumentLocs.push_back(loc);

  // Build function type matching the region signature.
  auto functionType =
      FunctionType::get(encodeOp.getContext(), argumentTypes, /*results=*/{});

  // Clone region into the function body.
  auto funcOp = mlir::func::FuncOp::create(loc, functionName, functionType);

  Block &block = funcOp.getBody().emplaceBlock();
  block.addArguments(argumentTypes, argumentLocs);
  OpBuilder builder(funcOp.getBody());

  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  int ordinalCount = 0;
  SmallVector<Value> dynamicDims;
  for (auto argument : block.getArguments()) {
    if (!llvm::isa<IndexType>(argument.getType())) {
      continue;
    }
    dynamicDims.push_back(builder.create<IREE::Flow::DispatchWorkloadOrdinalOp>(
        loc, argument, builder.getIndexAttr(ordinalCount++)));
  }
  auto sourceType = IREE::Flow::DispatchTensorType::get(
      IREE::Flow::TensorAccess::ReadOnly, encodeOp.getSourceEncoding());
  Value source = builder.create<IREE::Stream::BindingSubspanOp>(
      loc, sourceType, block.getArgument(0), zero, dynamicDims);
  auto destinationType = IREE::Flow::DispatchTensorType::get(
      IREE::Flow::TensorAccess::WriteOnly, encodeOp.getResultEncoding());
  Value destination = builder.create<IREE::Stream::BindingSubspanOp>(
      loc, destinationType, block.getArguments().back(), zero, dynamicDims);
  Value value = builder.create<IREE::Flow::DispatchTensorLoadOp>(
      loc, sourceType.asRankedTensorType(), source, dynamicDims);
  value = builder.create<IREE::Encoding::SetEncodingOp>(
      loc, destinationType.asRankedTensorType(), value);
  builder.create<IREE::Flow::DispatchTensorStoreOp>(loc, value, destination,
                                                    dynamicDims);
  builder.create<func::ReturnOp>(loc);

  return funcOp;
}

// TODO(hanchung): Refactor the function for better readability.
static IREE::Stream::ExecutableOp
createExecutableAndEntry(RewriterBase &rewriter,
                         IREE::Stream::TensorEncodeOp encodeOp,
                         int executableId) {

  auto parentFuncOp = encodeOp->getParentOfType<FunctionOpInterface>();
  ModuleOp parentModuleOp = parentFuncOp->getParentOfType<ModuleOp>();
  OpBuilder parentModuleBuilder(&parentModuleOp.getBody()->back());
  Location loc = encodeOp.getLoc();
  std::string executableName = "__encoding_" + std::to_string(executableId);
  auto executableOp = parentModuleBuilder.create<IREE::Stream::ExecutableOp>(
      loc, executableName);

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

  // Build the inner module and func op.
  std::string funcName = executableName + "$" + getDispatchFuncName(encodeOp);
  auto funcOp = createWorkgroupFunc(encodeOp, funcName);
  {
    OpBuilder builder(executableOp.getBody());
    auto innerModule = builder.create<mlir::ModuleOp>(loc);
    innerModule.push_back(funcOp);
  }

  executableOp.getOperation()->moveBefore(parentFuncOp);
  executableOp.setPrivate();

  // Add an export pointing at the entry point function.
  OpBuilder builder2(executableOp.getBody());
  auto exportOp = builder2.create<IREE::Stream::ExecutableExportOp>(
      loc, funcOp.getName(), SymbolRefAttr::get(funcOp));
  Block *block = builder2.createBlock(&exportOp.getWorkgroupCount(),
                                      exportOp.getWorkgroupCount().end(),
                                      workloadTypes, workloadLocs);
  builder2.setInsertionPointToStart(block);
  auto defaultCountOp =
      builder2.create<IREE::Flow::DispatchWorkgroupCountFromSliceOp>(
          loc, block->getArguments());
  builder2.create<IREE::Stream::ReturnOp>(loc, defaultCountOp.getResults());

  rewriter.setInsertionPoint(encodeOp);
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> operandOffsets = {zero};
  SmallVector<Value> operandEnds = {encodeOp.getSourceSize()};
  SmallVector<Value> operandLengths = {encodeOp.getSourceSize()};
  SmallVector<Value> operands = {encodeOp.getSource()};
  for (auto argument : encodeOp.getSourceEncodingDims()) {
    operands.push_back(argument);
  }

  // TODO(hanchung): Add the builder to Stream::DispatchTensorOp that takes
  // export op.
  StringRef executableOpSymName =
      exportOp->getParentOp()
          ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  auto entryPoint =
      SymbolRefAttr::get(rewriter.getContext(), executableOpSymName,
                         {SymbolRefAttr::get(exportOp)});

  SmallVector<int64_t> tiedArguments = {
      IREE::Util::TiedOpInterface::kUntiedIndex};

  SmallVector<Value> dynamicDims;
  for (Value argument : encodeOp.getSourceEncodingDims()) {
    dynamicDims.push_back(argument);
  }
  SmallVector<OpFoldResult> workload = getMixedValues(
      cast<RankedTensorType>(encodeOp.getSourceEncoding()).getShape(),
      dynamicDims, rewriter.getContext());

  rewriter.replaceOpWithNewOp<IREE::Stream::AsyncDispatchOp>(
      encodeOp, encodeOp.getResult().getType(),
      /*workload=*/dynamicDims, rewriter.getArrayAttr({entryPoint}), operands,
      encodeOp.getSourceSize(), operandOffsets, operandEnds, operandLengths,
      encodeOp.getResultSize(),
      /*tied_operands=*/
      cast<ArrayAttr>(rewriter.getIndexArrayAttr(tiedArguments)),
      encodeOp.getAffinityAttr());

  return executableOp;
}

void MaterializeEncodingsPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  SmallVector<IREE::Stream::TensorEncodeOp> encodeOps;
  moduleOp.walk(
      [&](IREE::Stream::TensorEncodeOp op) { encodeOps.push_back(op); });
  if (encodeOps.empty()) {
    return;
  }

  // TODO(hanchung): Cache the executables and reuse for the same encodings.
  MLIRContext *ctx = &getContext();
  IRRewriter rewriter(ctx);
  int executableId = 0;
  for (auto encodeOp : encodeOps) {
    (void)createExecutableAndEntry(rewriter, encodeOp, executableId++);
  }
}

} // namespace mlir::iree_compiler::IREE::Stream
