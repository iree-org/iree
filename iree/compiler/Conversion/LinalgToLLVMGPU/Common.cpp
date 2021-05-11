// Copyright 2021 Nod Labs
// Copyright 2021 Google LLC
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

#include "iree/compiler/Conversion/LinalgToLLVMGPU/Common.h"

namespace mlir {
namespace iree_compiler {

LogicalResult ConvertFunc::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto funcOp = cast<FuncOp>(op);
  FunctionType fnType = funcOp.getType();
  (void)fnType;
  if (!funcOp.isPublic()) return failure();

  // illegal FuncOp must have 0 inputs.
  assert(fnType.getNumInputs() == 0 && fnType.getNumResults() == 0);

  TypeConverter::SignatureConversion signatureConverter(/*numOrigInputs=*/0);
  SmallVector<Type, 8> llvmInputTypes;
  funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp input) {
    auto memrefType = input.getType().cast<MemRefType>();
    Type elType = memrefType.getElementType();
    auto llvmType =
        LLVM::LLVMPointerType::get(elType, memrefType.getMemorySpaceAsInt());
    llvmInputTypes.push_back(llvmType);
  });
  signatureConverter.addInputs(llvmInputTypes);

  // Construct newFunc with all attributes except return type & symbol name.
  SmallVector<NamedAttribute, 4> funcAttrs;
  for (auto attr : funcOp->getAttrs()) {
    if (attr.first == SymbolTable::getSymbolAttrName() ||
        attr.first == mlir::function_like_impl::getTypeAttrName()) {
      continue;
    }
    funcAttrs.push_back(attr);
  }

  auto llvmFuncType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(rewriter.getContext()), llvmInputTypes);
  auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
      funcOp.getLoc(), funcOp.getName(), llvmFuncType,
      LLVM::Linkage::External, funcAttrs);

  // Copy all of funcOp's operations into newFuncOp's body and perform region
  // type conversion.
  rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                              newFuncOp.end());
  if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                         &signatureConverter)))
    return failure();

  rewriter.eraseOp(funcOp);
  return success();
}

LogicalResult ConvertIREEBindingOp::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // Bail until nested under an LLVMFuncOp.
  auto llvmFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
  if (!llvmFuncOp) return failure();
  assert(llvmFuncOp.getNumArguments() > 0);

  Location loc = op->getLoc();
  auto ireeBindingOp = cast<IREE::HAL::InterfaceBindingSubspanOp>(op);
  IREE::HAL::InterfaceBindingSubspanOpAdaptor adaptor(operands);
  MemRefType memrefType =
      ireeBindingOp.getResult().getType().dyn_cast<MemRefType>();

  // Fetch the interface binding op and extract the buffer index from void**.
  auto symbol = SymbolTable::lookupNearestSymbolFrom(
      op, op->getAttrOfType<SymbolRefAttr>("binding"));
  auto interfaceBindingOp = cast<IREE::HAL::InterfaceBindingOp>(symbol);
  Value llvmBufferBasePtr =
      llvmFuncOp.getArgument(interfaceBindingOp.binding().getZExtValue());
  if (memrefType.hasStaticShape()) {
    auto desc = MemRefDescriptor::fromStaticShape(
        rewriter, loc, *getTypeConverter(), memrefType, llvmBufferBasePtr);
    rewriter.replaceOp(op, {desc});
  } else {
    // TODO: pull those paramters from HAL constants.
    assert(0 && "TODO: implement dynamic shape");
  }

  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
