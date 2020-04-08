// Copyright 2020 Google LLC
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

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Translation/CodegenPasses/Passes.h"
#include "iree/compiler/Translation/CodegenUtils/CodegenUtils.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// Converts the function call operation that takes and returns tensor
/// arguments, into one that takes memref arguments.  The number of arguments of
/// the converted op is equal to the sum of the number of arguments and number
/// of results of the original operation.
/// - Arguments that are tensor type are converted to memref type of same shape
///   and element type.
/// - Results that are tensor type are converted to memref type of same shape
///   and element type.
/// - Results that are not tensor type are converted to memref type of rank-0,
///   with the type as the element type.
/// Inserts the `iree.load_input` and `iree.store_output` instructions to allow
/// the body of the function to contain tensors operations.
struct FuncOpConversion : OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      FuncOp funcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

/// Converts a return statement to iree.store_output statements and empty
/// return.
struct ReturnOpConversion : OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ReturnOp returnOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

/// Converts the call operation to a function that is converted using the FuncOp
/// conversion pattern implemented above. Will insert hal.interface.get_memref
/// ops for the memrefs that are to be used as result buffers.
/// The called function (which is the dispatch function implementation) will be
/// annotated with
/// - spv.entry_point_abi attribute to use during SPIR-V lowering.
/// - spv.interface_var_abi attribute on function arguments to use during SPIR-V
/// lowering.
/// - iree.dispatch_fn_name attribute which contains the name of the entry point
///   function (which is not the implementation function). The generated SPIR-V
///   binary/LLVM module needs to have this function for the runtime to execute
///   the kernel.
// TODO(ravishankarm): The LLVM side doesnt need the spv.* attributes. Maybe
// make that optional.
struct CallOpConversion : OpConversionPattern<CallOp> {
  using OpConversionPattern<CallOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      CallOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

/// Pass to convert from HAL tensor interface to HAL memref interface.
struct HALInterfaceToMemrefPass
    : public PassWrapper<HALInterfaceToMemrefPass,
                         OperationPass<IREE::Flow::ExecutableOp>> {
  void runOnOperation() override;
};
}  // namespace

/// Convert a ranked tensor type to equivalent memref type.
static MemRefType convertTensorTypeOrNull(Type t) {
  auto tensorType = t.dyn_cast<RankedTensorType>();
  if (tensorType)
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  return nullptr;
}

LogicalResult FuncOpConversion::matchAndRewrite(
    FuncOp funcOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (funcOp.empty()) return failure();

  FunctionType fnType = funcOp.getType();
  Location loc = funcOp.getLoc();

  // Convert all tensor type input arguments to corresponding memref type.
  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  for (auto argType : enumerate(fnType.getInputs())) {
    MemRefType convertedType = convertTensorTypeOrNull(argType.value());
    if (!convertedType) {
      return funcOp.emitError(
          "expected dispatch function to have all tensor operands");
    }
    signatureConverter.addInputs(argType.index(), convertedType);
  }

  // Convert all tensor type output to corresponding memref type and append as
  // arguments to the new function. For non-tensor types, append a memref type
  // with the same element type and {} shape as argument to the new function.
  for (auto resultType : fnType.getResults()) {
    MemRefType convertedType = convertTensorTypeOrNull(resultType);
    if (!convertedType)
      return funcOp.emitError(
          "expected dispatch function to have all tensor return values");
    signatureConverter.addInputs(convertedType);
  }
  auto newFuncOp = rewriter.create<FuncOp>(
      loc, funcOp.getName(),
      rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                               llvm::None),
      /*attrs=*/ArrayRef<NamedAttribute>());
  rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                              newFuncOp.end());
  rewriter.applySignatureConversion(&newFuncOp.getBody(), signatureConverter);

  // For all inputs, get the tensor value back by inserting iree.load_input.
  OpBuilder::InsertionGuard insertionGuard(rewriter);
  rewriter.setInsertionPointToStart(&newFuncOp.getBody().front());
  for (auto inputType : enumerate(fnType.getInputs())) {
    Value loadInputVal = rewriter.create<IREE::LoadInputOp>(
        loc, inputType.value(), newFuncOp.getArgument(inputType.index()));
    rewriter.replaceUsesOfBlockArgument(
        newFuncOp.getArgument(inputType.index()), loadInputVal);
  }

  rewriter.eraseOp(funcOp);
  return success();
}

LogicalResult ReturnOpConversion::matchAndRewrite(
    ReturnOp returnOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto funcOp = returnOp.getParentOfType<FuncOp>();
  if (funcOp.getNumResults()) return failure();
  auto numArguments = funcOp.getNumArguments();
  auto numReturnVals = operands.size();
  auto loc = returnOp.getLoc();
  for (auto resultNum : llvm::seq<unsigned>(0, returnOp.getNumOperands())) {
    Value dst = funcOp.getArgument(numArguments - numReturnVals + resultNum);
    rewriter.create<IREE::StoreOutputOp>(loc, operands[resultNum], dst);
  }
  rewriter.replaceOpWithNewOp<ReturnOp>(returnOp);
  return success();
}

/// Map HAL descriptor type to SPIR-V storage class.
static Optional<spirv::StorageClass> getSPIRVStorageClass(
    IREE::HAL::DescriptorType descriptor) {
  switch (descriptor) {
    case IREE::HAL::DescriptorType::StorageBuffer:
      return spirv::StorageClass::StorageBuffer;
    case IREE::HAL::DescriptorType::UniformBuffer:
      return spirv::StorageClass::Uniform;
    default:
      return {};
  }
}

/// Build the spirv::InterfaceVarABIAttr for the binding associated with a
/// IREE::HAL::InterfaceLoadTensorOp/IREE::HAL::InterfaceStoreTensorOp.
template <typename InterfaceOp>
static spirv::InterfaceVarABIAttr getSPIRVInterfaceVarABIAttr(
    InterfaceOp op, MLIRContext *context) {
  SymbolRefAttr interface = op.binding();
  IREE::HAL::InterfaceBindingOp binding =
      dyn_cast_or_null<IREE::HAL::InterfaceBindingOp>(
          SymbolTable::lookupNearestSymbolFrom(op, interface));
  if (!binding) {
    op.emitError("unable to resolve binding symbol");
    return nullptr;
  }
  Optional<spirv::StorageClass> storageClass =
      getSPIRVStorageClass(binding.type());
  if (!storageClass) {
    op.emitError("unable to resolve descriptor type");
    return nullptr;
  }
  // TODO(ravishankarm, antiagainst): Setting the storage class for non-scalar
  // types using spv.interface_var_abi attr is currently an error. This needs to
  // be addressed for IREE's use case.
  if (storageClass.getValue() != spirv::StorageClass::StorageBuffer) {
    op.emitError("unable to handle descriptor type that is not StorageBuffer");
    return nullptr;
  }
  spirv::InterfaceVarABIAttr spvAttr = spirv::getInterfaceVarABIAttr(
      binding.set().getZExtValue(), binding.binding().getZExtValue(), {},
      context);
  if (!spvAttr) {
    op.emitError("unable to build spv.interface_var_abi attr");
    return nullptr;
  }
  return spvAttr;
}

LogicalResult CallOpConversion::matchAndRewrite(
    CallOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto callee = dyn_cast_or_null<FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(op, op.callee()));
  Location loc = op.getLoc();
  if (!callee)
    return op.emitError("unable to resolve function ") << op.callee();

  // Build the interface variable ABI attributes for the arguments.
  SmallVector<Value, 4> newOperands;
  SmallVector<spirv::InterfaceVarABIAttr, 4> argAttrs;
  newOperands.reserve(operands.size() + op.getNumResults());
  argAttrs.reserve(operands.size() + op.getNumResults());
  MLIRContext *context = rewriter.getContext();
  for (auto operand : operands) {
    // TODO(ravishankarm): In general we might need to traverse a DAG of
    // operations to get to the defining op. Unsure how that will work out, but
    // for now just assume that operand is going to be defined by a
    // hal.interface.load_tensor
    auto definingOp = dyn_cast_or_null<IREE::HAL::InterfaceLoadTensorOp>(
        operand.getDefiningOp());
    if (!definingOp)
      return op.emitError(
          "unable to find hal.interface. op while building "
          "spv.interface_var_abi attr");
    spirv::InterfaceVarABIAttr abiAttr =
        getSPIRVInterfaceVarABIAttr(definingOp, context);
    if (!abiAttr) return failure();
    argAttrs.push_back(abiAttr);
    MemRefType convertedType = convertTensorTypeOrNull(operand.getType());
    if (!convertedType)
      return op.emitError(
          "unable to convert type of operand to call operation");
    // TODO(ravishankarm): Convert this to use dynamically sizes allocation.
    if (!convertedType.hasStaticShape())
      return op.emitError("unable to handle dynamically sized tensors");

    // TODO(ravishankarm): We can create an alloc here cause finally only need
    // the binary that is created from this module. The alloc has no effect in
    // the end as of now. But that might change, so modify approach to account
    // for that.
    auto alloc = rewriter.create<AllocOp>(loc, convertedType);
    // Store the tensor value into the alloc.
    rewriter.create<TensorStoreOp>(loc, operand, alloc);
    newOperands.push_back(alloc);
  }

  // Build the interface varialbe ABI attributes for the result.
  SmallVector<Value, 1> resultBuffers;
  resultBuffers.reserve(op.getNumResults());
  for (Value result : op.getResults()) {
    if (!result.hasOneUse())
      return op.emitError(
          "unhandled multiple uses of result of dispatch impl function");
    // TODO(ravishankarm): In general we might need to traverse a DAG of
    // operations to get to the hal.interface.store_tensor op. Unsure how that
    // will work out, but for now just assume that the use is a
    // hal.interface.store_tensor op.
    auto storeTensorOp = dyn_cast<IREE::HAL::InterfaceStoreTensorOp>(
        result.use_begin()->getOwner());
    if (!storeTensorOp)
      return op.emitError(
          "expected result of dispatch impl function call to be used in a "
          "hal.interface.store_tensor op");
    spirv::InterfaceVarABIAttr abiAttr =
        getSPIRVInterfaceVarABIAttr(storeTensorOp, context);
    if (!abiAttr) return failure();
    argAttrs.push_back(abiAttr);
    MemRefType convertedType = convertTensorTypeOrNull(result.getType());
    if (!convertedType)
      return op.emitError("unable to convert type of result of call operation");

    // TODO(ravishankarm): Convert this to use dynamically sizes allocation.
    if (!convertedType.hasStaticShape())
      return op.emitError("unable to handle dynamically sized tensors");

    // TODO(ravishankarm): We can create an alloc here cause finally only need
    // the binary that is created from this module. The alloc has no effect in
    // the end as of now. But that might change, so modify approach to account
    // for that.
    auto alloc = rewriter.create<AllocOp>(loc, convertedType);
    newOperands.push_back(alloc);
    resultBuffers.push_back(alloc);
  }

  rewriter.create<CallOp>(loc, op.callee(), ArrayRef<Type>(), newOperands);

  // Load the tensor values from the result buffers.
  SmallVector<Value, 1> resultValues;
  resultValues.reserve(resultBuffers.size());
  for (auto buffer : resultBuffers)
    resultValues.push_back(rewriter.create<TensorLoadOp>(loc, buffer));
  rewriter.replaceOp(op, resultValues);

  // Set the entry point attribute for the callee and the interface variable ABI
  // attr for the callee arguments.
  SmallVector<int32_t, 3> localSize = {1, 1, 1};
  callee.setAttr(spirv::getEntryPointABIAttrName(),
                 spirv::getEntryPointABIAttr(localSize, context));
  for (auto argAttr : enumerate(argAttrs))
    callee.setArgAttr(argAttr.index(), spirv::getInterfaceVarABIAttrName(),
                      argAttr.value());
  // Set the name of the entry point function to use.
  callee.setAttr(
      getDispatchFuncAttrName(),
      rewriter.getStringAttr(op.getParentOfType<FuncOp>().getName()));
  return success();
}

static void populateImplFunctionConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<FuncOpConversion, ReturnOpConversion>(context);
}

static void populateHALInterfaceToMemrefPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<CallOpConversion>(context);
}

void HALInterfaceToMemrefPass::runOnOperation() {
  MLIRContext *context = &getContext();

  // Collect the dispatch functions within the flow.executable op and the
  // implementation function.
  IREE::Flow::ExecutableOp executableOp = getOperation();
  SmallVector<Operation *, 1> dispatchFns;
  llvm::SetVector<Operation *> implFns;
  SymbolTable symbolTable(executableOp.getInnerModule());
  for (auto entryOp :
       executableOp.getBlock().getOps<IREE::Flow::DispatchEntryOp>()) {
    auto dispatchFn = symbolTable.lookup<FuncOp>(entryOp.function_ref());
    if (!dispatchFn)
      entryOp.emitError("could not find dispatch function")
          << entryOp.function_ref();
    dispatchFns.push_back(dispatchFn);
    auto walkResult =
        dispatchFn.walk([&implFns, &symbolTable](CallOp callOp) -> WalkResult {
          auto implFn = symbolTable.lookup<FuncOp>(callOp.callee());
          if (!implFn) {
            callOp.emitError("unable to find definition of function ")
                << callOp.callee();
            return WalkResult::interrupt();
          }
          implFns.insert(implFn);
          return WalkResult::advance();
        });
    if (walkResult.wasInterrupted()) {
      return signalPassFailure();
    }
  }

  if (dispatchFns.empty() || implFns.empty()) return;

  // First convert all the functions that are invoked with the dispatch region
  // to operate on tensors.
  OwningRewritePatternList patterns;
  populateImplFunctionConversionPatterns(context, patterns);
  ConversionTarget implFnConversion(*context);
  implFnConversion.markUnknownOpDynamicallyLegal(
      [](Operation *op) -> bool { return true; });
  implFnConversion.addDynamicallyLegalOp<FuncOp>([](FuncOp funcOp) -> bool {
    auto fnType = funcOp.getType();
    return llvm::all_of(
               fnType.getInputs(),
               [](Type t) -> bool { return !t.isa<RankedTensorType>(); }) &&
           fnType.getNumResults() == 0;
  });
  implFnConversion.addDynamicallyLegalOp<ReturnOp>(
      [](ReturnOp returnOp) -> bool { return returnOp.getNumOperands() == 0; });
  populateHALInterfaceToMemrefPatterns(context, patterns);
  if (failed(applyFullConversion(implFns.getArrayRef(), implFnConversion,
                                 patterns, nullptr)))
    return signalPassFailure();

  // Convert the dispatch functions.
  patterns.clear();
  populateHALInterfaceToMemrefPatterns(context, patterns);
  ConversionTarget dispatchFnConversion(*context);
  dispatchFnConversion.markUnknownOpDynamicallyLegal(
      [](Operation *op) -> bool { return true; });
  dispatchFnConversion.addDynamicallyLegalOp<CallOp>([](CallOp op) -> bool {
    return llvm::all_of(
               op.getOperandTypes(),
               [](Type t) -> bool { return !t.isa<RankedTensorType>(); }) &&
           op.getNumResults() == 0;
  });
  if (failed(applyFullConversion(dispatchFns, dispatchFnConversion, patterns,
                                 nullptr)))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<IREE::Flow::ExecutableOp>>
createHALInterfaceToMemrefPass() {
  return std::make_unique<HALInterfaceToMemrefPass>();
}

static PassRegistration<HALInterfaceToMemrefPass> pass(
    "iree-convert-hal-interface-to-memref",
    "Converts the HAL interface to use memrefs (instead of tensors)");

}  // namespace iree_compiler
}  // namespace mlir
