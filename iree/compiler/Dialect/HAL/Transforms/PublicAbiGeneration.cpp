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

#include "absl/strings/string_view.h"
#include "iree/base/signature_mangle.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

using iree::RawSignatureParser;
using iree::AbiConstants::ScalarType;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

namespace {

Type mapScalarType(MLIRContext *ctx, ScalarType scalarType) {
  switch (scalarType) {
    case ScalarType::kIeeeFloat32:
      return FloatType::getF32(ctx);
    case ScalarType::kIeeeFloat64:
      return FloatType::getF64(ctx);
    case ScalarType::kIeeeFloat16:
      return FloatType::getF16(ctx);
    case ScalarType::kGoogleBfloat16:
      return FloatType::getBF16(ctx);
    case ScalarType::kSint32:
    case ScalarType::kUint32:
      return IntegerType::get(32, ctx);
    case ScalarType::kSint64:
    case ScalarType::kUint64:
      return IntegerType::get(64, ctx);
    case ScalarType::kSint16:
    case ScalarType::kUint16:
      return IntegerType::get(16, ctx);
    case ScalarType::kSint8:
    case ScalarType::kUint8:
      return IntegerType::get(8, ctx);
    default:
      return nullptr;
  }
}

LogicalResult mapRawAbiTypes(
    Location loc, SmallVectorImpl<RawSignatureParser::Description> &descs,
    SmallVectorImpl<Type> &types) {
  auto *ctx = loc.getContext();
  auto bufferViewType = HAL::BufferViewType::get(loc.getContext());
  for (auto &d : descs) {
    switch (d.type) {
      case RawSignatureParser::Type::kBuffer:
        // ABI buffers map to shape-erased ref of buffer_views.
        types.push_back(bufferViewType);
        break;
      case RawSignatureParser::Type::kRefObject: {
        // TODO(laurenzo): Map supported ref objects.
        std::string dstr;
        d.ToString(dstr);
        return emitError(loc) << "unsupported ABI type: " << dstr;
      }
      case RawSignatureParser::Type::kScalar: {
        auto t = mapScalarType(ctx, d.scalar.type);
        if (!t) {
          std::string dstr;
          d.ToString(dstr);
          return emitError(loc) << "unsupported ABI type: " << dstr;
        }
        types.push_back(t);
        break;
      }
    }
  }

  return success();
}

LogicalResult generateSynchronousBody(
    FuncOp rawCalleeFuncOp, FuncOp funcOp, OpBuilder moduleBuilder,
    SmallVectorImpl<Type> &inputTypes,
    SmallVectorImpl<RawSignatureParser::Description> &inputDescs,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<RawSignatureParser::Description> &resultDescs) {
  auto *ctx = funcOp.getContext();
  auto loc = funcOp.getLoc();
  Block *entryBlock = funcOp.addEntryBlock();
  OpBuilder builder = OpBuilder::atBlockEnd(entryBlock);

  // Build call operands.
  SmallVector<Value, 4> callOperands;
  for (const auto &input : llvm::enumerate(inputDescs)) {
    auto blockArg = entryBlock->getArgument(input.index());
    switch (input.value().type) {
      case RawSignatureParser::Type::kBuffer: {
        // Pass the backing buffer.
        // TODO(laurenzo): Validate shape.
        callOperands.push_back(
            builder.create<HAL::BufferViewBufferOp>(loc, blockArg));

        // Now, each dynamic dim is passed individually.
        for (auto dim : llvm::enumerate(input.value().dims)) {
          if (dim.value() >= 0) {
            // Static.
            continue;
          }
          // Dynamic: Get each dim individually.
          // There is an optimization potential here if more than a couple of
          // dynamic dims to use the bulk dim query op, but here just get one
          // at a time as needed.
          auto dimValue = builder.create<HAL::BufferViewDimOp>(
              loc, builder.getIndexType(), blockArg,
              builder.getI32IntegerAttr(dim.index()));
          callOperands.push_back(dimValue);
        }
        break;
      }
      case RawSignatureParser::Type::kScalar: {
        // Assume that scalars are pass-through.
        callOperands.push_back(blockArg);
        break;
      }
      case RawSignatureParser::Type::kRefObject: {
        // Assume that ref objects are pass-through.
        callOperands.push_back(blockArg);
        break;
      }
    }
  }

  // Build call.
  auto callOp = builder.create<CallOp>(loc, rawCalleeFuncOp, callOperands);

  // And convert each result. For any buffer results, this involves a
  // contraction from (buffer, index...) -> (buffer_view).
  auto callResults = callOp.getResults();
  auto callResultsIt = callResults.begin();
  SmallVector<Value, 4> funcResults;
  for (const auto &output : llvm::enumerate(resultDescs)) {
    if (callResultsIt == callResults.end()) {
      return emitError(loc)
             << "mismatched reflection metadata and function signature "
             << "(overall arity)";
    }
    Value nextCallResult = *(callResultsIt++);
    switch (output.value().type) {
      case RawSignatureParser::Type::kBuffer: {
        // Unpack dims (dynamic dims come from call result, static become
        // consts).
        SmallVector<Value, 4> dimValues;
        for (auto dim : llvm::enumerate(output.value().dims)) {
          if (dim.value() >= 0) {
            // Static.
            dimValues.push_back(
                builder.create<ConstantIndexOp>(loc, dim.value()));
          } else {
            // Dynamic.
            if (callResultsIt == callResults.end()) {
              return emitError(loc)
                     << "mismatched reflection metadata and function signature "
                     << "(dynamic dim)";
            }
            dimValues.push_back(*callResultsIt);
            ++callResultsIt;
          }
        }

        // Determine element type.
        Type mappedScalarType = mapScalarType(ctx, output.value().scalar.type);
        auto elementType = getElementTypeValue(mappedScalarType);
        if (!elementType) {
          return emitError(loc)
                 << "unsupported hal element type: " << mappedScalarType;
        }

        // Build buffer_view.
        funcResults.push_back(builder.create<BufferViewCreateOp>(
            loc, nextCallResult, dimValues, *elementType));
        break;
      }
      case RawSignatureParser::Type::kScalar: {
        // Assume that scalars are pass-through.
        funcResults.push_back(nextCallResult);
        break;
      }
      case RawSignatureParser::Type::kRefObject: {
        // Assume that ref objects are pass-through.
        funcResults.push_back(nextCallResult);
        break;
      }
    }
  }

  // Add the return.
  builder.create<mlir::ReturnOp>(loc, funcResults);
  return success();
}

LogicalResult generateAsynchronousBody(FuncOp funcOp, FuncOp syncFuncOp,
                                       OpBuilder moduleBuilder) {
  auto loc = funcOp.getLoc();
  Block *entryBlock = funcOp.addEntryBlock();
  OpBuilder builder = OpBuilder::atBlockEnd(entryBlock);

  // Wait until the wait semaphore reaches the wait value.
  auto waitSemaphore = entryBlock->getArgument(0);
  auto waitValue = entryBlock->getArgument(1);
  auto waitOp = builder.create<HAL::SemaphoreAwaitOp>(
      loc, builder.getIntegerType(32), waitSemaphore, waitValue);
  builder.create<HAL::CheckSuccessOp>(loc, waitOp.getResult(),
                                      "semaphore wait failed");

  // Trim the first (wait semaphore/value) and last (signal semaphore/value)
  // two arguments.
  SmallVector<Value, 4> callSyncArguments;
  for (int i = 2; i < entryBlock->getNumArguments() - 2; ++i) {
    callSyncArguments.push_back(entryBlock->getArguments()[i]);
  }
  // Call the sync op, passing through our arguments.
  auto callSyncOp = builder.create<CallOp>(loc, syncFuncOp, callSyncArguments);

  // Signal the signal semaphore to its signal value.
  auto signalSemaphore =
      entryBlock->getArgument(entryBlock->getNumArguments() - 2);
  auto signalValue = entryBlock->getArgument(entryBlock->getNumArguments() - 1);
  builder.create<HAL::SemaphoreSignalOp>(loc, signalSemaphore, signalValue);

  // Return results of the sync op.
  builder.create<mlir::ReturnOp>(loc, callSyncOp.getResults());

  return success();
}

LogicalResult generateRawAbiFunctions(OpBuilder &moduleBuilder,
                                      FuncOp rawCalleeFuncOp,
                                      StringRef exportName,
                                      DictionaryAttr reflection,
                                      StringRef signatureSr) {
  auto ctx = rawCalleeFuncOp.getContext();
  auto loc = rawCalleeFuncOp.getLoc();

  absl::string_view signature(signatureSr.data(), signatureSr.size());
  SmallVector<RawSignatureParser::Description, 4> inputDescs;
  SmallVector<RawSignatureParser::Description, 4> resultDescs;

  // Parse the reflection metadata.
  RawSignatureParser p;
  p.VisitInputs(signature, [&](const RawSignatureParser::Description &d) {
    inputDescs.push_back(d);
  });
  p.VisitResults(signature, [&](const RawSignatureParser::Description &d) {
    resultDescs.push_back(d);
  });
  if (p.GetError()) {
    return rawCalleeFuncOp.emitError()
           << "illegal abi signature ('" << signatureSr
           << "'): " << *p.GetError();
  }

  // Map to function signature types.
  SmallVector<Type, 4> inputTypes;
  SmallVector<Type, 4> resultTypes;
  if (failed(mapRawAbiTypes(loc, inputDescs, inputTypes))) {
    return failure();
  }
  assert(inputTypes.size() == inputDescs.size());
  if (failed(mapRawAbiTypes(loc, resultDescs, resultTypes))) {
    return failure();
  }
  assert(resultTypes.size() == resultDescs.size());

  // Create the new synchronous function export.
  SmallVector<NamedAttribute, 1> syncExportAttrs;
  syncExportAttrs.push_back(moduleBuilder.getNamedAttr(
      "iree.module.export", moduleBuilder.getStringAttr(exportName)));
  syncExportAttrs.push_back(
      moduleBuilder.getNamedAttr("iree.reflection", reflection));
  syncExportAttrs.push_back(
      moduleBuilder.getNamedAttr("iree.abi.stub", UnitAttr::get(ctx)));

  auto syncType = FunctionType::get(inputTypes, resultTypes, ctx);
  auto syncName = (rawCalleeFuncOp.getName() + "$sync").str();
  auto syncFuncOp =
      moduleBuilder.create<FuncOp>(loc, syncName, syncType, syncExportAttrs);

  if (failed(generateSynchronousBody(rawCalleeFuncOp, syncFuncOp, moduleBuilder,
                                     inputTypes, inputDescs, resultTypes,
                                     resultDescs))) {
    return failure();
  }

  // Create the new asynchronous function export.
  SmallVector<Type, 4> asyncInputTypes;
  // Prefix with wait semaphore and its value.
  // TODO(scotttodd): SemaphoreValue wrapper for single {semaphore, value}
  // TODO(scotttodd): SemaphoreList wrapper for list of SemaphoreValues
  asyncInputTypes.push_back(HAL::SemaphoreType::get(ctx));
  asyncInputTypes.push_back(moduleBuilder.getIndexType());
  for (const auto &inputType : inputTypes) {
    asyncInputTypes.push_back(inputType);
  }
  // Postfix with signal semaphore and its value.
  asyncInputTypes.push_back(HAL::SemaphoreType::get(ctx));
  asyncInputTypes.push_back(moduleBuilder.getIndexType());
  // TODO(scotttodd): populate async export attributes
  //   * iree.module.export with a name
  //   * iree.reflection (considering new args?)
  //   * iree.abi.stub
  SmallVector<NamedAttribute, 1> asyncExportAttrs;

  auto asyncType = FunctionType::get(asyncInputTypes, resultTypes, ctx);
  auto asyncName = (rawCalleeFuncOp.getName() + "$async").str();
  auto asyncFuncOp =
      moduleBuilder.create<FuncOp>(loc, asyncName, asyncType, asyncExportAttrs);

  if (failed(
          generateAsynchronousBody(asyncFuncOp, syncFuncOp, moduleBuilder))) {
    return failure();
  }

  return success();
}

LogicalResult generateAbiFunctions(FuncOp funcOp, StringRef exportName,
                                   DictionaryAttr reflection) {
  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPointAfter(funcOp);

  auto rawSignatureSpec = reflection.get("f").dyn_cast_or_null<StringAttr>();
  if (rawSignatureSpec) {
    if (failed(generateRawAbiFunctions(builder, funcOp, exportName, reflection,
                                       rawSignatureSpec.getValue()))) {
      return failure();
    }
  }

  return success();
}

Optional<StringRef> getFuncOpExportName(FuncOp op) {
  auto exportAttr = op.getAttr("iree.module.export");
  if (!exportAttr) return llvm::None;

  if (exportAttr.isa<UnitAttr>()) {
    // Just the function name.
    return op.getName();
  } else if (auto nameAttr = exportAttr.dyn_cast<StringAttr>()) {
    return nameAttr.getValue();
  }

  return llvm::None;
}

class PublicABIGenerationPass
    : public PassWrapper<PublicABIGenerationPass, OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    for (auto &op : getOperation().getBody()->getOperations()) {
      if (auto funcOp = dyn_cast<FuncOp>(op)) {
        // Skip functions we generate.
        if (funcOp.getAttr("iree.abi.stub")) continue;

        // Any function marked for export, we redirect to export with a
        // '$raw' suffix and then generate ABI wrappers with the original name.
        Optional<StringRef> exportName = getFuncOpExportName(funcOp);
        if (!exportName) continue;
        auto reflection = funcOp.getAttr("iree.reflection")
                              .dyn_cast_or_null<DictionaryAttr>();
        if (!reflection) continue;

        // Rename and remove reflection (it will go on the ABI entry point).
        funcOp.setAttr("iree.module.export",
                       StringAttr::get((*exportName + "$raw").str(), ctx));
        funcOp.removeAttr("iree.reflection");

        if (reflection) {
          if (failed(generateAbiFunctions(funcOp, *exportName, reflection))) {
            signalPassFailure();
            return;
          }
        }
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createPublicABIGenerationPass() {
  return std::make_unique<PublicABIGenerationPass>();
}

static PassRegistration<PublicABIGenerationPass> pass(
    "iree-hal-public-abi-generation", "Creates public ABI entry points");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
