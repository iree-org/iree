// Copyright 2019 Google LLC
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

#include <limits>

#include "iree/base/signature_mangle.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "llvm/ADT/Optional.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

using iree::RawSignatureMangler;
using iree::SignatureBuilder;
using iree::AbiConstants::ScalarType;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

llvm::Optional<ScalarType> mapScalarType(Type elementType) {
  // Map ScalarType.
  if (elementType.isSignlessInteger()) {
    auto bits = elementType.getIntOrFloatBitWidth();
    // TODO(laurenzo): These types are still signless. Assume signed and
    // preserve once represented.
    switch (bits) {
      // We represent bools as 8-bit integers right now.
      case 1:
      case 8:
        return ScalarType::kSint8;
      case 16:
        return ScalarType::kSint16;
      case 32:
        return ScalarType::kSint32;
      case 64:
        return ScalarType::kSint64;
      default:
        return llvm::None;
    }
  } else if (auto floatType = elementType.dyn_cast<FloatType>()) {
    if (floatType.isF32())
      return ScalarType::kIeeeFloat32;
    else if (floatType.isF64())
      return ScalarType::kIeeeFloat64;
    else if (floatType.isF16())
      return ScalarType::kIeeeFloat16;
    else if (floatType.isBF16())
      return ScalarType::kGoogleBfloat16;
    else
      return llvm::None;
  }

  return llvm::None;
}

llvm::Optional<RawSignatureMangler> mangleTensorType(TensorType t) {
  auto scalarType = mapScalarType(t.getElementType());
  if (!scalarType) return llvm::None;

  llvm::SmallVector<int, 4> dims;
  for (auto typeDim : t.getShape()) {
    if (typeDim < 0)
      dims.push_back(-1);
    else if (typeDim > std::numeric_limits<int>::max())
      return llvm::None;
    else
      dims.push_back(typeDim);
  }

  RawSignatureMangler mangler;
  // Tensors map to buffers in the ABI.
  mangler.AddShapedNDBuffer(*scalarType, absl::MakeConstSpan(dims));
  return mangler;
}

llvm::Optional<RawSignatureMangler> mangleScalarType(Type t) {
  auto mappedType = mapScalarType(t);
  if (!mappedType) return llvm::None;
  RawSignatureMangler mangler;
  mangler.AddScalar(*mappedType);
  return mangler;
}

StringAttr mangleType(Builder builder, Type type, char tag) {
  SignatureBuilder fBuilder;
  auto mangledType = mangleScalarType(type);
  if (auto tensorType = type.dyn_cast<TensorType>()) {
    mangledType = mangleTensorType(tensorType);
  }
  if (!mangledType) return nullptr;
  mangledType->builder().AppendTo(fBuilder, tag);
  return builder.getStringAttr(fBuilder.encoded());
}

StringAttr unrecognizedTypeAttr(Builder builder, char tag) {
  SignatureBuilder fBuilder;
  RawSignatureMangler mangler;
  mangler.AddUnrecognized();
  mangler.builder().AppendTo(fBuilder, tag);
  return builder.getStringAttr(fBuilder.encoded());
}

}  // namespace

class MaterializeExportedReflectionPass
    : public PassWrapper<MaterializeExportedReflectionPass, FunctionPass> {
  void runOnFunction() override {
    auto func = getFunction();
    auto funcType = func.getType();
    auto builder = Builder(&getContext());

    // Only process exported functions that are not marked to omit an abi.
    if (!func->getAttr("iree.module.export")) return;
    if (func->getAttr("iree.abi.stub")) return;
    if (func->getAttr("iree.abi.none")) return;

    // Arguments.
    for (int i = 0, e = funcType.getNumInputs(); i < e; ++i) {
      auto mangled = mangleType(builder, funcType.getInput(i), 'I');
      if (!mangled) {
        func.emitWarning()
            << "Argument #" << i << " of function " << func.getName()
            << " is not a recognized public ABI type and the function"
            << " may not be invokable by standard tools";
        mangled = unrecognizedTypeAttr(builder, 'I');
      }
      NamedAttrList l(
          func.getArgAttrOfType<DictionaryAttr>(i, "iree.reflection"));
      l.set(builder.getIdentifier("f_partial"), mangled);
      func.setArgAttr(i, "iree.reflection",
                      l.getDictionary(builder.getContext()));
    }

    // Results.
    for (int i = 0, e = funcType.getNumResults(); i < e; ++i) {
      auto mangled = mangleType(builder, funcType.getResult(i), 'R');
      if (!mangled) {
        func.emitWarning()
            << "Result #" << i << " of function " << func.getName()
            << " is not a recognized public ABI type and the function"
            << " may not be invokable by standard tools";
        mangled = unrecognizedTypeAttr(builder, 'R');
      }
      NamedAttrList l(
          func.getResultAttrOfType<DictionaryAttr>(i, "iree.reflection"));
      l.set(builder.getIdentifier("f_partial"), mangled);
      func.setResultAttr(i, "iree.reflection",
                         l.getDictionary(builder.getContext()));
    }
  }
};

std::unique_ptr<OperationPass<FuncOp>> createMaterializeExportedReflection() {
  return std::make_unique<MaterializeExportedReflectionPass>();
}

static PassRegistration<MaterializeExportedReflectionPass> pass(
    "iree-flow-materialize-exported-reflection",
    "Materializes argument/result level reflection metadata for exported "
    "functions.");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
