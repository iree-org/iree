// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <limits>

#include "iree/compiler/Bindings/SIP/Transforms/Passes.h"
#include "iree/compiler/Bindings/SIP/Utils/SignatureBuilder.h"
#include "llvm/ADT/Optional.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace SIP {

using AbiConstants::ScalarType;

static llvm::Optional<ScalarType> mapScalarType(Type elementType) {
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
    if (floatType.isF32()) {
      return ScalarType::kIeeeFloat32;
    } else if (floatType.isF64()) {
      return ScalarType::kIeeeFloat64;
    } else if (floatType.isF16()) {
      return ScalarType::kIeeeFloat16;
    } else if (floatType.isBF16()) {
      return ScalarType::kGoogleBfloat16;
    } else {
      return llvm::None;
    }
  }
  return llvm::None;
}

static LogicalResult mangleTensorType(TensorType t,
                                      RawSignatureMangler &mangler) {
  auto scalarType = mapScalarType(t.getElementType());
  if (!scalarType) return failure();

  llvm::SmallVector<int, 4> dims;
  for (auto typeDim : t.getShape()) {
    if (typeDim < 0) {
      dims.push_back(-1);
    } else if (typeDim > std::numeric_limits<int>::max()) {
      return failure();
    } else {
      dims.push_back(typeDim);
    }
  }

  // Tensors map to buffers in the ABI.
  mangler.AddShapedNDBuffer(*scalarType, dims);
  return success();
}

static LogicalResult mangleScalarType(Type t, RawSignatureMangler &mangler) {
  auto mappedType = mapScalarType(t);
  if (!mappedType) return failure();
  mangler.AddScalar(*mappedType);
  return success();
}

static LogicalResult mangleType(Type type, RawSignatureMangler &mangler) {
  if (auto tensorType = type.dyn_cast<TensorType>()) {
    return mangleTensorType(tensorType, mangler);
  }
  return mangleScalarType(type, mangler);
}

class MaterializeReflectionAttrsPass
    : public PassWrapper<MaterializeReflectionAttrsPass, FunctionPass> {
  void runOnFunction() override {
    auto func = getFunction();
    auto funcType = func.getType();
    auto builder = Builder(&getContext());

    // Only process exported functions that are not marked to omit an abi.
    if (!func->getAttr("iree.module.export")) return;
    if (func->getAttr("iree.abi.stub")) return;
    if (func->getAttr("iree.abi.none")) return;

    // Arguments.
    RawSignatureMangler inputsMangler;
    for (int i = 0, e = funcType.getNumInputs(); i < e; ++i) {
      if (failed(mangleType(funcType.getInput(i), inputsMangler))) {
        func.emitWarning()
            << "Argument #" << i << " of function " << func.getName()
            << " is not a recognized public ABI type and the function"
            << " may not be invokable by standard tools";
        inputsMangler.AddUnrecognized();
      }
    }

    // Results.
    RawSignatureMangler resultsMangler;
    for (int i = 0, e = funcType.getNumResults(); i < e; ++i) {
      if (failed(mangleType(funcType.getResult(i), resultsMangler))) {
        func.emitWarning()
            << "Result #" << i << " of function " << func.getName()
            << " is not a recognized public ABI type and the function"
            << " may not be invokable by standard tools";
        resultsMangler.AddUnrecognized();
      }
    }

    // Update the function level attribute.
    auto reflectionIdent = builder.getIdentifier("iree.reflection");
    auto fIdent = builder.getIdentifier("f");
    auto fVersionIdent = builder.getIdentifier("fv");
    SignatureBuilder functionSignature =
        RawSignatureMangler::ToFunctionSignature(inputsMangler, resultsMangler);
    NamedAttrList l(func->getAttrOfType<DictionaryAttr>(reflectionIdent));
    l.set(fIdent, builder.getStringAttr(functionSignature.encoded()));
    l.set(fVersionIdent, builder.getStringAttr("1"));
    func->setAttr(reflectionIdent, l.getDictionary(&getContext()));
  }
};

std::unique_ptr<OperationPass<FuncOp>> createMaterializeReflectionAttrsPass() {
  return std::make_unique<MaterializeReflectionAttrsPass>();
}

static PassRegistration<MaterializeReflectionAttrsPass> pass(
    "iree-sip-materialize-reflection-attrs",
    "Materializes argument/result level reflection metadata for exported "
    "functions.");

}  // namespace SIP
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
