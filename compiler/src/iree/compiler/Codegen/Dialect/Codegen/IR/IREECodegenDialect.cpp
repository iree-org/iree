// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp.inc"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir::iree_compiler::IREE::Codegen {

struct IREECodegenDialectOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (llvm::isa<TranslationInfoAttr>(attr)) {
      os << "translation";
      return AliasResult::OverridableAlias;
    } else if (llvm::isa<CompilationInfoAttr>(attr)) {
      os << "compilation";
      return AliasResult::OverridableAlias;
    } else if (llvm::isa<LoweringConfigAttr>(attr)) {
      os << "config";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

void IREECodegenDialect::initialize() {
  initializeCodegenAttrs();
  addInterfaces<IREECodegenDialectOpAsmInterface>();

  addOperations<
#define GET_OP_LIST
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.cpp.inc"
      >();
}

LogicalResult
IREECodegenDialect::verifyOperationAttribute(Operation *op,
                                             NamedAttribute attribute) {
  StringRef symbol = attribute.getName().strref();
  Attribute attr = attribute.getValue();
  // This function verifies the validity of a specific operation attribute.
  // - If the attribute's name matches `kTuningDefaultSpecAttrName`, make
  //   sure it contains a single named sequence op with name `__kernel_config`.
  // - If the attribute's name matches `kTuningSpecEntrypointAttrName`
  // ("iree_codegen.tuning_spec_entrypoint"):
  //   1. The attribute value must be a UnitAttr.
  //   2. If the operation is a transform::NamedSequenceOp:
  //      - The operation's function signature must satisfy the following:
  //         a. It must have exactly one result type, and the result must be of
  //         type `transform::AnyOpType`.
  //         b. It must have exactly one argument type, and the argument must be
  //         of type `transform::AnyOpType`.

  if (symbol == kTuningSpecDefaultEntrypointAttrName) {
    if (auto moduleOp = dyn_cast<ModuleOp>(op)) {
      if (!llvm::any_of(moduleOp.getOps(), [](auto &op) {
            if (auto namedSeqOp = dyn_cast<transform::NamedSequenceOp>(&op)) {
              return SymbolTable::getSymbolName(namedSeqOp).getValue() ==
                     kKernelConfigSpecName;
            }
            return false;
          })) {
        return moduleOp.emitError()
               << "The tuning specification must include a named "
                  "sequence with the symbol name '"
               << kKernelConfigSpecName << "'.";
      }
    }
  }

  if (symbol != kTuningSpecEntrypointAttrName)
    return success();

  if (!isa<UnitAttr>(attr)) {
    return op->emitError("'") << symbol << "' attribute must be a UnitAttr";
  }

  if (auto namedSeqOp = dyn_cast<transform::NamedSequenceOp>(op)) {
    ArrayRef<Type> resTypes = namedSeqOp.getFunctionType().getResults();
    if (resTypes.size() != 1 || !isa<transform::AnyOpType>(resTypes[0])) {
      return namedSeqOp.emitError()
             << "Tuning spec entry point expected to return any_op";
    }

    ArrayRef<Type> argTypes = namedSeqOp.getArgumentTypes();
    if (argTypes.size() != 1 || !isa<transform::AnyOpType>(argTypes[0])) {
      return namedSeqOp.emitError()
             << "Tuning spec entry point expected to have a "
                "single any_op argument";
    }
  }

  return success();
}

} // namespace mlir::iree_compiler::IREE::Codegen
