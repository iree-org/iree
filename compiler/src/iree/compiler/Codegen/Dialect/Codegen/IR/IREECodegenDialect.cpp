// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/DialectImplementation.h"

// clang-format off
#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp.inc" // IWYU pragma: export
// clang-format on

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

  addTypes<IREE::Codegen::NullPointerType>();
}

LogicalResult
IREECodegenDialect::verifyOperationAttribute(Operation *op,
                                             NamedAttribute attribute) {
  StringRef symbol = attribute.getName().strref();
  Attribute attr = attribute.getValue();
  // This function verifies the validity of a specific operation attribute.
  // - If the attribute's name matches kTuningSpecDefaultEntrypointAttrName
  // (`iree_codegen.tuning_spec_with_default_entrypoint`):
  //   1. Ensure that the module contains a single named sequence operation with
  //   the name `__kernel_config`.
  //   2. Verify that this `__kernel_config` named sequence operation has the
  //   attribute `iree_codegen.tuning_spec_entrypoint`.
  //   3. Ensure that the named sequence operation contains exactly **one**
  //   `ForeachMatchOp`.
  //   4. Ensure that only one named sequence operation with the
  //   `iree_codegen.tuning_spec_entrypoint` attribute.
  // - If the attribute's name matches `kTuningSpecEntrypointAttrName`
  // (`iree_codegen.tuning_spec_entrypoint`):
  //   1. The attribute value must be a UnitAttr.
  //   2. If the operation is a transform::NamedSequenceOp:
  //      - The operation's function signature must satisfy the following:
  //         a. It must have exactly one result type, and the result must be
  //         of type `transform::AnyOpType`. b. It must have exactly one
  //         argument type, and the argument must be of type
  //         `transform::AnyOpType`.

  if (symbol == kTuningSpecDefaultEntrypointAttrName) {
    if (auto moduleOp = dyn_cast<ModuleOp>(op)) {
      auto kernelConfigOpIt =
          llvm::find_if(moduleOp.getOps<transform::NamedSequenceOp>(),
                        [](transform::NamedSequenceOp op) {
                          return op.getName() == kKernelConfigSpecName;
                        });

      if (kernelConfigOpIt ==
          moduleOp.getOps<transform::NamedSequenceOp>().end()) {
        return moduleOp.emitError()
               << "The tuning specification must include a named sequence with "
                  "the symbol name '"
               << kKernelConfigSpecName << "'.";
      }

      transform::NamedSequenceOp kernelConfigOp = *kernelConfigOpIt;

      // Verify that the kernelConfigOp has the attribute
      // `iree_codegen.tuning_spec_entrypoint`.
      if (!kernelConfigOp->hasAttr(kTuningSpecEntrypointAttrName)) {
        return kernelConfigOp.emitError()
               << "The named sequence '" << kKernelConfigSpecName
               << "' must have the attribute '" << kTuningSpecEntrypointAttrName
               << "'.";
      }

      auto tuningSpecOps = llvm::filter_to_vector(
          moduleOp.getOps<transform::NamedSequenceOp>(),
          [](transform::NamedSequenceOp op) {
            return op->hasAttr(kTuningSpecEntrypointAttrName);
          });

      if (tuningSpecOps.size() != 1) {
        return moduleOp.emitError()
               << "Expected exactly one NamedSequenceOp with the attribute '"
               << kTuningSpecEntrypointAttrName << "', but found "
               << tuningSpecOps.size() << ".";
      }

      // Ensure there is exactly one ForeachMatchOp inside the kernelConfigOp.
      auto foreachMatchOps =
          llvm::to_vector(kernelConfigOp.getOps<transform::ForeachMatchOp>());
      if (foreachMatchOps.size() != 1) {
        return kernelConfigOp.emitError()
               << "The named sequence '" << kKernelConfigSpecName
               << "' must contain exactly one 'ForeachMatchOp', but found "
               << foreachMatchOps.size() << ".";
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
