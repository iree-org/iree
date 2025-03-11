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

static LogicalResult validateTuningEntrypoint(
    ModuleOp moduleOp, transform::NamedSequenceOp &kernelConfigOp,
    int &numTuningEntryPoints, const std::string &kRequiredByDefaultAttr) {
  for (Operation &op : moduleOp.getBody()->getOperations()) {
    if (auto namedSeqOp = dyn_cast<transform::NamedSequenceOp>(&op)) {
      if (namedSeqOp.getName() == kKernelConfigSpecName) {
        kernelConfigOp = namedSeqOp;
      }
    }

    if (op.hasAttr(kTuningSpecEntrypointAttrName)) {
      ++numTuningEntryPoints;
    }
  }

  if (!kernelConfigOp) {
    return moduleOp->emitError()
           << "Missing named sequence '" << kKernelConfigSpecName << "'"
           << kRequiredByDefaultAttr;
  }

  // Verify that the kernelConfigOp has the attribute
  // `iree_codegen.tuning_spec_entrypoint`.
  if (!kernelConfigOp->hasAttr(kTuningSpecEntrypointAttrName)) {
    return kernelConfigOp.emitError()
           << "Missing attribute '" << kTuningSpecEntrypointAttrName
           << "' in named sequence '" << kKernelConfigSpecName << "'"
           << kRequiredByDefaultAttr;
  }

  if (numTuningEntryPoints != 1) {
    return moduleOp.emitError()
           << "Expected one named sequence with '"
           << kTuningSpecEntrypointAttrName << "', but found "
           << numTuningEntryPoints << kRequiredByDefaultAttr;
  }

  return success();
}

LogicalResult
validateKernelConfigContents(transform::NamedSequenceOp kernelConfigOp,
                             const std::string &kRequiredByDefaultAttr) {
  transform::ForeachMatchOp foreachMatchOp;
  bool hasYieldOp = false;
  for (Block &block : kernelConfigOp.getBlocks()) {
    for (Operation &op : block) {
      if (auto foreachOp = dyn_cast<transform::ForeachMatchOp>(op)) {
        if (foreachMatchOp) {
          return kernelConfigOp.emitError()
                 << "'" << kKernelConfigSpecName
                 << "' must contain exactly one 'ForeachMatchOp'"
                 << kRequiredByDefaultAttr;
        }
        foreachMatchOp = foreachOp;
      } else if (isa<transform::YieldOp>(op)) {
        if (hasYieldOp) {
          return kernelConfigOp.emitError()
                 << "'" << kKernelConfigSpecName
                 << "' must contain exactly one 'transform::YieldOp'"
                 << kRequiredByDefaultAttr;
        }
        hasYieldOp = true;
      } else {
        return kernelConfigOp.emitError()
               << "Unexpected op '" << op.getName() << "' in '"
               << kKernelConfigSpecName << "'" << kRequiredByDefaultAttr;
      }
    }
  }

  // Ensure both required operations are present.
  if (!foreachMatchOp) {
    return kernelConfigOp.emitError()
           << "Missing 'ForeachMatchOp' in '" << kKernelConfigSpecName << "'"
           << kRequiredByDefaultAttr;
  }

  if (!hasYieldOp) {
    return kernelConfigOp.emitError()
           << "Missing 'transform::YieldOp' in '" << kKernelConfigSpecName
           << "'" << kRequiredByDefaultAttr;
  }

  if (foreachMatchOp.getRestrictRootAttr()) {
    return foreachMatchOp.emitError()
           << "'ForeachMatchOp' must not have 'restrict_root' attribute"
           << kRequiredByDefaultAttr;
  }

  if (foreachMatchOp.getFlattenResultsAttr()) {
    return foreachMatchOp.emitError()
           << "'ForeachMatchOp' must not have 'flatten_results' attribute"
           << kRequiredByDefaultAttr;
  }

  Type anyOpType = transform::AnyOpType::get(kernelConfigOp.getContext());
  SmallVector<Type> argTypes(foreachMatchOp.getOperandTypes());
  // Ensure the operation has exactly one argument of type any_op.
  if (argTypes.size() != 1 || argTypes.front() != anyOpType) {
    return foreachMatchOp.emitError()
           << "'ForeachMatchOp' must take exactly one 'any_op' argument"
           << kRequiredByDefaultAttr;
  }

  SmallVector<Type> resultTypes(foreachMatchOp.getResultTypes());
  // Ensure the operation has exactly one result of type any_op.
  if (resultTypes.size() != 1 || resultTypes.front() != anyOpType) {
    return foreachMatchOp.emitError()
           << "'ForeachMatchOp' must return exactly one 'any_op' result"
           << kRequiredByDefaultAttr;
  }

  return success();
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
  //      - ForeachMatchOp must not have `flatten_results` and `restrict_root`
  //        attributes.
  //      - ForeachMatchOp must have exactly one argument of type any_op.
  //      - ForeachMatchOp must have exactly one result of type any_op.
  //   4. Ensure that only one named sequence operation with the
  //   `iree_codegen.tuning_spec_entrypoint` attribute.
  // - If the attribute's name matches `kTuningSpecEntrypointAttrName`
  // (`iree_codegen.tuning_spec_entrypoint`):
  //   1. The attribute value must be a UnitAttr.
  //   2. If the operation is a transform::NamedSequenceOp:
  //      - The operation's function signature must satisfy the following:
  //         a. It must have exactly one result type, and the result must be
  //         of type `transform::AnyOpType`.
  //         b. It must have exactly one argument type, and the argument must be
  //         of type `transform::AnyOpType`.

  if (symbol == kTuningSpecDefaultEntrypointAttrName) {
    const std::string kRequiredByDefaultAttr =
        " (required by '" + std::string(kTuningSpecDefaultEntrypointAttrName) +
        "').";
    if (auto moduleOp = dyn_cast<ModuleOp>(op)) {
      transform::NamedSequenceOp kernelConfigOp;
      int numTuningEntryPoints = 0;
      if (failed(validateTuningEntrypoint(moduleOp, kernelConfigOp,
                                          numTuningEntryPoints,
                                          kRequiredByDefaultAttr))) {
        return failure();
      }

      if (failed(validateKernelConfigContents(kernelConfigOp,
                                              kRequiredByDefaultAttr))) {
        return failure();
      }
    }
  }

  if (symbol != kTuningSpecEntrypointAttrName)
    return success();

  const std::string kRequiredByEntrypoint =
      " (required by '" + std::string(kTuningSpecEntrypointAttrName) + "').";
  if (!isa<UnitAttr>(attr)) {
    return op->emitError("'") << symbol << "' attribute must be a UnitAttr";
  }

  if (auto namedSeqOp = dyn_cast<transform::NamedSequenceOp>(op)) {
    ArrayRef<Type> resTypes = namedSeqOp.getFunctionType().getResults();
    if (resTypes.size() != 1 || !isa<transform::AnyOpType>(resTypes[0])) {
      return namedSeqOp.emitError()
             << "Must return one 'any_op'" << kRequiredByEntrypoint;
    }

    ArrayRef<Type> argTypes = namedSeqOp.getArgumentTypes();
    if (argTypes.size() != 1 || !isa<transform::AnyOpType>(argTypes[0])) {
      return namedSeqOp.emitError()
             << "Must take one 'any_op'" << kRequiredByEntrypoint;
    }
  }

  return success();
}

} // namespace mlir::iree_compiler::IREE::Codegen
