// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Materializes a concrete tuning assignment for an
// `iree_codegen.smt.constraints` op. The common code only substitutes SMT knob
// leaves with assigned values; each pipeline owns the mechanical repackaging of
// selected outputs such as compilation_info.

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/SMTConstraintUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AttrTypeSubElements.h"

#include <optional>

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TESTMATERIALIZESMTCONSTRAINTASSIGNMENTSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

using IREE::Codegen::CompilationInfoAttr;
using IREE::Codegen::ConstraintsOp;
using IREE::Codegen::IntKnobAttr;
using IREE::Codegen::OneOfKnobAttr;
using IREE::Codegen::PipelineAttrInterface;

static InFlightDiagnostic emitMaterializationError(ConstraintsOp op,
                                                   StringRef attrName) {
  InFlightDiagnostic diag = op.emitError("failed to materialize ");
  diag << attrName << " from constraints: ";
  return diag;
}

// Recursively substitutes knob attributes in `attr` using `assignments`.
// Returns the materialized attribute tree, or null after emitting a diagnostic.
static Attribute
materializeKnobAttribute(ConstraintsOp op, Attribute attr,
                         const DenseMap<StringRef, int64_t> &assignments,
                         StringRef attrName) {
  MLIRContext *ctx = op.getContext();
  bool failedToMaterialize = false;
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](IntKnobAttr knob) -> std::optional<Attribute> {
    StringRef name = knob.getName().getValue();
    auto it = assignments.find(name);
    if (it == assignments.end()) {
      emitMaterializationError(op, attrName)
          << "missing assignment for knob '" << name << "'";
      failedToMaterialize = true;
      return Attribute();
    }
    return IntegerAttr::get(IntegerType::get(ctx, 64), it->second);
  });
  replacer.addReplacement([&](OneOfKnobAttr knob) -> std::optional<Attribute> {
    StringRef name = knob.getName().getValue();
    auto it = assignments.find(name);
    if (it == assignments.end()) {
      emitMaterializationError(op, attrName)
          << "missing assignment for knob '" << name << "'";
      failedToMaterialize = true;
      return Attribute();
    }
    ArrayAttr options = knob.getOptions();
    int64_t index = it->second;
    if (index < 0 || index >= static_cast<int64_t>(options.size())) {
      emitMaterializationError(op, attrName)
          << "assignment for knob '" << name << "' is out of range: " << index
          << " is not in [0, " << options.size() << ")";
      failedToMaterialize = true;
      return Attribute();
    }
    return options[index];
  });

  Attribute materialized = replacer.replace(attr);
  if (failedToMaterialize || !materialized) {
    return {};
  }
  return materialized;
}

static FailureOr<Attribute>
materializeConfigurationAttr(ConstraintsOp op, StringRef attrName,
                             const DenseMap<StringRef, int64_t> &assignments) {
  PipelineAttrInterface pipeline = op.getPipeline();
  auto emitError = [op, attrName]() -> InFlightDiagnostic {
    return emitMaterializationError(op, attrName);
  };
  auto materializeAttr =
      [op, attrName, &assignments](Attribute attr) -> FailureOr<Attribute> {
    if (Attribute materialized =
            materializeKnobAttribute(op, attr, assignments, attrName)) {
      return materialized;
    }
    return failure();
  };

  // This helper intentionally only substitutes knob leaves and repackages the
  // resulting template. Constraint satisfaction is checked by the verifier/SMT
  // solver path, not by materialization.
  return pipeline.materializeConfigurationAttr(attrName, op.getKnobs(),
                                               materializeAttr, emitError);
}

static FailureOr<DenseMap<StringRef, int64_t>>
getTestAssignments(ConstraintsOp op) {
  DenseMap<StringRef, int64_t> assignments;
  Attribute assignmentsAttr = op->getAttr("test.assignments");
  if (!assignmentsAttr) {
    return assignments;
  }
  auto dict = dyn_cast<DictionaryAttr>(assignmentsAttr);
  if (!dict) {
    op.emitError("expected 'test.assignments' to be a dictionary");
    return failure();
  }
  assignments.reserve(dict.size());
  for (NamedAttribute entry : dict) {
    auto value = dyn_cast<IntegerAttr>(entry.getValue());
    if (!value) {
      op.emitError("expected test assignment '")
          << entry.getName().getValue() << "' to be an integer";
      return failure();
    }
    assignments[entry.getName().getValue()] = value.getInt();
  }
  return assignments;
}

struct TestMaterializeSMTConstraintAssignmentsPass final
    : impl::TestMaterializeSMTConstraintAssignmentsPassBase<
          TestMaterializeSMTConstraintAssignmentsPass> {
  using Base::Base;

  void runOnOperation() override {
    WalkResult result = getOperation()->walk([&](ConstraintsOp op) {
      FailureOr<DenseMap<StringRef, int64_t>> assignmentMap =
          getTestAssignments(op);
      if (failed(assignmentMap)) {
        return WalkResult::interrupt();
      }
      FailureOr<CompilationInfoAttr> compilationInfo =
          materializeCompilationInfoFromConstraints(op, *assignmentMap);
      if (failed(compilationInfo)) {
        return WalkResult::interrupt();
      }
      op->setAttr("test.materialized_compilation_info", *compilationInfo);
      op->removeAttr("test.assignments");
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

FailureOr<CompilationInfoAttr> materializeCompilationInfoFromConstraints(
    ConstraintsOp op, const DenseMap<StringRef, int64_t> &assignments) {
  FailureOr<Attribute> output = materializeConfigurationAttrFromConstraints(
      op, kCompilationInfoOutputName, assignments);
  if (failed(output)) {
    return failure();
  }
  return cast<CompilationInfoAttr>(*output);
}

FailureOr<Attribute> materializeConfigurationAttrFromConstraints(
    ConstraintsOp op, StringRef attrName,
    const DenseMap<StringRef, int64_t> &assignments) {
  return materializeConfigurationAttr(op, attrName, assignments);
}

} // namespace mlir::iree_compiler
