// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Materializes a concrete tuning assignment for an
// `iree_codegen.smt.constraints` op. The pipeline merges existing dispatch
// config with explicit knob assignments before substitution. Each pipeline
// repackages the materialized knob dictionary into its concrete
// compilation_info attr.

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TESTMATERIALIZESMTCONSTRAINTASSIGNMENTSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

using IREE::Codegen::CompilationInfoAttr;
using IREE::Codegen::ConstraintsOp;
using IREE::Codegen::IntKnobAttr;
using IREE::Codegen::OneOfKnobAttr;
using IREE::Codegen::PipelineAttrInterface;

static DictionaryAttr
knobAssignmentsToDict(MLIRContext *ctx,
                      const DenseMap<StringRef, int64_t> &assignments) {
  Builder b(ctx);
  SmallVector<NamedAttribute> entries;
  entries.reserve(assignments.size());
  for (const auto &entry : assignments) {
    entries.emplace_back(b.getStringAttr(entry.first),
                         b.getI64IntegerAttr(entry.second));
  }
  return DictionaryAttr::get(ctx, entries);
}

static DenseMap<StringRef, int64_t>
knobAssignmentsFromDict(DictionaryAttr assignments) {
  DenseMap<StringRef, int64_t> result;
  result.reserve(assignments.size());
  for (NamedAttribute entry : assignments) {
    auto value = dyn_cast<IntegerAttr>(entry.getValue());
    if (!value) {
      continue;
    }
    result[entry.getName().getValue()] = value.getInt();
  }
  return result;
}

static InFlightDiagnostic emitMaterializationError(ConstraintsOp op) {
  return op.emitError(
      "failed to materialize compilation_info from constraints: ");
}

// Recursively substitutes knob attributes in `attr` using `assignments`.
// Returns the materialized attribute tree, or null after emitting a diagnostic.
static Attribute
materializeKnobAttribute(ConstraintsOp op, Attribute attr,
                         const DenseMap<StringRef, int64_t> &assignments) {
  MLIRContext *ctx = op.getContext();
  return llvm::TypeSwitch<Attribute, Attribute>(attr)
      .Case([&](IntKnobAttr knob) -> Attribute {
        StringRef name = knob.getName().getValue();
        auto it = assignments.find(name);
        if (it == assignments.end()) {
          emitMaterializationError(op)
              << "missing assignment for knob '" << name << "'";
          return {};
        }
        return IntegerAttr::get(IntegerType::get(ctx, 64), it->second);
      })
      .Case([&](OneOfKnobAttr knob) -> Attribute {
        StringRef name = knob.getName().getValue();
        auto it = assignments.find(name);
        if (it == assignments.end()) {
          emitMaterializationError(op)
              << "missing assignment for knob '" << name << "'";
          return {};
        }
        ArrayAttr options = knob.getOptions();
        int64_t index = it->second;
        if (index < 0 || index >= static_cast<int64_t>(options.size())) {
          emitMaterializationError(op)
              << "assignment for knob '" << name
              << "' is out of range: " << index << " is not in [0, "
              << options.size() << ")";
          return {};
        }
        return options[index];
      })
      .Case([&](ArrayAttr array) -> Attribute {
        SmallVector<Attribute> materialized;
        materialized.reserve(array.size());
        for (Attribute element : array) {
          Attribute materializedElement =
              materializeKnobAttribute(op, element, assignments);
          if (!materializedElement) {
            return {};
          }
          materialized.push_back(materializedElement);
        }
        return ArrayAttr::get(ctx, materialized);
      })
      .Case([&](DictionaryAttr dict) -> Attribute {
        SmallVector<NamedAttribute> materialized;
        materialized.reserve(dict.size());
        for (NamedAttribute entry : dict) {
          Attribute materializedValue =
              materializeKnobAttribute(op, entry.getValue(), assignments);
          if (!materializedValue) {
            return {};
          }
          materialized.emplace_back(entry.getName(), materializedValue);
        }
        return DictionaryAttr::get(ctx, materialized);
      })
      .Default([](Attribute attr) { return attr; });
}

static DictionaryAttr
materializeKnobsDictionary(ConstraintsOp op,
                           const DenseMap<StringRef, int64_t> &assignments) {
  if (Attribute materialized =
          materializeKnobAttribute(op, op.getKnobsAttr(), assignments)) {
    return cast<DictionaryAttr>(materialized);
  }
  return {};
}

static FailureOr<CompilationInfoAttr>
materializeCompilationInfo(ConstraintsOp op, DictionaryAttr materializedKnobs) {
  PipelineAttrInterface pipeline = op.getPipeline();
  auto emitError = [op]() -> InFlightDiagnostic {
    return emitMaterializationError(op);
  };

  // This helper intentionally only substitutes knob leaves and repackages the
  // resulting template. Constraint satisfaction is checked by the verifier/SMT
  // solver path, not by materialization.
  FailureOr<Attribute> attr =
      pipeline.materializeCompilationInfo(materializedKnobs, emitError);
  if (failed(attr)) {
    return failure();
  }
  return cast<CompilationInfoAttr>(*attr);
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
  PipelineAttrInterface pipeline = op.getPipeline();
  DictionaryAttr assignmentsDict =
      knobAssignmentsToDict(op.getContext(), assignments);
  DictionaryAttr effectiveDict =
      pipeline.mergeKnobAssignmentsForMaterialization(op.getOperation(),
                                                      assignmentsDict);
  DenseMap<StringRef, int64_t> effectiveAssignments =
      knobAssignmentsFromDict(effectiveDict);

  if (DictionaryAttr materializedKnobs =
          materializeKnobsDictionary(op, effectiveAssignments)) {
    // The merge here only overlays at the flat top level; nested
    // `lowering_config` / `translation_info` templates pass through as the
    // sole source of truth.
    DictionaryAttr mergedKnobs =
        pipeline.mergeMaterializedKnobsForMaterialization(op.getOperation(),
                                                          materializedKnobs);
    return materializeCompilationInfo(op, mergedKnobs);
  }
  return failure();
}

} // namespace mlir::iree_compiler
