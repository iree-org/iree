// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_UTIL_PASSES
#define IREE_DIALECT_UTIL_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Optimization and cleanup
//===----------------------------------------------------------------------===//

def ApplyPatterns : Pass<"iree-util-apply-patterns", ""> {
  let summary = "Applies some risky/IREE-specific canonicalization patterns.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createApplyPatternsPass()
  }];
}

def CombineInitializers : Pass<"iree-util-combine-initializers", "mlir::ModuleOp"> {
  let summary = "Combines global initializers into one.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createCombineInitializersPass()
  }];
}

def DropCompilerHints : Pass<"iree-util-drop-compiler-hints", ""> {
  let summary = "Deletes operations that have no runtime equivalent.";
  let description = [{
    Deletes operations that have no runtime equivalent and are only
    used in the compiler. This should be performed after all other
    compiler passes.
  }];
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createDropCompilerHintsPass()
  }];
}

def FixedPointIterator : Pass<"iree-util-fixed-point-iterator", ""> {
  let summary = "Iterates a sub-pipeline to a fixed point.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createFixedPointIteratorPass(
        mlir::OpPassManager("dummy_op"))
  }];
}

def IPO : Pass<"iree-util-ipo", "mlir::ModuleOp"> {
  let summary = "Performs basic inter-procedural optimization.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createIPOPass()
  }];
}

def PropagateSubranges : Pass<"iree-util-propagate-subranges", "mlir::ModuleOp"> {
  let summary = "Propagates resource subranges across the program.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createPropagateSubrangesPass()
  }];
}

def StripDebugOps : Pass<"iree-util-strip-debug-ops", ""> {
  let summary = "Strips debug ops, like assertions.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createStripDebugOpsPass()
  }];
}

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

def FoldGlobals : Pass<"iree-util-fold-globals", "mlir::ModuleOp"> {
  let summary = "Folds duplicate globals and propagates constants.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createFoldGlobalsPass()
  }];
}

def FuseGlobals : Pass<"iree-util-fuse-globals", "mlir::ModuleOp"> {
  let summary = "Fuses correlated globals together.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createFuseGlobalsPass()
  }];
}

def HoistIntoGlobals : Pass<"iree-util-hoist-into-globals", "mlir::ModuleOp"> {
  let summary = "Greedily hoists eligible constant expressions into globals.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createHoistIntoGlobalsPass()
  }];
}

def SimplifyGlobalAccesses :
    InterfacePass<"iree-util-simplify-global-accesses", "mlir::CallableOpInterface"> {
  let summary = "Hoists loads and sinks stores to variables to decrease data dependency regions.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createSimplifyGlobalAccessesPass()
  }];
}

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

def DemoteI64ToI32 : Pass<"iree-util-demote-i64-to-i32", "mlir::ModuleOp"> {
  let summary = "Demotes i64 types to i32 types.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createDemoteI64ToI32Pass()
  }];
}

def DemoteF32ToF16 : Pass<"iree-util-demote-f32-to-f16", "mlir::ModuleOp"> {
  let summary = "Demotes f32 types to f16 types.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createDemoteF32ToF16Pass()
  }];
}

def DemoteF64ToF32 : Pass<"iree-util-demote-f64-to-f32", "mlir::ModuleOp"> {
  let summary = "Demotes f64 types to f32 types.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createDemoteF64ToF32Pass()
  }];
}

def PromoteF16ToF32 : Pass<"iree-util-promote-f16-to-f32", "mlir::ModuleOp"> {
  let summary = "Promotes f16 types to f32 types.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createPromoteF16ToF32Pass()
  }];
}

def PromoteBF16ToF32 : Pass<"iree-util-promote-bf16-to-f32", "mlir::ModuleOp"> {
  let summary = "Promotes bf16 types to f32 types.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createPromoteBF16ToF32Pass()
  }];
}

//===----------------------------------------------------------------------===//
// Debug/test passes
//===----------------------------------------------------------------------===//

def AnnotateOpOrdinals : Pass<"iree-util-annotate-op-ordinals", "mlir::ModuleOp"> {
  let summary = "Annotates ops with globally unique IDs for debugging.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createAnnotateOpOrdinalsPass()
  }];
}

def TestConversion : Pass<"iree-util-test-conversion", "mlir::ModuleOp"> {
  let summary = "Tests util dialect conversion patterns.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createTestConversionPass()
  }];
}

def TestFloatRangeAnalysis : Pass<"iree-util-test-float-range-analysis", ""> {
  let summary = "Tests floating point range analysis.";
  let description = [{
    Tests floating point range analysis by evaluating any
    'iree_unregistered.test_fprange' op and setting the results on an attribute.
  }];
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createTestFloatRangeAnalysisPass()
  }];
}

#endif  // IREE_DIALECT_UTIL_PASSES
