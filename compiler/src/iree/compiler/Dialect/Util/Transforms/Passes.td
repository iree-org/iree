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

def ApplyPatternsPass : Pass<"iree-util-apply-patterns", ""> {
  let summary = "Applies some risky/IREE-specific canonicalization patterns.";
}

def CombineInitializersPass : Pass<"iree-util-combine-initializers", "mlir::ModuleOp"> {
  let summary = "Combines global initializers into one.";
}

def DropCompilerHintsPass : Pass<"iree-util-drop-compiler-hints", ""> {
  let summary = "Deletes operations that have no runtime equivalent.";
  let description = [{
    Deletes operations that have no runtime equivalent and are only
    used in the compiler. This should be performed after all other
    compiler passes.
  }];
}

def DumpModulePass : Pass<"iree-util-dump-module", "mlir::ModuleOp"> {
  let summary = "Dumps the module IR to the given file path.";
  let description = [{
    Dumps the module IR to the given file path in either textual (.mlir) or
    binary (.mlirbc) format. Source locations remain unchanged.
  }];
  let options = [
    Option<"path", "path",
      "std::string", /*default=*/"",
      "File path to write the module text or binary into."
    >
  ];
}

def FixedPointIteratorPass : Pass<"iree-util-fixed-point-iterator", ""> {
  let summary = "Iterates a sub-pipeline to a fixed point.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createFixedPointIteratorPass(
        mlir::OpPassManager())
  }];
}

def IPOPass : Pass<"iree-util-ipo", "mlir::ModuleOp"> {
  let summary = "Performs basic inter-procedural optimization.";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::iree_compiler::IREE::Util::UtilDialect"
  ];
}

def OptimizeIntArithmeticPass : Pass<"iree-util-optimize-int-arithmetic", ""> {
  let summary = "Optimizes integer arithmetic using a variety of dataflow analysis and patterns.";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::iree_compiler::IREE::Util::UtilDialect"
  ];
  let options = [
    Option<"narrowToI32", "narrow-to-i32", "bool",
      /*default=*/"false",
      "Flag indicating if computations that can be performed with 32 bits should be."
      " Mainly used for GPU code generation to not waste registers.">
  ];
}

def PropagateSubrangesPass : Pass<"iree-util-propagate-subranges", "mlir::ModuleOp"> {
  let summary = "Propagates resource subranges across the program.";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::iree_compiler::IREE::Util::UtilDialect"
  ];

}

def StripAndSplatConstantsPass :
    Pass<"iree-util-strip-and-splat-constants", "mlir::ModuleOp"> {
  let summary = "Strips constant util.global ops and replaces them with splats.";
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::Util::UtilDialect"
  ];
}

def StripDebugOpsPass : Pass<"iree-util-strip-debug-ops", ""> {
  let summary = "Strips debug ops, like assertions.";
}

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

def FoldGlobalsPass : Pass<"iree-util-fold-globals", "mlir::ModuleOp"> {
  let summary = "Folds duplicate globals and propagates constants.";

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::iree_compiler::IREE::Util::UtilDialect"
  ];

  let statistics = [
    Statistic<"beforeFoldingGlobals", "global ops before folding",
      "Number of util.global ops before folding">,
    Statistic<"afterFoldingGlobals", "global ops after folding",
      "Number of util.global ops after folding">
  ];
}

def FuseGlobalsPass : Pass<"iree-util-fuse-globals", "mlir::ModuleOp"> {
  let summary = "Fuses correlated globals together.";
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::Util::UtilDialect"
  ];
}

def HoistIntoGlobalsPass : Pass<"iree-util-hoist-into-globals", "mlir::ModuleOp"> {
  let summary = "Greedily hoists eligible constant expressions into globals.";
  // Note: has a custom options struct that lets you register dependent dialects
  let options = [
    Option<"maxSizeIncreaseThreshold", "max-size-increase-threshold", "int64_t",
      /*default=*/"1048576",
      "Maximum byte size increase allowed for constant expr hoisting policy to"
      "allow hoisting. The threshold is 1MB by default.">
  ];
}

def SimplifyGlobalAccessesPass :
    InterfacePass<"iree-util-simplify-global-accesses", "mlir::CallableOpInterface"> {
  let summary = "Hoists loads and sinks stores to variables to decrease data dependency regions.";
}

//===----------------------------------------------------------------------===//
// Resource Management
//===----------------------------------------------------------------------===//

def ImportResourcesPass : Pass<"iree-util-import-resources", ""> {
  let summary = "Imports IR with arbitrary large-data into resources that IREE can manage efficiently";
  let description = [{
    MLIR has many interesting ways to store large constants, most of which
    derive from *ElementsAttr. Given the uniquing/inline behavior, this exacts
    very large runtime and memory overhead costs.

    This is a temporary pass to convert a majority of the legacy
    DenseElementsAttr attributes to DenseResourceElementsAttr. Ideally this
    is done at the source (frontend), but this pass is provided to aid
    transition and testing by doing a manual conversion with iree-opt.
  }];

  let dependentDialects = [
    "::mlir::BuiltinDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Debug/test passes
//===----------------------------------------------------------------------===//

def AnnotateOpOrdinalsPass : Pass<"iree-util-annotate-op-ordinals", "mlir::ModuleOp"> {
  let summary = "Annotates ops with globally unique IDs for debugging.";
}

def TestConversionPass : Pass<"iree-util-test-conversion", "mlir::ModuleOp"> {
  let summary = "Tests util dialect conversion patterns.";
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::Util::UtilDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::memref::MemRefDialect"
  ];
  let options = [
    Option<"widenIntegers", "widen-integers",
      "bool", /*default=*/"false",
      "Tests type conversion by widening integers to i32">
  ];
}

def TestFloatRangeAnalysisPass : Pass<"iree-util-test-float-range-analysis", ""> {
  let summary = "Tests floating point range analysis.";
  let description = [{
    Tests floating point range analysis by evaluating any
    'iree_unregistered.test_fprange' op and setting the results on an attribute.
  }];
}

#endif  // IREE_DIALECT_UTIL_PASSES
