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

def AttributeCallGraphPass : Pass<"iree-util-attribute-call-graph", "mlir::ModuleOp"> {
  let summary = "Propagates attributes from callees to call sites.";
  let description = [{
    Walks all call operations and propagates relevant attributes from the
    called functions to the call sites. This includes:
    * Attributes implementing CallAnnotationAttrInterface
    * nosideeffects (for side-effect analysis)

    This allows subsequent passes to check attributes directly on call sites
    without expensive symbol table lookups during analysis.
  }];
}

def CombineInitializersPass : Pass<"iree-util-combine-initializers", "mlir::ModuleOp"> {
  let summary = "Combines global initializers into one.";
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::Util::UtilDialect"
  ];
}

def DropCompilerHintsPass : Pass<"iree-util-drop-compiler-hints", ""> {
  let summary = "Deletes operations that have no runtime equivalent.";
  let description = [{
    Deletes operations that have no runtime equivalent and are only
    used in the compiler. This should be performed after all other
    compiler passes.

    With keep-assume-int=true, leaves util.int.assume operations in place
    so they can be propagated to backends. This is a temporary measure
    until all bbackends have a rewrite for those assumptions (currently
    they're only handled by the patterns that target LLVM).
  }];

  let options = [
    Option<
      "keepAssumeInt", "keep-assume-int",
      "bool", "false",
      "Whether annotations about the ranges and divisibility of integers should be kept."
    >,
  ];
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
        mlir::OpPassManager("dummy_op"))
  }];
}

def IPOPass : Pass<"iree-util-ipo", "mlir::ModuleOp"> {
  let summary = "Performs basic inter-procedural optimization.";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::iree_compiler::IREE::Util::UtilDialect"
  ];
}

def LiftCFGToSCFPass : Pass<"iree-util-lift-cfg-to-scf", "mlir::ModuleOp"> {
  let summary = "Lift unstructured control flow to SCF/Util dialect for callable ops.";
  let description = [{
    Transforms unstructured control flow (CFG) in `util.initializer`/`util.func`
    ops to structured control flow (SCF) operations. All CFG terminator ops like
    `cf.br` and `cf.cond_br` along with non-terminator ops like `cf.switch` are
    converted to their structured equivalents like `scf.if`, `scf.while`, and
    `scf.index_switch` ops.

    Limitations:
    * Irreducible control flow (loops with multiple entry points) will cause
      the transformation to fail. This is rare in practice as most high-level
      languages and ML frameworks generate reducible control flow.
    * Operations with side-effecting branches are not supported.
    * Operations that produce successor operands dynamically are not supported.
    * Infinite loops require the insertion of `util.unreachable` terminators.

    The algorithm handles:
    * Simple if/else patterns -> `scf.if`
    * Loops with single entry -> `scf.while`
    * Switch statements -> `scf.index_switch`
    * Multiple returns -> exit block combining
    * Unreachable code -> `util.unreachable` insertion

    Based on the algorithm from:
    Bahmann et al. 2015. Perfect Reconstructability of Control Flow from
    Demand Dependence Graphs. ACM TACO. https://dl.acm.org/doi/10.1145/2693261
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::cf::ControlFlowDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::ub::UBDialect",
  ];
}

def LinkModulesPass : Pass<"iree-util-link-modules", "mlir::ModuleOp"> {
  let summary = "Links external functions from source modules into the target module.";
  let description = [{
    Resolves external function declarations by loading definitions from source
    modules. This pass enables composing programs from multiple MLIR modules
    where functions can reference implementations from separately compiled
    modules using dotted name scoping (e.g., `@module_a.compute`).

    For each external function in the target module:

    1. Searches for the definition in explicitly provided source modules
    2. If not found and the symbol has a module prefix (`@prefix.name`) attempts
       to automatically load `prefix.mlir` or `prefix.mlirbc` from library paths
    3. Clones the function definition along with transitive dependencies
       (including `IREE::Util::ObjectLike` operations like `flow.executable`)
    4. Detects and resolves conflicts between private symbols from different
       source modules or with existing target module definitions by renaming
       conflicting symbols (e.g., `@scale_factor`, `@scale_factor_0`)
    5. Renames imported symbols to match the external declaration name
    6. Forces all imported symbols to private visibility

    ObjectLike operations (such as `flow.executable`) are treated as
    self-contained units: their internal structure is preserved but not scanned
    for additional dependencies, preventing nested symbols from being treated as
    external references.

    Module name scoping: Source modules with `module @foo { util.func @bar }`
    export public symbols as `@foo.bar`. Private symbols keep their local names.
    Internal references within the module use the simple name `@bar`, while
    external references use the dotted notation.

    Supports both text (`.mlir`) and bytecode (`.mlirbc`) formats for all
    modules.
  }];

  let options = [
    ListOption<
      "linkModules", "link-module",
      "std::string",
      "Explicit paths to source modules to link from."
    >,
    ListOption<
      "libraryPaths", "library-path",
      "std::string",
      "Directories to search for modules during automatic discovery."
    >,
  ];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
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

def VerifyInitializationOrderPass :
    Pass<"iree-util-verify-initialization-order", "mlir::ModuleOp"> {
  let summary = "Verifies module initialization order constraints.";
  let description = [{
    Verifies that module initialization follows the rules documented in the
    util.initializer and util.global operations:

    1. Initializers can only access globals defined before them in module order.
    2. Immutable globals can only be initialized once (either by initial_value
       OR by stores in initializers, never both).
    3. Globals with initial values cannot be modified by initializers that
       precede them in module order.
    4. Stores to immutable globals must occur in initializers or functions
       only reachable from initializers.

    The pass uses conservative analysis and emits warnings for patterns it
    cannot definitively verify (e.g. conditional stores in initializer-only
    functions).
  }];
}

def VerifyStructuredControlFlowPass :
    InterfacePass<"iree-util-verify-structured-control-flow", "mlir::FunctionOpInterface"> {
  let summary = "Verifies that functions contain only structured control flow.";
  let description = [{
    Verifies that functions contain only structured control flow (no ops
    implementing `BranchOpInterface` like `cf.br` or `cf.cond_br`).
  }];
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
      "Tests type conversion by widening integers to i32.">,
    Option<"structuralConversion", "structural-conversion",
      "bool", /*default=*/"true",
      "Tests generic structural conversion ops.">
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
