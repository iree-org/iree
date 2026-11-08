// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_PASSES
#define IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// IREE-specific passes
//===----------------------------------------------------------------------===//

def ConvertStableHloToIreeInputDialects :
    Pass<"iree-stablehlo-to-iree-input", "ModuleOp"> {
  let summary = "Convert StableHLO ops to IREE input dialects ops";
  let description = [{
    Converts StableHLO input to IREE input dialects. These include: Linalg,
    Arith, Tensor, Complex, Shape, ML Program, Flow, and Util.
  }];
}

def ConvertStableHloToLinalgExt :
    InterfacePass<"iree-stablehlo-to-linalg-ext", "mlir::FunctionOpInterface"> {
  let summary =
      "Converts from StableHLO ops to LinalgExt ops and distribute to Flow ops";
}

//===----------------------------------------------------------------------===//
// General passes
//===----------------------------------------------------------------------===//

def LegalizeDeprecatedOpsPass :
    Pass<"iree-stablehlo-legalize-deprecated-ops", "ModuleOp"> {
  let summary = "Normalizes deprecated StableHLO operations";
  let description = [{
    Replaces deprecated operation spellings in module functions with their
    supported equivalents.
    Leaves map and rng operations unchanged. Rejects quantized unary einsums.
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def LegalizeDynamicReduceWindow :
    Pass<"iree-stablehlo-legalize-dynamic-reduce-window", "ModuleOp"> {
  let summary =
      "Lowers stablehlo.dynamic_reduce_window custom calls that compute scans";
  let description = [{
    Converts recognized dynamic prefix and suffix windows to LinalgExt scans.
    After a
    successful conversion, removes only private reducers proven unused in the
    containing symbol scope. Externally visible reducers remain callable.
  }];
}

def LegalizeControlFlow :
    InterfacePass<"iree-stablehlo-legalize-control-flow", "mlir::FunctionOpInterface"> {
  let summary = "Legalizes from StableHLO control flow to SCF control flow";
}

def LegalizeStableHLOCustomCalls :
    InterfacePass<"iree-stablehlo-legalize-custom-calls", "mlir::FunctionOpInterface"> {
  let summary = "Legalizes specialized custom calls to decomposed implementations";
  let description = [{
    Lowers supported custom calls, including shape assertions, to their IREE
    implementations. Shape assertions remain runtime status checks. Supported
    dimension equalities, affine ties, bounds and divisibility facts are analyzed
    without modifying IR, then materialized after the assertions establishing
    them. Only dominated original uses are rewritten; assertion predicates retain
    their original operands. Unsupported or overflowing proofs add no facts.
  }];
}

def LegalizeShapeComputations :
    InterfacePass<"iree-stablehlo-legalize-shape-computations", "mlir::FunctionOpInterface"> {
  let summary = "Legalizes StableHLO shape operations to core-mlir operations";
  let description = [{
    Recognizes scalar and vector shape computations and expands elementwise
    operations with statically shaped results into scalar arithmetic. Dynamic
    results and integer operations with explicit signedness remain available
    for ordinary tensor lowering.
  }];
}

def VerifyCompilerStableHloInputLegality :
    Pass<"iree-stablehlo-verify-compiler-input-legality", "ModuleOp"> {
  let summary =
      "Verifies that only supported IR constructs are passed to the compiler";
}

def CheckVHLOStableHloMixUsage :
    Pass<"iree-check-vhlostablehlo-mix-usage", "ModuleOp"> {
  let summary =
      "Check and report an error when VHLO and StableHLO are used in the same module";
}

def InlineShapeAssertionHelpers :
    Pass<"iree-stablehlo-inline-shape-assertion-helpers", "ModuleOp"> {
  let summary =
      "Inlines the private scalar helpers computing shape_assertion operands";
  let description = [{
    Analyzes private, pure, single-block scalar-integer helper closures without
    changing IR. Rejects recursive dependencies and preserves no-inline and
    affinity boundaries on functions and calls. Applies eligible calls in
    dependency order with a pass-wide cloning allowance of eight times the
    original module operation count plus 4096 operations. Expanded-body
    estimates saturate and count each call occurrence, including nested-region
    operations. Over-budget or failed inlining leaves valid calls and runtime
    assertions intact; the allowance limits this optimization, not input size.
  }];
  let dependentDialects = ["mlir::func::FuncDialect"];
}

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_PASSES
