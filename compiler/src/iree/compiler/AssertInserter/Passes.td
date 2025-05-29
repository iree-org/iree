#ifndef IREE_COMPILER_ASSERTINSERTER_PASSES
#define IREE_COMPILER_ASSERTINSERTER_PASSES

include "mlir/Pass/PassBase.td"

def AssertInBoundsPass : Pass<"memref-assert-in-bounds"> {
  let summary = "Insert assertions that memref accesses are in bounds.";
  let dependentDialects = [
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
    "memref::MemRefDialect",
  ];
  let options = [
    Option<"warnOnUnknown", "warn-on-unknown", "bool", /*default=*/"true",
      "Warn on unknown side-effecting operations">,
    Option<"includeVectorLoadStore", "include-vector-load-store", "bool",
      /*default=*/"false",
      "Include vector.load/store operations despite them allowing "
      "out-of-bounds">,
    Option<"checkEachDim", "check-each-dim", "bool", /*default=*/"true",
      "Check each dimension individually">,
    // TODO: consider making this a separate pass as it has to operate at
    // module level.
    Option<"createSpeculativeFuncs", "create-speculative-funcs", "bool",
      /*default=*/"false",
      "Create a function that performs assertions speculatively instead of "
      "in-place checks">
  ];
}

def CheckStaticAssertionsPass : Pass<"check-static-assertions"> {
  let summary = "Fail if the IR contains a known-false assertion.";
}

#endif // IREE_COMPILER_ASSERTINSERTER_PASSES
