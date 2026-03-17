// RUN: iree-opt --split-input-file --iree-codegen-convert-constraints-to-smt %s | FileCheck %s

// CHECK-LABEL: iree_codegen.smt.constraints
// CHECK-SAME:    target = <set = 0>
// CHECK-SAME:    pipeline = LLVMGPUVectorDistribute

// CHECK:         smt.solver() : () -> ()

// Knobs become smt.declare_fun constants.
// CHECK:         %wg_m = smt.declare_fun "wg_m" : !smt.int
// CHECK:         %mma_idx = smt.declare_fun "mma_idx" : !smt.int

// Lookup [0,1]->[16,32] lowers to: ite(idx == 0, 16, 32).
// CHECK:         %c32 = smt.int.constant 32
// CHECK:         %c0 = smt.int.constant 0
// CHECK:         %c16 = smt.int.constant 16
// CHECK:         %0 = smt.eq %mma_idx, %c0 : !smt.int
// CHECK:         %1 = smt.ite %0, %c16, %c32 : !smt.int
// CHECK-NOT:     smt.ite

// smt.int.cmp is cloned as-is.
// CHECK:         %2 = smt.int.cmp le %wg_m, %wg_m

// iree_codegen.smt.assert becomes smt.assert.
// CHECK:         smt.assert %2

// CHECK-NOT:     iree_codegen.smt.knob
// CHECK-NOT:     iree_codegen.smt.lookup
// CHECK-NOT:     iree_codegen.smt.assert

iree_codegen.smt.constraints
    target = <set = 0>,
    pipeline = LLVMGPUVectorDistribute,
    knobs = {wg_m = #iree_codegen.smt.int_knob<"wg_m">,
             mma_idx = #iree_codegen.smt.int_knob<"mma_idx">}
    dims() {
^bb0:
  %wg_m = iree_codegen.smt.knob "wg_m" : !smt.int
  %idx = iree_codegen.smt.knob "mma_idx" : !smt.int
  %mma_m = iree_codegen.smt.lookup %idx [0, 1] -> [16, 32] : !smt.int
  %cond = smt.int.cmp le %wg_m, %wg_m
  iree_codegen.smt.assert %cond, "wg_m <= wg_m" : !smt.bool
}
