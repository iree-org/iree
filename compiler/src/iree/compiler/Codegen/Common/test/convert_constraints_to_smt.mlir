// RUN: iree-opt --split-input-file --iree-codegen-convert-constraints-to-smt %s | FileCheck %s

// CHECK-LABEL: iree_codegen.smt.constraints
// CHECK-SAME:    target = <set = 0>
// CHECK-SAME:    pipeline = LLVMGPUVectorDistribute

// CHECK:         smt.solver() : () -> ()

// Knobs become smt.declare_fun constants.
// CHECK:         %[[WG_M:.+]] = smt.declare_fun "wg_m" : !smt.int
// CHECK:         %[[IDX:.+]] = smt.declare_fun "mma_idx" : !smt.int

// Lookup [0,1]->[16,32] lowers to: ite(idx == 0, 16, 32).
// CHECK:         %[[C32:.+]] = smt.int.constant 32
// CHECK:         %[[C0:.+]] = smt.int.constant 0
// CHECK:         %[[C16:.+]] = smt.int.constant 16
// CHECK:         %[[EQ:.+]] = smt.eq %[[IDX]], %[[C0]] : !smt.int
// CHECK:         %{{.+}} = smt.ite %[[EQ]], %[[C16]], %[[C32]] : !smt.int
// CHECK-NOT:     smt.ite

// smt.int.cmp is cloned as-is.
// CHECK:         %[[CMP:.+]] = smt.int.cmp le %[[WG_M]], %[[WG_M]]

// iree_codegen.smt.assert becomes smt.assert.
// CHECK:         smt.assert %[[CMP]]

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
