// RUN: iree-dialects-opt -linalg-interp-transforms %s | FileCheck %s


//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0, s1, s2] -> (s1 - (-s0 + s1) mod s2)>
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0)>
//      CHECK: func @fully_dynamic_bounds(
// CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index
//      CHECK:   %[[C0_I32:.*]] = arith.constant 0 : i32
//      CHECK:   %[[NEW_UB:.*]] = affine.apply #[[MAP0]]()[%[[LB]], %[[UB]], %[[STEP]]]
//      CHECK:   %[[CAST:.*]] = arith.index_cast %[[STEP]] : index to i32
//      CHECK:   %[[LOOP:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[NEW_UB]]
// CHECK-SAME:       step %[[STEP]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      CHECK:     %[[ADD:.*]] = arith.addi %[[ACC]], %[[CAST]] : i32
//      CHECK:     scf.yield %[[ADD]]
//      CHECK:   }
//      CHECK:   %[[RESULT:.*]] = scf.for %[[IV2:.*]] = %[[NEW_UB]] to %[[UB]]
// CHECK-SAME:       step %[[STEP]] iter_args(%[[ACC2:.*]] = %[[LOOP]]) -> (i32) {
//      CHECK:     %[[REM:.*]] = affine.apply #[[MAP1]](%[[IV2]])[%[[UB]]]
//      CHECK:     %[[CAST2:.*]] = arith.index_cast %[[REM]]
//      CHECK:     %[[ADD2:.*]] = arith.addi %[[ACC2]], %[[CAST2]]
//      CHECK:     scf.yield %[[ADD2]]
//      CHECK:   }
//      CHECK:   return %[[RESULT]]
#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func @fully_dynamic_bounds(%lb : index, %ub: index, %step: index) -> i32 {
  %c0 = arith.constant 0 : i32
  %r = scf.for %iv = %lb to %ub step %step iter_args(%arg = %c0) -> i32 {
    %s = affine.min #map(%ub, %iv)[%step]
    %casted = arith.index_cast %s : index to i32
    %0 = arith.addi %arg, %casted : i32
    scf.yield %0 : i32
  }
  return %r : i32
}

pdl.pattern @pdl_target : benefit(1) {
  %args = operands
  %results = types
  %0 = pdl.operation "scf.for"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
  apply_native_constraint "nestedInFunc"[@fully_dynamic_bounds](%0 : !pdl.operation)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  rewrite %0 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %0 = match @pdl_target
  peel_loop %0
}
