// RUN: iree-dialects-opt -pydm-local-propagate-types --allow-unregistered-dialect %s | FileCheck  --dump-input-filter=all %s

// CHECK-LABEL: @sink_static_info_cast_into_refinable
// "neg" implements TypeRefinableOpInterface and thus can have casts sunk
// into it.
// "make_list" does too and is used to make testing easier.
// It is difficult to test one concept in isolation, so this is also pretty
// close to a trivial full test.
// Subsequent tests will test more local characteristics only if possible.
iree_pydm.func @sink_static_info_cast_into_refinable(%arg0 : !iree_pydm.object<!iree_pydm.integer<32>>) -> (!iree_pydm.exception_result, !iree_pydm.list) {
  // CHECK: %{{.*}}, %[[UNBOXED:.*]] = unbox %arg0 : <!iree_pydm.integer<32>> -> !iree_pydm.integer<32>
  // CHECK: %[[NEG:.*]] = neg %[[UNBOXED]] : !iree_pydm.integer<32> -> !iree_pydm.integer<32>
  // CHECK: %[[BOXED:.*]] = box %[[NEG]] : !iree_pydm.integer<32> -> <!iree_pydm.integer<32>>
  // CHECK: make_list %[[BOXED]]
  %0 = static_info_cast %arg0 : !iree_pydm.object<!iree_pydm.integer<32>> -> !iree_pydm.object
  %1 = neg %0 : !iree_pydm.object -> !iree_pydm.object
  %list = make_list %1 : !iree_pydm.object -> !iree_pydm.list
  return %list : !iree_pydm.list
}

// -----
// CHECK-LABEL: @sink_static_info_cast_into_branch
// This is a simple test illustrating CFG permutation and cast sinking. Note
// that the cast in the entry block is sunk into copies of ^bb1 and ^bb2, and
// because of the donotoptimize op, the fully generic path of ^bb2 also must
// be preserved (as ^bb3).
// CHECK:   std.cond_br %arg0, ^bb1(%arg1 : !iree_pydm.object<!iree_pydm.integer<32>>), ^bb2(%arg1 : !iree_pydm.object<!iree_pydm.integer<32>>)
// CHECK: ^bb1(%[[BB1_PHI0:.*]]: !iree_pydm.object<!iree_pydm.integer<32>>): // pred: ^bb0
// CHECK:   %[[BB1_V0:.*]] = static_info_cast %[[BB1_PHI0]] : !iree_pydm.object<!iree_pydm.integer<32>> -> !iree_pydm.object
// CHECK:   %[[BB1_V1:.*]] = "custom.donotoptimize"(%[[BB1_V0]]) : (!iree_pydm.object) -> !iree_pydm.object
// CHECK:   std.br ^bb3(%[[BB1_V1]] : !iree_pydm.object)
// CHECK: ^bb2(%[[BB2_PHI0:.*]]: !iree_pydm.object<!iree_pydm.integer<32>>): // pred: ^bb0
// CHECK:   %[[BB2_V0:.*]] = make_list %[[BB2_PHI0]]
// CHECK:   return %[[BB2_V0]]
// CHECK: ^bb3(%[[BB3_PHI0:.*]]: !iree_pydm.object): // pred: ^bb1
// CHECK:    %[[BB3_V0:.*]] = make_list %[[BB3_PHI0]]
// CHECK:    return %[[BB3_V0]]
iree_pydm.func @sink_static_info_cast_into_branch(%pred : i1, %arg0 : !iree_pydm.object<!iree_pydm.integer<32>>) -> (!iree_pydm.exception_result, !iree_pydm.list) {
  %0 = static_info_cast %arg0 : !iree_pydm.object<!iree_pydm.integer<32>> -> !iree_pydm.object
  std.cond_br %pred, ^bb1(%0 : !iree_pydm.object), ^bb2(%0 : !iree_pydm.object)
^bb1(%phi0 : !iree_pydm.object):
  %1 = "custom.donotoptimize"(%phi0) : (!iree_pydm.object) -> (!iree_pydm.object)
  std.br ^bb2(%1 : !iree_pydm.object)
^bb2(%phi1 : !iree_pydm.object):
  %list = make_list %phi1 : !iree_pydm.object -> !iree_pydm.list
  return %list : !iree_pydm.list
}

// -----
// CHECK-LABEL: @rrt_apply_binary_numeric
// CHECK: apply_binary "add", %arg0, %arg1 : !iree_pydm.integer<32>, !iree_pydm.integer<32> -> !iree_pydm.integer<32>
iree_pydm.func @rrt_apply_binary_numeric(%arg0 : !iree_pydm.integer<32>, %arg1 : !iree_pydm.integer<32>) -> (!iree_pydm.exception_result, !iree_pydm.list) {
  %0 = apply_binary "add", %arg0, %arg1 : !iree_pydm.integer<32>, !iree_pydm.integer<32> -> !iree_pydm.object
  %list = make_list %0 : !iree_pydm.object -> !iree_pydm.list
  return %list : !iree_pydm.list
}

// -----
// CHECK-LABEL: @rrt_apply_binary_mul_list_numeric_left
// CHECK: sequence_clone %arg0 * %arg1 : !iree_pydm.list, !iree_pydm.integer<32> -> !iree_pydm.list
// Note: Also tests that the canonicalization to sequence_clone takes place.
iree_pydm.func @rrt_apply_binary_mul_list_numeric_left(%arg0 : !iree_pydm.list, %arg1 : !iree_pydm.integer<32>) -> (!iree_pydm.exception_result, !iree_pydm.list) {
  %0 = apply_binary "mul", %arg0, %arg1 : !iree_pydm.list, !iree_pydm.integer<32> -> !iree_pydm.object
  %list = make_list %0 : !iree_pydm.object -> !iree_pydm.list
  return %list : !iree_pydm.list
}

// -----
// CHECK-LABEL: @rrt_apply_binary_mul_list_numeric_right
// CHECK: sequence_clone %arg0 * %arg1 : !iree_pydm.list, !iree_pydm.integer<32> -> !iree_pydm.list
// Note: Also tests that the canonicalization to sequence_clone and type refinement takes place.
iree_pydm.func @rrt_apply_binary_mul_list_numeric_right(%arg0 : !iree_pydm.list, %arg1 : !iree_pydm.integer<32>) -> (!iree_pydm.exception_result, !iree_pydm.list) {
  %0 = apply_binary "mul", %arg1, %arg0 : !iree_pydm.integer<32>, !iree_pydm.list -> !iree_pydm.object
  %list = make_list %0 : !iree_pydm.object -> !iree_pydm.list
  return %list : !iree_pydm.list
}

// -----
// CHECK-LABEL: @rrt_neg
// CHECK: neg %arg0 : !iree_pydm.integer<32> -> !iree_pydm.integer<32>
iree_pydm.func @rrt_neg(%arg0 : !iree_pydm.integer<32>) -> (!iree_pydm.exception_result, !iree_pydm.list) {
  %0 = neg %arg0 : !iree_pydm.integer<32> -> !iree_pydm.object
  %list = make_list %0 : !iree_pydm.object -> !iree_pydm.list
  return %list : !iree_pydm.list
}
