// RUN: iree-dialects-opt --split-input-file --allow-unregistered-dialect --canonicalize %s | FileCheck  --dump-input-filter=all %s

// CHECK-LABEL: @elide_boxing_noop
iree_pydm.func @elide_boxing_noop(%arg0 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.object) {
  %0 = box %arg0 : !iree_pydm.object -> !iree_pydm.object
  // CHECK: return %arg0
  return %0 : !iree_pydm.object
}

// -----
// CHECK-LABEL: @preserves_boxing
iree_pydm.func @preserves_boxing(%arg0 : !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.object) {
  // NOTE: Canonicalizes to a specialized object.
  // CHECK: %[[BOXED:.*]] = box %arg0 : !iree_pydm.integer -> <!iree_pydm.integer>
  // CHECK: %[[CASTED:.*]] = static_info_cast %[[BOXED]] : !iree_pydm.object<!iree_pydm.integer> -> !iree_pydm.object
  %0 = box %arg0 : !iree_pydm.integer -> !iree_pydm.object
  // CHECK: return %[[CASTED]]
  return %0 : !iree_pydm.object
}

// -----
// CHECK-LABEL: @elide_unboxing_from_boxed_noop
iree_pydm.func @elide_unboxing_from_boxed_noop(%arg0 : !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.integer) {
  // CHECK-NOT: raise_on_failure
  %0 = box %arg0 : !iree_pydm.integer -> !iree_pydm.object<!iree_pydm.integer>
  %exc_result, %1 = unbox %0 : !iree_pydm.object<!iree_pydm.integer> -> !iree_pydm.integer
  raise_on_failure %exc_result : !iree_pydm.exception_result
  // CHECK: return %arg0
  return %1 : !iree_pydm.integer
}

// -----
// CHECK-LABEL: @preserve_unboxing
iree_pydm.func @preserve_unboxing(%arg0 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.integer) {
  %0 = box %arg0 : !iree_pydm.object -> !iree_pydm.object
  // CHECK: %[[STATUS:.*]], %[[UNBOXED:.*]] = unbox
  %exc_result, %1 = unbox %0 : !iree_pydm.object -> !iree_pydm.integer
  raise_on_failure %exc_result : !iree_pydm.exception_result
  // CHECK: return %[[UNBOXED]]
  return %1 : !iree_pydm.integer
}

// -----
// CHECK-LABEL: @unbox_apply_binary
// Note that many ops use the generic operand unboxing facility. It is exhaustively checked
// here and then just checked for indications on others.
iree_pydm.func @unbox_apply_binary(%arg0 : !iree_pydm.object<!iree_pydm.integer<32>>, %arg1 : !iree_pydm.object<!iree_pydm.integer<32>>) -> (!iree_pydm.exception_result, !iree_pydm.object) {
  // CHECK: %[[EL:.*]], %[[LHS:.*]] = unbox %arg0 : <!iree_pydm.integer<32>> -> !iree_pydm.integer<32>
  // CHECK: %[[ER:.*]], %[[RHS:.*]] = unbox %arg1 : <!iree_pydm.integer<32>> -> !iree_pydm.integer<32>
  // CHECK: %[[R:.*]] = apply_binary "add", %[[LHS]], %[[RHS]] : !iree_pydm.integer<32>, !iree_pydm.integer<32> -> !iree_pydm.object
  %0 = apply_binary "add", %arg0, %arg1 : !iree_pydm.object<!iree_pydm.integer<32>>, !iree_pydm.object<!iree_pydm.integer<32>> -> !iree_pydm.object
  return %0 : !iree_pydm.object
}

// -----
// CHECK-LABEL: @unbox_assign_subscript
// Note: Only the first two operands are unboxed
iree_pydm.func @unbox_assign_subscript(%arg0 : !iree_pydm.object<!iree_pydm.list>, %arg1 : !iree_pydm.object<!iree_pydm.integer<32>>, %arg2 : !iree_pydm.object<!iree_pydm.integer<32>>) -> (!iree_pydm.exception_result, !iree_pydm.object<!iree_pydm.list>) {
  // CHECK: assign_subscript %primitive[%primitive_1] = %arg2 : !iree_pydm.list, !iree_pydm.integer<32>, !iree_pydm.object<!iree_pydm.integer<32>>
  %0 = assign_subscript %arg0[%arg1] = %arg2 : !iree_pydm.object<!iree_pydm.list>, !iree_pydm.object<!iree_pydm.integer<32>>, !iree_pydm.object<!iree_pydm.integer<32>>
  return %arg0 : !iree_pydm.object<!iree_pydm.list>
}

// -----
// CHECK-LABEL: @unbox_dynamic_binary_promote
// This one is phrased a little funny because it is hard to canonicalize the unboxing
// in isolation. We check that each operand unboxes individually.
iree_pydm.func @unbox_dynamic_binary_promote(%arg0 : !iree_pydm.object<!iree_pydm.integer<32>>, %arg1 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.tuple) {
  // CHECK: dynamic_binary_promote {{.*}} : !iree_pydm.integer<32>, !iree_pydm.object
  // CHECK: dynamic_binary_promote {{.*}} : !iree_pydm.object, !iree_pydm.integer<32>
  %0, %1 = dynamic_binary_promote %arg0, %arg1 : !iree_pydm.object<!iree_pydm.integer<32>>, !iree_pydm.object
  %2, %3 = dynamic_binary_promote %arg1, %arg0 : !iree_pydm.object, !iree_pydm.object<!iree_pydm.integer<32>>
  %4 = make_tuple %0, %1, %2, %3 : !iree_pydm.object, !iree_pydm.object, !iree_pydm.object, !iree_pydm.object -> !iree_pydm.tuple
  return %4 : !iree_pydm.tuple
}

// -----
// CHECK-LABEL: @unbox_neg
iree_pydm.func @unbox_neg(%arg0 : !iree_pydm.object<!iree_pydm.integer<32>>) -> (!iree_pydm.exception_result, !iree_pydm.object<!iree_pydm.integer<32>>) {
  // CHECK: neg {{.*}} : !iree_pydm.integer<32> -> !iree_pydm.object<!iree_pydm.integer<32>>
  %0 = neg %arg0 : !iree_pydm.object<!iree_pydm.integer<32>> -> !iree_pydm.object<!iree_pydm.integer<32>>
  return %0 : !iree_pydm.object<!iree_pydm.integer<32>>
}

// -----
// CHECK-LABEL: @unbox_subscript
iree_pydm.func @unbox_subscript(%arg0 : !iree_pydm.object<!iree_pydm.list>, %arg1 : !iree_pydm.object<!iree_pydm.integer<32>>) -> (!iree_pydm.exception_result, !iree_pydm.object) {
  // CHECK: subscript {{.*}} : !iree_pydm.list, !iree_pydm.integer<32> -> !iree_pydm.object<!iree_pydm.integer<32>>
  %exc, %0 = subscript %arg0[%arg1] : !iree_pydm.object<!iree_pydm.list>, !iree_pydm.object<!iree_pydm.integer<32>> -> !iree_pydm.object<!iree_pydm.integer<32>>
  return %0 : !iree_pydm.object<!iree_pydm.integer<32>>
}
