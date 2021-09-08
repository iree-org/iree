// RUN: iree-dialects-opt -split-input-file --allow-unregistered-dialect -canonicalize %s | FileCheck --enable-var-scope --dump-input-filter=all %s

// CHECK-LABEL: @elide_boxing_noop
iree_pydm.func @elide_boxing_noop(%arg0 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.object) {
  %0 = box %arg0 : !iree_pydm.object -> !iree_pydm.object
  // CHECK: return %arg0
  return %0 : !iree_pydm.object
}

// -----
// CHECK-LABEL: @preserves_boxing
iree_pydm.func @preserves_boxing(%arg0 : !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.object) {
  // CHECK: %[[BOXED:.*]] = box %arg0
  %0 = box %arg0 : !iree_pydm.integer -> !iree_pydm.object
  // CHECK: return %[[BOXED]]
  return %0 : !iree_pydm.object
}

// -----
// CHECK-LABEL: @elide_unboxing_from_boxed_noop
iree_pydm.func @elide_unboxing_from_boxed_noop(%arg0 : !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.integer) {
  // CHECK-NOT: raise_on_failure
  %0 = box %arg0 : !iree_pydm.integer -> !iree_pydm.object
  %exc_result, %1 = unbox %0 : !iree_pydm.object -> !iree_pydm.integer
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
