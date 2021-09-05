// RUN: iree-dialects-opt -split-input-file --allow-unregistered-dialect -canonicalize %s | FileCheck --enable-var-scope --dump-input-filter=all %s

// CHECK-LABEL: @elide_raise_on_failure_success
iree_pydm.func @elide_raise_on_failure_success() -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK-NOT: raise_on_failure
  %0 = success -> !iree_pydm.exception_result
  raise_on_failure %0 : !iree_pydm.exception_result
  %none = none
  return %none : !iree_pydm.none
}

// -----
// CHECK-LABEL: @preserve_raise_on_failure_failure
iree_pydm.func @preserve_raise_on_failure_failure() -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK: raise_on_failure
  %0 = failure -> !iree_pydm.exception_result
  raise_on_failure %0 : !iree_pydm.exception_result
  %none = none
  return %none : !iree_pydm.none
}

// -----
// CHECK-LABEL: @preserve_raise_on_failure_unknown
iree_pydm.func @preserve_raise_on_failure_unknown(%arg0 : !iree_pydm.exception_result) -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK: raise_on_failure
  raise_on_failure %arg0 : !iree_pydm.exception_result
  %none = none
  return %none : !iree_pydm.none
}
