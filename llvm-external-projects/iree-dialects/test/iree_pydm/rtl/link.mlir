// RUN: iree-dialects-opt -verify-diagnostics -split-input-file -lower-iree-pydm-to-rtl -link-iree-pydm-rtl='rtl-file=%resources_dir/PyDMRTL/PyDMRTLBase.mlir' %s | FileCheck --enable-var-scope --dump-input-filter=all %s

// CHECK-LABEL: module @multi_with_private
// A multiple module import with additional private symbols included.
// Verifies that externs are replaced with definitions and are made private.
// The MLIR verifier will ensure that any dependent symbols are included.
// NOTE: The order here is not important but should be deterministic.
// CHECK-NOT: iree_pydm.func private @pydmrtl$dynamic_binary_promote{{.*}}{
// CHECK: iree_pydm.func private @pydmrtl$object_as_bool{{.*}}{
module @multi_with_private {
  iree_pydm.func @object_as_bool(%arg0 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
    %0 = as_bool %arg0 : !iree_pydm.object -> !iree_pydm.bool
    return %0 : !iree_pydm.bool
  }

  // TODO: Re-enable.
  // iree_pydm.func @dynamic_binary_promote(%arg0 : !iree_pydm.object, %arg1 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.object) {
  //   %left, %right = dynamic_binary_promote %arg0, %arg1 : !iree_pydm.object, !iree_pydm.object
  //   %result = make_tuple %left, %right : !iree_pydm.object, !iree_pydm.object -> !iree_pydm.tuple
  //   return %result : !iree_pydm.tuple
  // }
}

// -----
module @symbol_not_found {
  iree_pydm.func @DOES_NOT_EXIST(%arg0: !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
    %exc_result, %result = call @pydmrtl$DOES_NOT_EXIST(%arg0) : (!iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.bool)
    raise_on_failure %exc_result : !iree_pydm.exception_result
    return %result : !iree_pydm.bool
  }
  // expected-error@+1 {{could not resolve extern "pydmrtl$DOES_NOT_EXIST"}}
  iree_pydm.func private @pydmrtl$DOES_NOT_EXIST(%arg0: !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.bool)
}
