// RUN: iree-dialects-opt -split-input-file -lower-iree-pydm-to-rtl %s | FileCheck --enable-var-scope --dump-input-filter=all %s

// CHECK-LABEL: @object_as_bool
// Doubles as a check for the general machinery.
iree_pydm.func @object_as_bool(%arg0 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
  // CHECK: %[[EXC:.*]], %[[V:.*]] = call @pydmrtl$object_as_bool(%arg0) : (!iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.bool)
  // CHECK: raise_on_failure %[[EXC]]
  // CHECK: return %[[V]]
  %0 = as_bool %arg0 : !iree_pydm.object -> !iree_pydm.bool
  return %0 : !iree_pydm.bool
}
// CHECK: iree_pydm.func private @pydmrtl$object_as_bool(!iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.bool)

// -----
// CHECK-LABEL: @no_duplicate_import
// Doubles as a check for the general machinery.
iree_pydm.func @no_duplicate_import(%arg0 : !iree_pydm.object, %arg1 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.tuple) {
  %0 = as_bool %arg0 : !iree_pydm.object -> !iree_pydm.bool
  %1 = as_bool %arg1 : !iree_pydm.object -> !iree_pydm.bool
  %2 = make_tuple %0, %1 : !iree_pydm.bool, !iree_pydm.bool -> !iree_pydm.tuple
  return %2 : !iree_pydm.tuple
}
// CHECK: iree_pydm.func private @pydmrtl$object_as_bool
// CHECK-NOT: iree_pydm.func private @pydmrtl$object_as_bool

// -----
// CHECK-LABEL: @dynamic_binary_promote
// Doubles as a check for the general machinery.
iree_pydm.func @dynamic_binary_promote(%arg0 : !iree_pydm.object, %arg1 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.object) {
  // CHECK: %[[exc_result:.*]], %[[result:.*]] = call @pydmrtl$dynamic_binary_promote(%arg0, %arg1) : (!iree_pydm.object, !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.tuple)
  // CHECK: raise_on_failure %[[exc_result]]
  // CHECK: %[[exc_result_0:.*]], %[[slots:.*]]:2 = dynamic_unpack %[[result]] : !iree_pydm.tuple -> !iree_pydm.exception_result, [!iree_pydm.object, !iree_pydm.object]
  // CHECK: raise_on_failure %[[exc_result_0]]
  // CHECK: make_tuple %[[slots]]#0, %[[slots]]#1
  %left, %right = dynamic_binary_promote %arg0, %arg1 : !iree_pydm.object, !iree_pydm.object
  %result = make_tuple %left, %right : !iree_pydm.object, !iree_pydm.object -> !iree_pydm.tuple
  return %result : !iree_pydm.tuple
}
