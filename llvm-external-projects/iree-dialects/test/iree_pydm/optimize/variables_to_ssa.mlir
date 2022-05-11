// RUN: iree-dialects-opt --pydm-variables-to-ssa --allow-unregistered-dialect %s | FileCheck  --dump-input-filter=all %s

// CHECK-LABEL: @entry_block_does_not_hoist
// Hoisting must be disabled for the entry block as that would change the
// func signature (which is tested because it would fail verification).
iree_pydm.func @entry_block_does_not_hoist() -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK: load_var
  %a = alloc_free_var "a" -> !iree_pydm.free_var_ref
  %0 = load_var %a : !iree_pydm.free_var_ref -> !iree_pydm.object
  "custom.donotoptimize"(%0) : (!iree_pydm.object) -> ()
  %none = none
  return %none : !iree_pydm.none
}

// CHECK-LABEL: @free_var_eliminate_same_block_load
iree_pydm.func @free_var_eliminate_same_block_load() -> (!iree_pydm.exception_result, !iree_pydm.none) {
  %a = alloc_free_var "a" -> !iree_pydm.free_var_ref
  // CHECK: %[[R:.*]] = load_var %a
  // CHECK-NOT: load_var
  %0 = load_var %a : !iree_pydm.free_var_ref -> !iree_pydm.object
  %1 = load_var %a : !iree_pydm.free_var_ref -> !iree_pydm.object
  // CHECK: "custom.donotoptimize"(%[[R]], %[[R]])
  "custom.donotoptimize"(%0, %1) : (!iree_pydm.object, !iree_pydm.object) -> ()
  %none = none
  return %none : !iree_pydm.none
}

// CHECK-LABEL: @free_var_store_forwards_to_load
iree_pydm.func @free_var_store_forwards_to_load(%arg0 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK-NOT: store_var
  // CHECK-NOT: load_var
  %a = alloc_free_var "a" -> !iree_pydm.free_var_ref
  store_var %a = %arg0 : !iree_pydm.free_var_ref, !iree_pydm.object
  %0 = load_var %a : !iree_pydm.free_var_ref -> !iree_pydm.object
  %1 = load_var %a : !iree_pydm.free_var_ref -> !iree_pydm.object
  // CHECK: "custom.donotoptimize"(%arg0, %arg0)
  "custom.donotoptimize"(%0, %1) : (!iree_pydm.object, !iree_pydm.object) -> ()
  %none = none
  return %none : !iree_pydm.none
}

// CHECK-LABEL: @free_var_store_forwards_to_next_load
iree_pydm.func @free_var_store_forwards_to_next_load(%arg0 : !iree_pydm.object, %arg1 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK-NOT: store_var
  // CHECK-NOT: load_var
  %a = alloc_free_var "a" -> !iree_pydm.free_var_ref
  store_var %a = %arg0 : !iree_pydm.free_var_ref, !iree_pydm.object
  %0 = load_var %a : !iree_pydm.free_var_ref -> !iree_pydm.object
  store_var %a = %arg1 : !iree_pydm.free_var_ref, !iree_pydm.object
  %1 = load_var %a : !iree_pydm.free_var_ref -> !iree_pydm.object
  // CHECK: "custom.donotoptimize"(%arg0, %arg1)
  "custom.donotoptimize"(%0, %1) : (!iree_pydm.object, !iree_pydm.object) -> ()
  %none = none
  return %none : !iree_pydm.none
}

// CHECK-LABEL: @free_var_non_local_load_hoists
iree_pydm.func @free_var_non_local_load_hoists(%arg0 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK-NOT: store_var
  %a = alloc_free_var "a" -> !iree_pydm.free_var_ref
  store_var %a = %arg0 : !iree_pydm.free_var_ref, !iree_pydm.object
  // CHECK: cf.br ^bb1(%arg0 : !iree_pydm.object)
  cf.br ^bb1
  // CHECK: ^bb1(%[[PHI:.*]]: !iree_pydm.object): // pred: ^bb0
^bb1:
  // CHECK-NOT: load_var
  %0 = load_var %a : !iree_pydm.free_var_ref -> !iree_pydm.object
  %1 = load_var %a : !iree_pydm.free_var_ref -> !iree_pydm.object
  // CHECK: "custom.donotoptimize"(%[[PHI]], %[[PHI]])
  "custom.donotoptimize"(%0, %1) : (!iree_pydm.object, !iree_pydm.object) -> ()
  %none = none
  return %none : !iree_pydm.none
}
