// RUN: iree-dialects-opt -split-input-file -convert-iree-pydm-to-iree %s | FileCheck --enable-var-scope --dump-input-filter=all %s

// CHECK-LABEL: @none_constant
iree_pydm.func @none_constant() -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK: %[[CST0:.*]] = constant 0 : i32
  // CHECK: %[[CST1:.*]] = constant 0 : i32
  // CHECK: return %[[CST1]], %[[CST0]]
  %0 = none
  return %0 : !iree_pydm.none
}

// -----
// CHECK-LABEL: @box
// NOTE: "9" is the type code for integer
iree_pydm.func @box(%arg0 : !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.object<!iree_pydm.integer>) {
  // CHECK: %[[LIST:.*]] = iree.list.create : !iree.list<!iree.variant>
  // CHECK: %[[c2:.*]] = constant 2 : index
  // CHECK: iree.list.resize %[[LIST]], %c2 : !iree.list<!iree.variant>
  // CHECK: %[[c0:.*]] = constant 0 : index
  // CHECK: %[[c9:.*]] = constant 9 : i32
  // CHECK: iree.list.set %[[LIST]][%[[c0]]], %[[c9]] : !iree.list<!iree.variant>, i32
  // CHECK: %[[c1:.*]] = constant 1 : index
  // CHECK: iree.list.set %[[LIST]][%[[c1]]], %arg0 : !iree.list<!iree.variant>, i32
  // CHECK: %[[c0_i32:.*]] = constant 0 : i32
  // return %[[c0_i32]], %[[LIST]] : i32, !iree.list<!iree.variant>
  %0 = box %arg0 : !iree_pydm.integer -> !iree_pydm.object<!iree_pydm.integer>
  return %0 : !iree_pydm.object<!iree_pydm.integer>
}

// -----
// CHECK-LABEL: @alloc_store_load_var
// NOTE: 256 is the type code for a plain object
iree_pydm.func @alloc_store_load_var(%arg0 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.object) {
  // CHECK: %[[A:.*]] = iree.list.create : !iree.list<!iree.variant>
  %a = alloc_free_var "a" -> !iree_pydm.free_var_ref
  // CHECK: %[[c2:.*]] = constant 2 : index
  // CHECK: iree.list.resize %[[A]], %[[c2]] : !iree.list<!iree.variant>
  // CHECK: %[[c0:.*]] = constant 0 : index
  // CHECK: %[[object_code:.*]] = constant 256 : i32
  // CHECK: iree.list.set %[[A]][%[[c0]]], %[[object_code]]
  // CHECK: %[[c1:.*]] = constant 1 : index
  // CHECK: iree.list.set %[[A]][%[[c1]]], %arg0 : !iree.list<!iree.variant>, !iree.list<!iree.variant>
  store_var %a = %arg0 : !iree_pydm.free_var_ref, !iree_pydm.object
  // CHECK: %[[c1_0:.*]] = constant 1 : index
  // CHECK: %[[LOADED:.*]] = iree.list.get %[[A]][%[[c1_0]]] : !iree.list<!iree.variant> -> !iree.list<!iree.variant>
  %0 = load_var %a : !iree_pydm.free_var_ref -> !iree_pydm.object
  // CHECK: return {{.*}}, %[[LOADED]]
  return %0 : !iree_pydm.object
}

// -----
// CHECK-LABEL: @unbox
// NOTE: "9" is the type code for integer
iree_pydm.func @unbox(%arg0 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.integer) {
  // CHECK:  %[[c0:.*]] = constant 0 : index
  // CHECK: %[[NEEDED_TYPE_CODE:.*]] = constant 9 : i32
  // CHECK: %[[TYPE_CODE:.*]] = iree.list.get %arg0[%[[c0]]] : !iree.list<!iree.variant> -> i32
  // CHECK: %[[TYPE_EQ:.*]] = cmpi eq, %[[NEEDED_TYPE_CODE]], %[[TYPE_CODE]] : i32
  // CHECK: cond_br %[[TYPE_EQ]], ^bb1, ^bb4

  // bb1: On equal
  // CHECK: ^bb1:
  // CHECK: %[[c1:.*]] = constant 1 : index
  // CHECK: %[[c0_i32:.*]] = constant 0 : i32
  // CHECK: %[[CONTENTS:.*]] = iree.list.get %arg0[%[[c1]]] : !iree.list<!iree.variant> -> i32
  // CHECK: br ^bb2(%[[c0_i32]], %[[CONTENTS]] : i32, i32)

  // bb2: Check status code (from raise_on_failure)
  // CHECK: ^bb2(%3: i32, %4: i32):  // 2 preds: ^bb1, ^bb4

  // bb3: Return success
  // CHECK: ^bb3

  // bb4: Signal ValueError (-4 == ValueError)
  // CHECK: ^bb4:
  // CHECK: %[[VALUE_ERROR_CODE:.*]] = constant -4 : i32
  // CHECK: %[[c0_i32_2:.*]] = constant 0 : i32
  // CHECK: br ^bb2(%[[VALUE_ERROR_CODE]], %[[c0_i32_2]] : i32, i32)
  %status, %primitive = unbox %arg0 : !iree_pydm.object -> !iree_pydm.integer
  raise_on_failure %status : !iree_pydm.exception_result
  return %primitive : !iree_pydm.integer
}

// -----
// CHECK-LABEL: @raise_on_failure_object_return
iree_pydm.func @raise_on_failure_object_return(%arg0 : !iree_pydm.exception_result, %arg1: !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.object) {
  // CHECK: %[[c0_i32:.*]] = constant 0 : i32
  // CHECK: %[[CMP:.*]] = cmpi eq, %[[c0_i32]], %arg0 : i32
  // CHECK: cond_br %[[CMP]], ^bb1, ^bb2
  // bb1: success
  // CHECK: ^bb1:
  // CHECK: %[[c0_i32_0:.*]] = constant 0 : i32
  // CHECK: return %[[c0_i32_0]], %arg1 : i32, !iree.list<!iree.variant>
  // bb2: failure
  // CHECK: ^bb2:
  // CHECK: %[[NULL:.*]] = iree.null : !iree.list<!iree.variant>
  // CHECK: return %arg0, %[[NULL]] : i32, !iree.list<!iree.variant>
  raise_on_failure %arg0 : !iree_pydm.exception_result
  return %arg1 : !iree_pydm.object
}

// -----
// CHECK-LABEL: @raise_on_failure_builtin
iree_pydm.func @raise_on_failure_builtin(%arg0 : !iree_pydm.exception_result, %arg1: !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.integer) {
  // bb2: failure
  // CHECK: ^bb2:
  // CHECK: %[[ZERO:.*]] = constant 0 : i32
  // CHECK: return %arg0, %[[ZERO]] : i32, i32
  raise_on_failure %arg0 : !iree_pydm.exception_result
  return %arg1 : !iree_pydm.integer
}
