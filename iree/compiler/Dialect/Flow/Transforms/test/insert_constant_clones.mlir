// RUN: iree-opt -split-input-file -pass-pipeline='builtin.func(iree-flow-insert-constant-clones)' %s | IreeFileCheck %s

// CHECK-LABEL: @function_return
func @function_return() -> (tensor<8xf32>, i32) {
  // CHECK-DAG: %[[SCALAR:.+]] = arith.constant 5
  %1 = arith.constant 5 : i32
  // CHECK-DAG: %[[CST:.+]] = arith.constant dense
  %cst = arith.constant dense<1.200000e+00> : tensor<8xf32>
  // CHECK-NEXT: %[[RESHAPE:.+]] = flow.tensor.reshape %[[CST]]
  %0 = flow.tensor.reshape %cst : tensor<8xf32> -> tensor<8xf32>
  // CHECK-NEXT: %[[CLONE:.+]] = flow.tensor.clone %[[RESHAPE]] : tensor<8xf32>
  // CHECK-NEXT: return %[[CLONE]], %[[SCALAR]]
  return %0, %1 : tensor<8xf32>, i32
}

// -----

// CHECK-LABEL: @branch_argument
func @branch_argument() -> tensor<8xf32> {
  // CHECK: %[[CST:.+]] = arith.constant dense
  %cst = arith.constant dense<1.200000e+00> : tensor<8xf32>
  // CHECK-NEXT: %[[CLONE:.+]] = flow.tensor.clone %[[CST]] : tensor<8xf32>
  // CHECK-NEXT: br ^[[EXIT:.+]](%[[CLONE]] : tensor<8xf32>
  br ^exit(%cst : tensor<8xf32>)
  // CHECK-NEXT: ^[[EXIT]](%[[BBARG:.+]]: tensor<8xf32>)
^exit(%0 : tensor<8xf32>):
  // CHECK-NEXT: = flow.tensor.reshape %[[CST]]
  %other_use = flow.tensor.reshape %cst : tensor<8xf32> -> tensor<8xf32>
  // CHECK-NEXT: return %[[BBARG]]
  return %0 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @branch_argument_reuse
func @branch_argument_reuse(%cond : i1) -> tensor<8xf32> {
  // CHECK: %[[CST:.+]] = arith.constant dense
  %cst = arith.constant dense<1.200000e+00> : tensor<8xf32>
  // CHECK-NEXT: %[[CLONE:.+]] = flow.tensor.clone %[[CST]] : tensor<8xf32>
  // CHECK-NEXT: cond_br %{{.+}}, ^[[BBT:.+]](%[[CLONE]] : tensor<8xf32>), ^[[BBF:.+]](%[[CLONE]] : tensor<8xf32>)
  cond_br %cond, ^exit_t(%cst : tensor<8xf32>), ^exit_f(%cst : tensor<8xf32>)
  // CHECK: ^[[BBT]](%[[BBARGT:.+]]: tensor<8xf32>)
^exit_t(%0 : tensor<8xf32>):
  // CHECK-NEXT: return %[[BBARGT]]
  return %0 : tensor<8xf32>
  // CHECK-NEXT: ^[[BBF]](%[[BBARGF:.+]]: tensor<8xf32>)
^exit_f(%1 : tensor<8xf32>):
  // CHECK-NEXT: return %[[BBARGF]]
  return %1 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @constant_variable
util.global private @constant_variable = dense<1.200000e+00> : tensor<8xf32>
func @constant_load() -> (tensor<8xf32>) {
  // CHECK: %[[VAR:.+]] = util.global.load @constant_variable
  %0 = util.global.load @constant_variable : tensor<8xf32>
  // CHECK-NEXT: %[[CLONE:.+]] = flow.tensor.clone %[[VAR]] : tensor<8xf32>
  // CHECK-NEXT: return %[[CLONE]]
  return %0 : tensor<8xf32>
}
