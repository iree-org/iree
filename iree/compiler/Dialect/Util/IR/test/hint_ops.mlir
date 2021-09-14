// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @parse_print_do_not_optimize
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
func @parse_print_do_not_optimize(%arg0 : tensor<i32>, %arg1 : tensor<i32>) {
  // CHECK-NEXT: util.do_not_optimize(%[[ARG0]]) : tensor<i32>
  %1 = util.do_not_optimize(%arg0) : tensor<i32>

  // CHECK-NEXT: util.do_not_optimize(%[[ARG0]], %[[ARG1]]) : tensor<i32>, tensor<i32>
  %2:2 = util.do_not_optimize(%arg0, %arg1) : tensor<i32>, tensor<i32>

  // CHECK-NEXT: util.do_not_optimize(%[[ARG0]]) {some_unit} : tensor<i32>
  %has_attr = util.do_not_optimize(%arg0) {some_unit} : tensor<i32>

  return
}

// -----

// CHECK-LABEL: @parse_print_unfoldable_constant
func @parse_print_unfoldable_constant(%arg0 : tensor<i32>, %arg1 : tensor<i32>) {
  // CHECK-NEXT: util.unfoldable_constant 42
  %c42 = util.unfoldable_constant 42 : i32

  // CHECK: util.unfoldable_constant {attr = "foo"} 43 : i32
  %cattr = util.unfoldable_constant {attr = "foo"} 43 : i32

  // CHECK: util.unfoldable_constant @func_with_args : (f32) -> ()
  %csymref = util.unfoldable_constant @func_with_args : (f32) -> ()

  return
}

// -----

// CHECK-LABEL: @parse_print_dynamic_shape_constant
func @parse_print_dynamic_shape_constant() {
  // CHECK-NEXT: util.dynamic_shape_constant dense<2> : tensor<2xi32> -> tensor<?xi32>
  %c = util.dynamic_shape_constant dense<2> : tensor<2xi32> -> tensor<?xi32>

  // CHECK-NEXT: util.dynamic_shape_constant dense<2> : tensor<2xi32> {attr = "foo"} -> tensor<?xi32>
  %has_attr = util.dynamic_shape_constant dense<2> : tensor<2xi32> {attr = "foo"} -> tensor<?xi32>
  return
}
