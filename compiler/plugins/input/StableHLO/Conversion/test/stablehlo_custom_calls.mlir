// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-stablehlo-legalize-custom-calls),cse)" \
// RUN:   %s | FileCheck %s

// CHECK-LABEL: @householder
func.func public @householder(%arg0: tensor<4x3xf32>, %arg1: tensor<2xf32>) -> (tensor<4x3xf32>) {
  // CHECK: linalg.generic
  // CHECK: scf.for
  // CHECK:   linalg.generic
  // CHECK:   stablehlo.dot_general
  %0 = stablehlo.custom_call @ProductOfElementaryHouseholderReflectors(%arg0, %arg1) : (tensor<4x3xf32>, tensor<2xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// -----

// CHECK-LABEL: @noop_sharding_custom_call
func.func public @noop_sharding_custom_call(%arg0: tensor<2xui32>) {
  %result = "stablehlo.custom_call"(%arg0) <{call_target_name = "Sharding"}> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@empty_mesh, [{}]>]>"}, mhlo.sharding = "{replicated}"} : (tensor<2xui32>) -> tensor<2xui32>
  // CHECK-NOT: stablehlo.custom_call
  // CHECK-NOT: sharding
  return
}
