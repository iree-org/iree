// RUN:  iree-opt --iree-stablehlo-preprocessing-stateful-rng %s | FileCheck %s

// CHECK-LABEL:   func.func @main() -> tensor<1x1xi32> {
// CHECK:           %[[VAL_0:.*]] = ml_program.global_load @global_hlo_rng_state
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = "stablehlo.rng_bit_generator"(%[[VAL_0]])
// CHECK:           ml_program.global_store @global_hlo_rng_state = %[[VAL_1]]
// CHECK:           return %[[VAL_2]] : tensor<1x1xi32>
module {
  func.func @main() -> tensor<1x1xi32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.constant dense<5> : tensor<i32>
    %2 = stablehlo.constant dense<1> : tensor<2xi64>
    %3 = "stablehlo.rng"(%0, %1, %2) {rng_distribution = #stablehlo<rng_distribution UNIFORM>} : (tensor<i32>, tensor<i32>, tensor<2xi64>) -> tensor<1x1xi32>
    return %3 : tensor<1x1xi32>
  }
}

