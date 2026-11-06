// torch.cumsum over a symbolic axis, exported through Torchax: JAX emits the
// window as the stablehlo.dynamic_reduce_window custom call. One module runs
// at two lengths.

// RUN: iree-compile %s --iree-input-type=stablehlo --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic -o %t.vmfb
// RUN: iree-run-module --device=local-task --module=%t.vmfb --function=main \
// RUN:   --input="3x2xi32=[1 2][3 -4][5 6]" \
// RUN:   --expected_output="3x2xi32=[1 3][3 -1][5 11]" | FileCheck %s
// RUN: iree-run-module --device=local-task --module=%t.vmfb --function=main \
// RUN:   --input="3x5xi32=[1 2 3 4 5][-1 -2 -3 -4 -5][7 0 7 0 7]" \
// RUN:   --expected_output="3x5xi32=[1 3 6 10 15][-1 -3 -6 -10 -15][7 7 14 14 21]" | FileCheck %s

// CHECK: [SUCCESS]
module @jit_func attributes {jax.uses_shape_polymorphism = true, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<3x?xi32>) -> (tensor<3x?xi32> {jax.result_info = "result[1][0]"}) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<64> : tensor<i64>
    %c_1 = stablehlo.constant dense<-1> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<3x?xi32>) -> tensor<i32>
    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<i64>
    %2 = stablehlo.compare GE, %1, %c_2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%2, %1) {api_version = 2 : i32, error_message = "Input shapes do not match the polymorphic shapes specification. Expected value >= 1 for dimension variable 's27'. Using the following polymorphic shapes specifications: args[2][0].shape = (3, s27). Obtained dimension variables: 's27' = {0} from specification 's27' for dimension args[2][0].shape[1] (= {0}), . Please see https://docs.jax.dev/en/latest/export/shape_poly.html#shape-assertion-errors for more details.", has_side_effect = true} : (tensor<i1>, tensor<i64>) -> ()
    %3 = stablehlo.convert %1 : tensor<i64>
    %4 = stablehlo.multiply %3, %c_1 : tensor<i64>
    %5 = stablehlo.add %4, %c_0 : tensor<i64>
    %6 = stablehlo.compare GE, %5, %c, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %7 = stablehlo.convert %1 : tensor<i64>
    %8 = stablehlo.multiply %7, %c_1 : tensor<i64>
    %9 = stablehlo.add %8, %c_0 : tensor<i64>
    stablehlo.custom_call @shape_assertion(%6, %9, %1) {api_version = 2 : i32, error_message = "Input shapes do not match the symbolic shape constraint s27 <= 64. Expected '- s27 + 64' to be greater or equal to 0, but found {0}.  Using the following polymorphic shapes specifications: args[2][0].shape = (3, s27). Obtained dimension variables: 's27' = {1} from specification 's27' for dimension args[2][0].shape[1] (= {1}), . Please see https://docs.jax.dev/en/latest/export/shape_poly.html#shape-assertion-errors for more details.", has_side_effect = true} : (tensor<i1>, tensor<i64>, tensor<i64>) -> ()
    %10 = call @_wrapped_jax_export_main(%1, %arg0) : (tensor<i64>, tensor<3x?xi32>) -> tensor<3x?xi32>
    return %10 : tensor<3x?xi32>
  }
  func.func @reduce_window_int32_reducer(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @_wrapped_jax_export_main(%arg0: tensor<i64> {jax.global_constant = "s27"}, %arg1: tensor<3x?xi32>) -> (tensor<3x?xi32> {jax.result_info = "result[1][0]"}) {
    %0 = call @cumsum(%arg0, %arg1) : (tensor<i64>, tensor<3x?xi32>) -> tensor<3x?xi32>
    return %0 : tensor<3x?xi32>
  }
  func.func private @cumsum(%arg0: tensor<i64> {jax.global_constant = "s27"}, %arg1: tensor<3x?xi32>) -> tensor<3x?xi32> {
    %0 = call @cumsum_0(%arg0, %arg1) : (tensor<i64>, tensor<3x?xi32>) -> tensor<3x?xi32>
    return %0 : tensor<3x?xi32>
  }
  func.func private @cumsum_0(%arg0: tensor<i64>, %arg1: tensor<3x?xi32>) -> tensor<3x?xi32> {
    %c = stablehlo.constant dense<1> : tensor<1xi32>
    %c_0 = stablehlo.constant dense<-1> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<i32>
    %1 = stablehlo.concatenate %c_1, %c_1, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %2 = stablehlo.reshape %1 : (tensor<2xi32>) -> tensor<1x2xi32>
    %3 = stablehlo.convert %arg0 : tensor<i64>
    %4 = stablehlo.add %3, %c_0 : tensor<i64>
    %5 = stablehlo.convert %4 : (tensor<i64>) -> tensor<i32>
    %6 = stablehlo.reshape %5 : (tensor<i32>) -> tensor<1xi32>
    %7 = stablehlo.concatenate %6, %c_1, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %8 = stablehlo.reshape %7 : (tensor<2xi32>) -> tensor<1x2xi32>
    %9 = stablehlo.concatenate %2, %8, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
    %10 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %11 = stablehlo.reshape %10 : (tensor<i32>) -> tensor<1xi32>
    %12 = stablehlo.concatenate %c, %11, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %13 = stablehlo.concatenate %c, %c, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %14 = stablehlo.concatenate %c, %c, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %15 = stablehlo.concatenate %c, %c, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %16 = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%arg1, %0, %12, %13, %14, %15, %9) {api_version = 2 : i32, called_computations = [@reduce_window_int32_reducer]} : (tensor<3x?xi32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<3x?xi32>
    return %16 : tensor<3x?xi32>
  }
}
