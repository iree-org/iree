// RUN: iree-compile --compile-to=input --split-input-file %s | FileCheck %s

// Check that the auto input conversion pipeline uses this plugin.

// CHECK-LABEL: util.func public @simple_add_stablehlo
// CHECK:  arith.addi
func.func @simple_add_stablehlo(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = stablehlo.add %arg0, %arg1 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// -----

// Check that quantized StableHLO is lowered to primitive integer/float math.

// CHECK-LABEL: util.func public @qdq_linear
// CHECK-SAME: %{{.*}}: tensor<1x256xf32>
// CHECK-SAME: %{{.*}}: tensor<256x512xi8>
// CHECK: linalg.vecmat
// CHECK-NOT: !quant.uniform
// CHECK-NOT: stablehlo.uniform_quantize
// CHECK-NOT: stablehlo.uniform_dequantize
func.func @qdq_linear(
    %input: tensor<1x256xf32>,
    %weight: tensor<256x512x!quant.uniform<i8:f32, 0.01:0>>
  ) -> tensor<1x512xf32> {
  %quantized_input = stablehlo.uniform_quantize %input :
      (tensor<1x256xf32>) -> tensor<1x256x!quant.uniform<i8:f32, 0.01:0>>
  %matmul_result = stablehlo.dot_general %quantized_input, %weight,
      contracting_dims = [1] x [0] :
      (tensor<1x256x!quant.uniform<i8:f32, 0.01:0>>,
       tensor<256x512x!quant.uniform<i8:f32, 0.01:0>>) ->
      tensor<1x512x!quant.uniform<i32:f32, 0.0001:0>>
  %output = stablehlo.uniform_dequantize %matmul_result :
      (tensor<1x512x!quant.uniform<i32:f32, 0.0001:0>>) -> tensor<1x512xf32>
  return %output : tensor<1x512xf32>
}

// -----

// CHECK-LABEL: util.func public @vhlo_func
vhlo.func_v1 @vhlo_func(%arg0: !vhlo.tensor_v1<!vhlo.i32_v1>) -> (!vhlo.tensor_v1<!vhlo.i32_v1>) {
  // CHECK: arith.constant
  %0 = "vhlo.constant_v1"() <{value = #vhlo.tensor_v1<dense<1> : tensor<i32>>}> : () -> !vhlo.tensor_v1<!vhlo.i32_v1>
  // CHECK: return
  "vhlo.return_v1"(%0) : (!vhlo.tensor_v1<!vhlo.i32_v1>) -> ()
} {arg_attrs = #vhlo.array_v1<[]>, res_attrs = #vhlo.array_v1<[]>, sym_visibility = #vhlo.string_v1<"">}

// ----

// CHECK-LABEL: util.func public @dot_vhlo_example
vhlo.func_v1 @dot_vhlo_example(%arg0: !vhlo.tensor_v1<8x16x!vhlo.f32_v1>, %arg1: !vhlo.tensor_v1<16x8x!vhlo.f32_v1>) -> (!vhlo.tensor_v1<8x8x!vhlo.f32_v1>) {
  // CHECK: linalg.matmul
  %0 = "vhlo.dot_v1"(%arg0, %arg1) <{precision_config = #vhlo.array_v1<[#vhlo<precision_v1 DEFAULT>, #vhlo<precision_v1 DEFAULT>]>}> : (!vhlo.tensor_v1<8x16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x8x!vhlo.f32_v1>) -> !vhlo.tensor_v1<8x8x!vhlo.f32_v1>
  // CHECK: return
  "vhlo.return_v1"(%0) : (!vhlo.tensor_v1<8x8x!vhlo.f32_v1>) -> ()
} {arg_attrs = #vhlo.array_v1<[]>, res_attrs = #vhlo.array_v1<[]>, sym_visibility = #vhlo.string_v1<"">}

// -----

// CHECK-LABEL: util.func public @gather_vhlo_example
vhlo.func_v1 @gather_vhlo_example(%arg0: !vhlo.tensor_v1<16x8x!vhlo.f32_v1>) -> (!vhlo.tensor_v1<16x16x!vhlo.f32_v1>) {
  // CHECK: flow.collective.all_gather
  %0 = "vhlo.all_gather_v1"(%arg0) <{all_gather_dim = #vhlo.integer_v1<1 : i64>, channel_id = #vhlo.integer_v1<0 : i64>, replica_groups = #vhlo.tensor_v1<dense<[[0], [1]]> : tensor<2x1xi64>>, use_global_device_ids = #vhlo.bool_v1<false>}> : (!vhlo.tensor_v1<16x8x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x16x!vhlo.f32_v1>
  // CHECK: return
  "vhlo.return_v1"(%0) : (!vhlo.tensor_v1<16x16x!vhlo.f32_v1>) -> ()
} {arg_attrs = #vhlo.array_v1<[]>, res_attrs = #vhlo.array_v1<[]>, sym_visibility = #vhlo.string_v1<"">}

// -----

// CHECK-LABEL: util.func public @compare_vhlo_example
vhlo.func_v1 @compare_vhlo_example(%arg0: !vhlo.tensor_v1<!vhlo.f32_v1>, %arg1: !vhlo.tensor_v1<!vhlo.f32_v1>) -> (!vhlo.tensor_v1<!vhlo.bool_v1>) {
  // CHECK: arith.cmpf
  %0 = "vhlo.compare_v1"(%arg0, %arg1) <{compare_type = #vhlo<comparison_type_v1 NOTYPE>, comparison_direction = #vhlo<comparison_direction_v1 EQ>}> : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.bool_v1>
  // CHECK: return
  "vhlo.return_v1"(%0) : (!vhlo.tensor_v1<!vhlo.bool_v1>) -> ()
} {arg_attrs = #vhlo.array_v1<[]>, res_attrs = #vhlo.array_v1<[]>, sym_visibility = #vhlo.string_v1<"">}
