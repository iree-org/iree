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

// -----

// A tensor encoding must trigger conversion without any StableHLO operations.
// CHECK-LABEL: util.func public @bounds_identity
// CHECK-SAME: (%{{.+}}: tensor<?xf32>) -> tensor<?xf32>
// CHECK: util.assume.int %{{.+}}<umax = 16>
func.func @bounds_identity(%a: tensor<?xf32, #stablehlo.bounds<16>>) -> tensor<?xf32, #stablehlo.bounds<16>> {
  return %a : tensor<?xf32, #stablehlo.bounds<16>>
}

// -----

// Nested encodings must select the tuple-flattening pipeline.
// CHECK-LABEL: util.func public @bounds_tuple_identity
// CHECK-SAME: (%{{.+}}: tensor<?xf32>) -> tensor<?xf32>
// CHECK: util.assume.int %{{.+}}<umax = 16>
func.func @bounds_tuple_identity(%a: tuple<tuple<tensor<?xf32, #stablehlo.bounds<16>>>>) -> tuple<tuple<tensor<?xf32, #stablehlo.bounds<16>>>> {
  return %a : tuple<tuple<tensor<?xf32, #stablehlo.bounds<16>>>>
}

// -----

// A tuple alone does not select an input dialect conversion.
// CHECK-LABEL: util.func public @plain_tuple_identity
// CHECK-SAME: (%[[VALUE:.+]]: tuple<tensor<?xf32>>) -> tuple<tensor<?xf32>>
// CHECK: util.return %[[VALUE]] : tuple<tensor<?xf32>>
func.func @plain_tuple_identity(%a: tuple<tensor<?xf32>>) -> tuple<tensor<?xf32>> {
  return %a : tuple<tensor<?xf32>>
}

// -----

// An unused input has no SSA use exposing its bounds to the detector.
// CHECK-LABEL: util.func public @unused_bounded_input
// CHECK-SAME: (%{{.+}}: tensor<?xf32>) -> i32
func.func @unused_bounded_input(%arg: tensor<?xf32, #stablehlo.bounds<16>>) -> i32 {
  %c = arith.constant 0 : i32
  return %c : i32
}

// -----

// External inputs have no entry block exposing their types.
// CHECK-LABEL: util.func private @external_bounds_input(%{{.+}}: tensor<?xf32>)
func.func private @external_bounds_input(tensor<?xf32, #stablehlo.bounds<16>>)
