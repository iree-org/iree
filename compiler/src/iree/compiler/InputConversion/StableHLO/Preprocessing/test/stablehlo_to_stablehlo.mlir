// RUN: iree-opt --iree-stablehlo-to-stablehlo-preprocessing \
// RUN:   --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @batch_norm_inference
// CHECK-SAME: %[[X:[^:[:space:]]+]]
// CHECK-SAME: %[[SCALE:[^:[:space:]]+]]
// CHECK-SAME: %[[OFFSET:[^:[:space:]]+]]
// CHECK-SAME: %[[MEAN:[^:[:space:]]+]]
// CHECK-SAME: %[[VARIANCE:[^:[:space:]]+]]
func.func @batch_norm_inference(
    %x: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>,
    %mean: tensor<256xf32>, %variance: tensor<256xf32>)
    -> (tensor<4x256xf32>) {
  // CHECK-DAG: %[[EPS_BCAST:.+]] = stablehlo.constant dense<1.001000e-05> : tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = stablehlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<256xf32>
  // CHECK-DAG: %[[STDDEV:.+]] = stablehlo.sqrt %[[VARIANCE_EPS]] : tensor<256xf32>
  // CHECK-DAG: %[[STDDEV_BCAST:.+]] = stablehlo.broadcast_in_dim %[[STDDEV]], dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[SCALE_BCAST:.+]] = stablehlo.broadcast_in_dim %[[SCALE]], dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[OFFSET_BCAST:.+]] = stablehlo.broadcast_in_dim %[[OFFSET]], dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[MEAN_BCAST:.+]] = stablehlo.broadcast_in_dim %[[MEAN]], dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[X_CENTER:.+]] = stablehlo.subtract %[[X]], %[[MEAN_BCAST]] : tensor<4x256xf32>
  // CHECK:     %[[X_SCALED:.+]] = stablehlo.multiply %[[X_CENTER]], %[[SCALE_BCAST]] : tensor<4x256xf32>
  // CHECK:     %[[X_NORMED:.+]] = stablehlo.divide %[[X_SCALED]], %[[STDDEV_BCAST]] : tensor<4x256xf32>
  // CHECK:     %[[RESULT:.+]] = stablehlo.add %[[X_NORMED]], %[[OFFSET_BCAST]] : tensor<4x256xf32>
  %0 = "stablehlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-NEXT: return %[[RESULT]]
  return %0 : tensor<4x256xf32>
}

// -----

// CHECK-LABEL: @rng_bitcast_f32 
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<4xi32>)
func.func @rng_bitcast_f32(%arg0: tensor<4xi32>) -> (tensor<4xi32>, tensor<8xf32>) {
  // CHECK: %[[OUT_STATE:.*]], %[[OUT_INT:.*]] = stablehlo.rng_bit_generator %[[ARG0]]
  // CHECK-SAME: -> (tensor<4xi32>, tensor<8xi32>)
  %output_state, %output = "stablehlo.rng_bit_generator"(%arg0) {rng_algorithm = #stablehlo<rng_algorithm PHILOX>} : (tensor<4xi32>) -> (tensor<4xi32>, tensor<8xf32>)
  // CHECK: %[[OUT_FLOAT:.*]] stablehlo.bitcast_convert %[[OUT_INT:.*]] -> tensor<8xf32>
  // CHECK: return %[[OUT_STATE:.*]], %[[OUT_FLOAT:.*]]
  return %output_state, %output : tensor<4xi32>, tensor<8xf32>
}

// -----

// CHECK-LABEL: func.func @rng_normal
// CHECK-SAME:              (%[[ARG0:.+]]: tensor<f32>, %[[ARG1:.+]]: tensor<f32>)
func.func @rng_normal(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<3x5xf32> {
  %shape = stablehlo.constant dense<[3, 5]> : tensor<2xi64>
  %0 = "stablehlo.rng"(%arg0, %arg1, %shape) {rng_distribution = #stablehlo<rng_distribution NORMAL>} : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}
// CHECK:         %{{.+}} = stablehlo.constant dense<{{.+}}> : tensor<8xf32>
// CHECK:         %{{.+}} = stablehlo.constant dense<{{.+}}> : tensor<8xf32>
// CHECK:         %{{.+}} = stablehlo.constant dense<{{.+}}> : tensor<8xf32>
// CHECK:         %[[SIGMA:.+]] = stablehlo.broadcast %[[ARG1]], sizes = [8] : (tensor<f32>) -> tensor<8xf32>
//
//                mag = sigma * sqrt(-2.0 * log(u1)) where sqrt values are
//                constants.
//
// CHECK:         %[[MAG:.+]] = stablehlo.multiply %[[SIGMA]], %{{.+}} : tensor<8xf32>
//
//                z0  = mag * cos(two_pi * u2) + mu;
//                z1  = mag * sin(two_pi * u2) + mu;
//
// CHECK:         %[[MU:.+]] = stablehlo.broadcast %[[ARG0]], sizes = [8] : (tensor<f32>) -> tensor<8xf32>
// CHECK:         %[[T1:.+]] = stablehlo.multiply %[[MAG]], %{{.+}} : tensor<8xf32>
// CHECK:         %[[Z0:.+]] = stablehlo.add %[[T1:.+]], %[[MU]] : tensor<8xf32>
// CHECK:         %[[T2:.+]] = stablehlo.multiply %[[MAG]], %{{.+}} : tensor<8xf32>
// CHECK:         %[[Z1:.+]] = stablehlo.add %[[T2:.+]], %[[MU]] : tensor<8xf32>
//
//                Concate and reshape the output.
// CHECK:         %[[CON:.+]] = stablehlo.concatenate %[[Z0]], %[[Z1]], dim = 0 : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>
// CHECK:         %[[SLICE:.+]] = tensor.extract_slice %[[CON]][0] [15] [1] : tensor<16xf32> to tensor<15xf32>
// CHECK:         %[[RES:.+]] = stablehlo.reshape %[[SLICE]] : (tensor<15xf32>) -> tensor<3x5xf32>
// CHECK:         return %[[RES]]

// -----

// CHECK-LABEL: @mul_float_bool_cast
func.func @mul_float_bool_cast(%arg0 : tensor<?xi1>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<?xi1>) -> tensor<?xf32>
  %1 = "stablehlo.multiply"(%0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[BTOF:.+]] = stablehlo.convert %arg0 : (tensor<?xi1>) -> tensor<?xf32>
// CHECK: %[[FTOB:.+]] = stablehlo.convert %[[BTOF]] : (tensor<?xf32>) -> tensor<?xi1>
// CHECK: %[[SHP:.+]] = shape.shape_of %[[BTOF]] : tensor<?xf32> -> tensor<1xindex>
// CHECK: %[[BROADCAST:.+]] = stablehlo.dynamic_broadcast_in_dim %[[ZERO]], %[[SHP]], dims = []
// CHECK: %[[SELECT:.+]] = stablehlo.select %[[FTOB]], %arg1, %[[BROADCAST]]

// -----

// CHECK-LABEL: @mul_float_bool_cast_broadcast
func.func @mul_float_bool_cast_broadcast(%arg0: tensor<5xi1>, %arg1: tensor<5x6xf32>) -> tensor<5x6xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<5xi1>) -> tensor<5xf32>
  %1 = "stablehlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<5xf32>) -> tensor<5x6xf32>
  %2 = stablehlo.multiply %1, %arg1 : tensor<5x6xf32>
  return %2 : tensor<5x6xf32>
}

// CHECK: stablehlo.select

// -----

// CHECK-LABEL: @mul_float_bool_cast_dyn_broadcast
func.func @mul_float_bool_cast_dyn_broadcast(%arg0: tensor<?xi1>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<?xi1>) -> tensor<?xf32>
    %1 = shape.shape_of %arg1 : tensor<?x?xf32> -> tensor<2xindex>
    %2 = "stablehlo.dynamic_broadcast_in_dim"(%0, %1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
    %3 = stablehlo.multiply %2, %arg1 : tensor<?x?xf32>
    return %3 : tensor<?x?xf32>
}

// CHECK: stablehlo.select

// -----

// CHECK-LABEL: @dot_general_fuse_both_with_attrs
// CHECK-SAME:     (%[[ARG0:.+]]: tensor<16x64x128xf16>, %[[ARG1:.+]]: tensor<16x128x3072xf16>)
func.func @dot_general_fuse_both_with_attrs(%arg0: tensor<16x64x128xf16>, %arg1: tensor<16x128x3072xf16>) -> tensor<16x64x3072xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<16x64x128xf16>) -> tensor<16x64x128xf32>
  %1 = stablehlo.convert %arg1 : (tensor<16x128x3072xf16>) -> tensor<16x128x3072xf32>
  // CHECK: stablehlo.dot_general %[[ARG0]], %[[ARG1]]
  // CHECK-SAME: batching_dims = [0] x [0]
  // CHECK-SAME: contracting_dims = [2] x [1]
  // CHECK-SAME: precision = [DEFAULT, DEFAULT]
  // CHECK-SAME: -> tensor<16x64x3072xf32>
  %2 = "stablehlo.dot_general"(%0, %1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x64x128xf32>, tensor<16x128x3072xf32>) -> tensor<16x64x3072xf32>
  return %2 : tensor<16x64x3072xf32>
}

// -----

// CHECK-LABEL: @dot_general_fuse_one
// CHECK-SAME:     (%[[ARG0:.+]]: tensor<{{.+}}xf64>, %[[ARG1:.+]]: tensor<{{.+}}xf16>)
func.func @dot_general_fuse_one(%arg0: tensor<16x64x128xf64>, %arg1: tensor<16x128x3072xf16>) -> tensor<16x64x3072xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<16x64x128xf64>) -> tensor<16x64x128xf32>
  %1 = stablehlo.convert%arg1 : (tensor<16x128x3072xf16>) -> tensor<16x128x3072xf32>
  // CHECK: %[[CONVERT:.+]] = stablehlo.convert %[[ARG0]]
  // CHECK: stablehlo.dot_general %[[CONVERT]], %[[ARG1]]
  %2 = "stablehlo.dot_general"(%0, %1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x64x128xf32>, tensor<16x128x3072xf32>) -> tensor<16x64x3072xf32>
  return %2 : tensor<16x64x3072xf32>
}

// -----

// CHECK-LABEL: @dot_basic
// CHECK-SAME:     (%[[ARG0:.+]]: tensor<4x4xf16>, %[[ARG1:.+]]: tensor<4x4xf16>)
func.func @dot_basic(%arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf16>) -> tensor<4x4xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xf16>) -> tensor<4x4xf32>
  %1 = stablehlo.convert %arg1 : (tensor<4x4xf16>) -> tensor<4x4xf32>
  // CHECK: %[[DOT:.+]] = stablehlo.dot %[[ARG0]], %[[ARG1]]
  %2 = "stablehlo.dot"(%0, %1) {precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK: return %[[DOT]]
  return %2 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @convolution
// CHECK-SAME:     (%[[ARG0:.+]]: tensor<{{.+}}xbf16>, %[[ARG1:.+]]: tensor<{{.+}}xbf16>)
func.func @convolution(%arg0: tensor<16x32x256xbf16>, %arg1: tensor<1x256x256xbf16>) -> tensor<16x32x256xf32> {
  %cast = stablehlo.convert %arg0 : (tensor<16x32x256xbf16>) -> tensor<16x32x256xf32>
  // CHECK: %[[CONV:.+]] = stablehlo.convolution(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: -> tensor<16x32x256xf32>
  %0 = "stablehlo.convolution"(%cast, %arg1) {
     batch_group_count = 1 : i64,
     dimension_numbers = #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>,
     feature_group_count = 1 : i64,
     lhs_dilation = dense<1> : tensor<1xi64>,
     padding = dense<0> : tensor<1x2xi64>,
     precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
     rhs_dilation = dense<1> : tensor<1xi64>,
     window_strides = dense<1> : tensor<1xi64>
   } : (tensor<16x32x256xf32>, tensor<1x256x256xbf16>) -> tensor<16x32x256xf32>
  // CHECK: return %[[CONV]]
  func.return %0 : tensor<16x32x256xf32>
}

// -----

// CHECK-LABEL: @dynamic_dot_general
// This verifies non-crashing, the lowering to linalg happens elsewhere.
func.func @dynamic_dot_general(%arg1: tensor<?x1024x16x64xf32>, %arg2: tensor<?x1024x16x64xf32>) -> tensor<?x16x1024x1024xf32> {
  %2 = "stablehlo.dot_general"(%arg2, %arg1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0, 2], rhs_batching_dimensions = [0, 2], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [3]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<?x1024x16x64xf32>, tensor<?x1024x16x64xf32>) -> tensor<?x16x1024x1024xf32>
  return %2 : tensor<?x16x1024x1024xf32>
}

// -----

func.func @custom_call_topk_tuple(%arg0: tensor<4x8000xbf16>) -> (tensor<4x40xbf16>, tensor<4x40xi32>) {
  %0 = stablehlo.custom_call @TopK(%arg0) {called_computations = [@comparison], xla_shape = "(bf16[4,40]{1,0}, s32[4,40]{1,0})"} : (tensor<4x8000xbf16>) -> tuple<tensor<4x40xbf16>, tensor<4x40xi32>>
  %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<4x40xbf16>, tensor<4x40xi32>>) -> tensor<4x40xbf16>
  %2 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<4x40xbf16>, tensor<4x40xi32>>) -> tensor<4x40xi32>
  return %1, %2 : tensor<4x40xbf16>, tensor<4x40xi32>
}
func.func private @comparison(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = stablehlo.compare  GT, %arg0, %arg1,  TOTALORDER : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}

// CHECK-LABEL: @custom_call_topk_tuple
// CHECK-SAME: %[[ARG0:[a-z0-9]+]]
// CHECK: %[[VALUES:.+]], %[[INDICES:.+]] = chlo.top_k(%[[ARG0]], k = 40) : tensor<4x8000xbf16> -> (tensor<4x40xbf16>, tensor<4x40xi32>)
// CHECK: return %[[VALUES]], %[[INDICES]] : tensor<4x40xbf16>, tensor<4x40xi32>

// -----

func.func @custom_call_topk_returns(%arg0: tensor<4x8000xbf16>) -> (tensor<4x40xbf16>, tensor<4x40xi32>) {
  %0:2 = stablehlo.custom_call @TopK(%arg0) {called_computations = [@comparison], xla_shape = "(bf16[4,40]{1,0}, s32[4,40]{1,0})"} : (tensor<4x8000xbf16>) -> (tensor<4x40xbf16>, tensor<4x40xi32>)
  return %0#0, %0#1 : tensor<4x40xbf16>, tensor<4x40xi32>
}
func.func private @comparison(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = stablehlo.compare  GT, %arg0, %arg1,  TOTALORDER : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}

// CHECK-LABEL: @custom_call_topk_returns
// CHECK-SAME: %[[ARG0:[a-z0-9]+]]
// CHECK: %[[VALUES:.+]], %[[INDICES:.+]] = chlo.top_k(%[[ARG0]], k = 40) : tensor<4x8000xbf16> -> (tensor<4x40xbf16>, tensor<4x40xi32>)
// CHECK: return %[[VALUES]], %[[INDICES]] : tensor<4x40xbf16>, tensor<4x40xi32>

// -----

func.func @iota_sort_slice_is_topk(%in : tensor<16x16xf32>) -> (tensor<16x8xf32>, tensor<16x8xi32>) {
  %iota = "stablehlo.iota"() { iota_dimension = 1 : i64 } : () -> tensor<16x16xi32>
  %0:2 = "stablehlo.sort"(%in, %iota) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  %1 = "stablehlo.slice"(%0#0) { start_indices = dense<[0, 0]> : tensor<2xi64>, limit_indices = dense<[16, 8]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64> } : (tensor<16x16xf32>) -> tensor<16x8xf32> 
  %2 = "stablehlo.slice"(%0#1) { start_indices = dense<[0, 0]> : tensor<2xi64>, limit_indices = dense<[16, 8]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64> } : (tensor<16x16xi32>) -> tensor<16x8xi32>
  return %1, %2 : tensor<16x8xf32>, tensor<16x8xi32>
}

// CHECK-LABEL: @iota_sort_slice_is_topk
// CHECK-SAME: %[[IN:[a-z0-9]+]]
// CHECK: %[[VALUES:.+]], %[[INDICES:.+]] = chlo.top_k(%[[IN]], k = 8) : tensor<16x16xf32> -> (tensor<16x8xf32>, tensor<16x8xi32>)
// CHECK: return %[[VALUES]], %[[INDICES]] : tensor<16x8xf32>, tensor<16x8xi32>

// -----

func.func @broadcast_iota_sort_slice_is_topk(%in : tensor<16x16x16xf32>) -> (tensor<16x16x8xf32>, tensor<16x16x8xi32>) {
  %iota = "stablehlo.iota"() { iota_dimension = 0 : i64 } : () -> tensor<16xi32>
  %broadcasted_0 = "stablehlo.broadcast_in_dim"(%iota) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<16xi32>) -> tensor<16x16xi32>
  %broadcasted_1 = "stablehlo.broadcast_in_dim"(%broadcasted_0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<16x16xi32>) -> tensor<16x16x16xi32>
  %0:2 = "stablehlo.sort"(%in, %broadcasted_1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 2 : i64, is_stable = true} : (tensor<16x16x16xf32>, tensor<16x16x16xi32>) -> (tensor<16x16x16xf32>, tensor<16x16x16xi32>)
  %1 = "stablehlo.slice"(%0#0) { start_indices = dense<[0, 0, 0]> : tensor<3xi64>, limit_indices = dense<[16, 16, 8]> : tensor<3xi64>, strides = dense<[1, 1, 1]> : tensor<3xi64> } : (tensor<16x16x16xf32>) -> tensor<16x16x8xf32> 
  %2 = "stablehlo.slice"(%0#1) { start_indices = dense<[0, 0, 0]> : tensor<3xi64>, limit_indices = dense<[16, 16, 8]> : tensor<3xi64>, strides = dense<[1, 1, 1]> : tensor<3xi64> } : (tensor<16x16x16xi32>) -> tensor<16x16x8xi32>
  return %1, %2 : tensor<16x16x8xf32>, tensor<16x16x8xi32>
}

// CHECK-LABEL: @broadcast_iota_sort_slice_is_topk
// CHECK-SAME: %[[IN:[a-z0-9]+]]
// CHECK: %[[VALUES:.+]], %[[INDICES:.+]] = chlo.top_k(%[[IN]], k = 8) : tensor<16x16x16xf32> -> (tensor<16x16x8xf32>, tensor<16x16x8xi32>)
// CHECK: return %[[VALUES]], %[[INDICES]] : tensor<16x16x8xf32>, tensor<16x16x8xi32>

// -----

func.func @broadcast_iota_sort_slice_incorrect_dims(%in : tensor<16x16x16xf32>) -> (tensor<16x16x8xf32>, tensor<16x16x8xi32>) {
  %iota = "stablehlo.iota"() { iota_dimension = 0 : i64 } : () -> tensor<16xi32>
  %broadcasted_0 = "stablehlo.broadcast_in_dim"(%iota) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<16xi32>) -> tensor<16x16xi32>
  %broadcasted_1 = "stablehlo.broadcast_in_dim"(%broadcasted_0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<16x16xi32>) -> tensor<16x16x16xi32>
  %0:2 = "stablehlo.sort"(%in, %broadcasted_1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 2 : i64, is_stable = true} : (tensor<16x16x16xf32>, tensor<16x16x16xi32>) -> (tensor<16x16x16xf32>, tensor<16x16x16xi32>)
  %1 = "stablehlo.slice"(%0#0) { start_indices = dense<[0, 0, 0]> : tensor<3xi64>, limit_indices = dense<[16, 16, 8]> : tensor<3xi64>, strides = dense<[1, 1, 1]> : tensor<3xi64> } : (tensor<16x16x16xf32>) -> tensor<16x16x8xf32> 
  %2 = "stablehlo.slice"(%0#1) { start_indices = dense<[0, 0, 0]> : tensor<3xi64>, limit_indices = dense<[16, 16, 8]> : tensor<3xi64>, strides = dense<[1, 1, 1]> : tensor<3xi64> } : (tensor<16x16x16xi32>) -> tensor<16x16x8xi32>
  return %1, %2 : tensor<16x16x8xf32>, tensor<16x16x8xi32>
}

// CHECK-LABEL: @broadcast_iota_sort_slice_incorrect_dims
// CHECK-NOT: chlo.top_k

// -----

// CHECK-LABEL: @concat_remove_zero_extents
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
func.func @concat_remove_zero_extents(%arg0: tensor<2x3xi32>, %arg1 : tensor<2x3xi32>, %arg2 : tensor<2x0xi32>) -> tensor<2x6xi32> {
  %0 = stablehlo.concatenate %arg0, %arg1, %arg2, dim = 1 : (tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x0xi32>) -> tensor<2x6xi32>
  // CHECK: [[R0:%.+]] = stablehlo.concatenate  %[[ARG0]], %[[ARG1]], dim = 1 : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x6xi32>
  return %0 : tensor<2x6xi32>
}

