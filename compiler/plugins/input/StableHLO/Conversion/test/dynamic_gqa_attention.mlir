// RUN: iree-compile --iree-input-type=stablehlo --compile-to=flow \
// RUN:   --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-cpu=generic %s \
// RUN:   | FileCheck %s --implicit-check-not=elementwise_broadcast

// A decode step of grouped-query attention over a cache of s rows grown by one
// row, as a Torch export with a symbolic s writes it: each KV head repeats for
// its 3 query heads. The concatenations are copies and the repeats fold into
// the attention, which reads 4 KV heads.
// CHECK: flow.executable.export public @{{.+}}_attention_8x3x6x16xD
func.func public @main(%q: tensor<2x12x1x16xf32>, %k_cache: tensor<2x4x?x16xf32>, %v_cache: tensor<2x4x?x6xf32>, %k_row: tensor<2x4x1x16xf32>, %v_row: tensor<2x4x1x6xf32>) -> tensor<2x12x1x6xf32> {
  %one = stablehlo.constant dense<1> : tensor<i64>
  %zero = stablehlo.constant dense<0> : tensor<i64>
  %limit = stablehlo.constant dense<1024> : tensor<i64>
  %k_rows = stablehlo.get_dimension_size %k_cache, dim = 2 : (tensor<2x4x?x16xf32>) -> tensor<i32>
  %s = stablehlo.convert %k_rows : (tensor<i32>) -> tensor<i64>
  %v_rows = stablehlo.get_dimension_size %v_cache, dim = 2 : (tensor<2x4x?x6xf32>) -> tensor<i32>
  %v_s = stablehlo.convert %v_rows : (tensor<i32>) -> tensor<i64>
  %positive = stablehlo.compare GE, %s, %one, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%positive, %s) {api_version = 2 : i32, error_message = "s >= 1", has_side_effect = true} : (tensor<i1>, tensor<i64>) -> ()
  %same = stablehlo.compare EQ, %v_s, %s, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%same, %v_s, %s) {api_version = 2 : i32, error_message = "v rows {0} != s {1}", has_side_effect = true} : (tensor<i1>, tensor<i64>, tensor<i64>) -> ()
  %room = stablehlo.subtract %limit, %s : tensor<i64>
  %bounded = stablehlo.compare GE, %room, %zero, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%bounded, %room, %s) {api_version = 2 : i32, error_message = "s <= 1024", has_side_effect = true} : (tensor<i1>, tensor<i64>, tensor<i64>) -> ()
  %r = call @step(%s, %q, %k_cache, %v_cache, %k_row, %v_row) : (tensor<i64>, tensor<2x12x1x16xf32>, tensor<2x4x?x16xf32>, tensor<2x4x?x6xf32>, tensor<2x4x1x16xf32>, tensor<2x4x1x6xf32>) -> tensor<2x12x1x6xf32>
  return %r : tensor<2x12x1x6xf32>
}

func.func private @step(%s: tensor<i64> {jax.global_constant = "s"}, %q: tensor<2x12x1x16xf32>, %k_cache: tensor<2x4x?x16xf32>, %v_cache: tensor<2x4x?x6xf32>, %k_row: tensor<2x4x1x16xf32>, %v_row: tensor<2x4x1x6xf32>) -> tensor<2x12x1x6xf32> {
  %c2 = stablehlo.constant dense<2> : tensor<1xi32>
  %c3 = stablehlo.constant dense<3> : tensor<1xi32>
  %c4 = stablehlo.constant dense<4> : tensor<1xi32>
  %c6 = stablehlo.constant dense<6> : tensor<1xi32>
  %c12 = stablehlo.constant dense<12> : tensor<1xi32>
  %c16 = stablehlo.constant dense<16> : tensor<1xi32>
  %one = stablehlo.constant dense<1> : tensor<i64>
  %k = stablehlo.concatenate %k_cache, %k_row, dim = 2 : (tensor<2x4x?x16xf32>, tensor<2x4x1x16xf32>) -> tensor<2x4x?x16xf32>
  %v = stablehlo.concatenate %v_cache, %v_row, dim = 2 : (tensor<2x4x?x6xf32>, tensor<2x4x1x6xf32>) -> tensor<2x4x?x6xf32>
  %grown = stablehlo.add %s, %one : tensor<i64>
  %grown32 = stablehlo.convert %grown : (tensor<i64>) -> tensor<i32>
  %rows = stablehlo.reshape %grown32 : (tensor<i32>) -> tensor<1xi32>
  %k_shape = stablehlo.concatenate %c2, %c4, %c3, %rows, %c16, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<5xi32>
  %k_wide = stablehlo.dynamic_broadcast_in_dim %k, %k_shape, dims = [0, 1, 3, 4] : (tensor<2x4x?x16xf32>, tensor<5xi32>) -> tensor<2x4x3x?x16xf32>
  %k_heads = stablehlo.concatenate %c2, %c12, %rows, %c16, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %k_all = stablehlo.dynamic_reshape %k_wide, %k_heads : (tensor<2x4x3x?x16xf32>, tensor<4xi32>) -> tensor<2x12x?x16xf32>
  %v_shape = stablehlo.concatenate %c2, %c4, %c3, %rows, %c6, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<5xi32>
  %v_wide = stablehlo.dynamic_broadcast_in_dim %v, %v_shape, dims = [0, 1, 3, 4] : (tensor<2x4x?x6xf32>, tensor<5xi32>) -> tensor<2x4x3x?x6xf32>
  %v_heads = stablehlo.concatenate %c2, %c12, %rows, %c6, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %v_all = stablehlo.dynamic_reshape %v_wide, %v_heads : (tensor<2x4x3x?x6xf32>, tensor<4xi32>) -> tensor<2x12x?x6xf32>
  %o = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k_all, %v_all {
    composite_attributes = {enable_gqa = false, is_causal = false},
    decomposition = @sdpa_impl, version = 1 : i32
  } : (tensor<2x12x1x16xf32>, tensor<2x12x?x16xf32>, tensor<2x12x?x6xf32>) -> tensor<2x12x1x6xf32>
  return %o : tensor<2x12x1x6xf32>
}

// Placeholder plumbing; the composite lowers to attention.
func.func private @sdpa_impl(%q: tensor<2x12x1x16xf32>, %k: tensor<2x12x?x16xf32>, %v: tensor<2x12x?x6xf32>) -> tensor<2x12x1x6xf32> {
  %s = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<2x12x1x16xf32>, tensor<2x12x?x16xf32>) -> tensor<2x12x1x?xf32>
  %o = stablehlo.dot_general %s, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<2x12x1x?xf32>, tensor<2x12x?x6xf32>) -> tensor<2x12x1x6xf32>
  return %o : tensor<2x12x1x6xf32>
}
