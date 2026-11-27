// Keys [0, 8) and [16, 24) are hidden in every row and query row 2 sees no
// key. Safe softmax makes row 2 zero, as in the producer. The keys are zero, so
// all visible scores are equal and a row's output is the mean of the values it
// sees. Key 8 + i holds i in every lane, hidden keys hold 100.0, and rows 0, 1
// and 3 see keys [8, 16), [8, 12) and [8, 10): means 3.5, 1.5 and 0.5.
func.func @sdpa_hidden_row() {
  %q = util.unfoldable_constant dense<1.0> : tensor<1x1x4x6xf32>
  %k = util.unfoldable_constant dense<0.0> : tensor<1x1x24x6xf32>
  %hidden_v = util.unfoldable_constant dense<100.0> : tensor<1x1x8x5xf32>
  %per_key = util.unfoldable_constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]> : tensor<8xf32>
  %visible_v = stablehlo.broadcast_in_dim %per_key, dims = [2] : (tensor<8xf32>) -> tensor<1x1x8x5xf32>
  %v = stablehlo.concatenate %hidden_v, %visible_v, %hidden_v, dim = 2
      : (tensor<1x1x8x5xf32>, tensor<1x1x8x5xf32>, tensor<1x1x8x5xf32>) -> tensor<1x1x24x5xf32>
  %hidden_mask = util.unfoldable_constant dense<false> : tensor<4x8xi1>
  %middle_mask = util.unfoldable_constant dense<[
      [true, true, true, true, true, true, true, true],
      [true, true, true, true, false, false, false, false],
      [false, false, false, false, false, false, false, false],
      [true, true, false, false, false, false, false, false]]> : tensor<4x8xi1>
  %mask = stablehlo.concatenate %hidden_mask, %middle_mask, %hidden_mask, dim = 1
      : (tensor<4x8xi1>, tensor<4x8xi1>, tensor<4x8xi1>) -> tensor<4x24xi1>
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v, %mask {
    composite_attributes = {enable_gqa = false, is_causal = false},
    decomposition = @sdpa_hidden_row_impl, version = 1 : i32
  } : (tensor<1x1x4x6xf32>, tensor<1x1x24x6xf32>, tensor<1x1x24x5xf32>, tensor<4x24xi1>) -> tensor<1x1x4x5xf32>
  check.expect_almost_eq_const(%0, dense<[[[
      [3.5, 3.5, 3.5, 3.5, 3.5],
      [1.5, 1.5, 1.5, 1.5, 1.5],
      [0.0, 0.0, 0.0, 0.0, 0.0],
      [0.5, 0.5, 0.5, 0.5, 0.5]]]]> : tensor<1x1x4x5xf32>) : tensor<1x1x4x5xf32>
  return
}

// Unused: the composite is rewritten to attention before decompositions run.
func.func private @sdpa_hidden_row_impl(%q: tensor<1x1x4x6xf32>, %k: tensor<1x1x24x6xf32>, %v: tensor<1x1x24x5xf32>, %m: tensor<4x24xi1>) -> tensor<1x1x4x5xf32> {
  %0 = stablehlo.constant dense<0.0> : tensor<1x1x4x5xf32>
  return %0 : tensor<1x1x4x5xf32>
}

// Causal exclusion must follow score scaling, including zero/negative scales.
func.func @causal_dynamic_zero() {
  %q = flow.tensor.dynamic_constant dense<[[1.0], [2.0]]> : tensor<2x1xf32> -> tensor<?x1xf32>
  %k = util.unfoldable_constant dense<[[1.0], [3.0]]> : tensor<2x1xf32>
  %v = util.unfoldable_constant dense<[[2.0], [10.0]]> : tensor<2x1xf32>
  %result = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = false, is_causal = true, scale = 0.0 : f32},
    decomposition = @causal_dynamic_zero_impl, version = 1 : i32
  } : (tensor<?x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<?x1xf32>
  %barrier = util.optimization_barrier %result : tensor<?x1xf32>
  %checked = tensor.cast %barrier : tensor<?x1xf32> to tensor<2x1xf32>
  check.expect_almost_eq_const(%checked, dense<[[2.0], [6.0000000]]> : tensor<2x1xf32>) : tensor<2x1xf32>
  return
}

// This fallback computes the same two-row causal attention for finite inputs;
// the specialization is required by the compiler-level handler tests.
func.func private @causal_dynamic_zero_impl(%q: tensor<?x1xf32>, %k: tensor<2x1xf32>, %v: tensor<2x1xf32>) -> tensor<?x1xf32> {
  %scale = stablehlo.constant dense<0.0> : tensor<f32>
  %scores = stablehlo.dot_general %q, %k, contracting_dims = [1] x [1] : (tensor<?x1xf32>, tensor<2x1xf32>) -> tensor<?x2xf32>
  %factor = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
  %static_scores = tensor.cast %scores : tensor<?x2xf32> to tensor<2x2xf32>
  %scaled = stablehlo.multiply %static_scores, %factor : tensor<2x2xf32>
  %mask = stablehlo.constant dense<[[true, false], [true, true]]> : tensor<2x2xi1>
  %negative_inf = stablehlo.constant dense<0xFF800000> : tensor<2x2xf32>
  %masked = stablehlo.select %mask, %scaled, %negative_inf : tensor<2x2xi1>, tensor<2x2xf32>
  %exp = stablehlo.exponential %masked : tensor<2x2xf32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %sums = stablehlo.reduce(%exp init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
  %denominator = stablehlo.broadcast_in_dim %sums, dims = [0] : (tensor<2xf32>) -> tensor<2x2xf32>
  %probability = stablehlo.divide %exp, %denominator : tensor<2x2xf32>
  %output = stablehlo.dot_general %probability, %v, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %result = tensor.cast %output : tensor<2x1xf32> to tensor<?x1xf32>
  return %result : tensor<?x1xf32>
}

// Causal exclusion must follow score scaling, including zero/negative scales.
func.func @causal_dynamic_negative() {
  %q = flow.tensor.dynamic_constant dense<[[1.0], [2.0]]> : tensor<2x1xf32> -> tensor<?x1xf32>
  %k = util.unfoldable_constant dense<[[1.0], [3.0]]> : tensor<2x1xf32>
  %v = util.unfoldable_constant dense<[[2.0], [10.0]]> : tensor<2x1xf32>
  %result = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = false, is_causal = true, scale = -1.0 : f32},
    decomposition = @causal_dynamic_negative_impl, version = 1 : i32
  } : (tensor<?x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<?x1xf32>
  %barrier = util.optimization_barrier %result : tensor<?x1xf32>
  %checked = tensor.cast %barrier : tensor<?x1xf32> to tensor<2x1xf32>
  check.expect_almost_eq_const(%checked, dense<[[2.0], [2.1438897]]> : tensor<2x1xf32>) : tensor<2x1xf32>
  return
}

// This fallback computes the same two-row causal attention for finite inputs;
// the specialization is required by the compiler-level handler tests.
func.func private @causal_dynamic_negative_impl(%q: tensor<?x1xf32>, %k: tensor<2x1xf32>, %v: tensor<2x1xf32>) -> tensor<?x1xf32> {
  %scale = stablehlo.constant dense<-1.0> : tensor<f32>
  %scores = stablehlo.dot_general %q, %k, contracting_dims = [1] x [1] : (tensor<?x1xf32>, tensor<2x1xf32>) -> tensor<?x2xf32>
  %factor = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
  %static_scores = tensor.cast %scores : tensor<?x2xf32> to tensor<2x2xf32>
  %scaled = stablehlo.multiply %static_scores, %factor : tensor<2x2xf32>
  %mask = stablehlo.constant dense<[[true, false], [true, true]]> : tensor<2x2xi1>
  %negative_inf = stablehlo.constant dense<0xFF800000> : tensor<2x2xf32>
  %masked = stablehlo.select %mask, %scaled, %negative_inf : tensor<2x2xi1>, tensor<2x2xf32>
  %exp = stablehlo.exponential %masked : tensor<2x2xf32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %sums = stablehlo.reduce(%exp init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
  %denominator = stablehlo.broadcast_in_dim %sums, dims = [0] : (tensor<2xf32>) -> tensor<2x2xf32>
  %probability = stablehlo.divide %exp, %denominator : tensor<2x2xf32>
  %output = stablehlo.dot_general %probability, %v, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %result = tensor.cast %output : tensor<2x1xf32> to tensor<?x1xf32>
  return %result : tensor<?x1xf32>
}

// Causal exclusion must follow score scaling, including zero/negative scales.
func.func @causal_dynamic_positive() {
  %q = flow.tensor.dynamic_constant dense<[[1.0], [2.0]]> : tensor<2x1xf32> -> tensor<?x1xf32>
  %k = util.unfoldable_constant dense<[[1.0], [3.0]]> : tensor<2x1xf32>
  %v = util.unfoldable_constant dense<[[2.0], [10.0]]> : tensor<2x1xf32>
  %result = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = false, is_causal = true, scale = 1.0 : f32},
    decomposition = @causal_dynamic_positive_impl, version = 1 : i32
  } : (tensor<?x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<?x1xf32>
  %barrier = util.optimization_barrier %result : tensor<?x1xf32>
  %checked = tensor.cast %barrier : tensor<?x1xf32> to tensor<2x1xf32>
  check.expect_almost_eq_const(%checked, dense<[[2.0], [9.8561103]]> : tensor<2x1xf32>) : tensor<2x1xf32>
  return
}

// This fallback computes the same two-row causal attention for finite inputs;
// the specialization is required by the compiler-level handler tests.
func.func private @causal_dynamic_positive_impl(%q: tensor<?x1xf32>, %k: tensor<2x1xf32>, %v: tensor<2x1xf32>) -> tensor<?x1xf32> {
  %scale = stablehlo.constant dense<1.0> : tensor<f32>
  %scores = stablehlo.dot_general %q, %k, contracting_dims = [1] x [1] : (tensor<?x1xf32>, tensor<2x1xf32>) -> tensor<?x2xf32>
  %factor = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
  %static_scores = tensor.cast %scores : tensor<?x2xf32> to tensor<2x2xf32>
  %scaled = stablehlo.multiply %static_scores, %factor : tensor<2x2xf32>
  %mask = stablehlo.constant dense<[[true, false], [true, true]]> : tensor<2x2xi1>
  %negative_inf = stablehlo.constant dense<0xFF800000> : tensor<2x2xf32>
  %masked = stablehlo.select %mask, %scaled, %negative_inf : tensor<2x2xi1>, tensor<2x2xf32>
  %exp = stablehlo.exponential %masked : tensor<2x2xf32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %sums = stablehlo.reduce(%exp init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
  %denominator = stablehlo.broadcast_in_dim %sums, dims = [0] : (tensor<2xf32>) -> tensor<2x2xf32>
  %probability = stablehlo.divide %exp, %denominator : tensor<2x2xf32>
  %output = stablehlo.dot_general %probability, %v, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %result = tensor.cast %output : tensor<2x1xf32> to tensor<?x1xf32>
  return %result : tensor<?x1xf32>
}

// Causal exclusion must follow score scaling, including zero/negative scales.
func.func @causal_static_negative() {
  %q = util.unfoldable_constant dense<[[1.0], [2.0]]> : tensor<2x1xf32>
  %k = util.unfoldable_constant dense<[[1.0], [3.0]]> : tensor<2x1xf32>
  %v = util.unfoldable_constant dense<[[2.0], [10.0]]> : tensor<2x1xf32>
  %result = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = false, is_causal = true, scale = -1.0 : f32},
    decomposition = @causal_static_negative_impl, version = 1 : i32
  } : (tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  check.expect_almost_eq_const(%result, dense<[[2.0], [2.1438897]]> : tensor<2x1xf32>) : tensor<2x1xf32>
  return
}

// This fallback computes the same two-row causal attention for finite inputs;
// the specialization is required by the compiler-level handler tests.
func.func private @causal_static_negative_impl(%q: tensor<2x1xf32>, %k: tensor<2x1xf32>, %v: tensor<2x1xf32>) -> tensor<2x1xf32> {
  %scale = stablehlo.constant dense<-1.0> : tensor<f32>
  %scores = stablehlo.dot_general %q, %k, contracting_dims = [1] x [1] : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
  %factor = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
  %static_scores = tensor.cast %scores : tensor<2x2xf32> to tensor<2x2xf32>
  %scaled = stablehlo.multiply %static_scores, %factor : tensor<2x2xf32>
  %mask = stablehlo.constant dense<[[true, false], [true, true]]> : tensor<2x2xi1>
  %negative_inf = stablehlo.constant dense<0xFF800000> : tensor<2x2xf32>
  %masked = stablehlo.select %mask, %scaled, %negative_inf : tensor<2x2xi1>, tensor<2x2xf32>
  %exp = stablehlo.exponential %masked : tensor<2x2xf32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %sums = stablehlo.reduce(%exp init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
  %denominator = stablehlo.broadcast_in_dim %sums, dims = [0] : (tensor<2xf32>) -> tensor<2x2xf32>
  %probability = stablehlo.divide %exp, %denominator : tensor<2x2xf32>
  %output = stablehlo.dot_general %probability, %v, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %result = tensor.cast %output : tensor<2x1xf32> to tensor<2x1xf32>
  return %result : tensor<2x1xf32>
}

func.func @unmasked_negative_infinity_f32() {
  %query = util.unfoldable_constant dense<0xFF800000> : tensor<1x1xf32>
  %key = util.unfoldable_constant dense<1.0> : tensor<1x1xf32>
  %value = util.unfoldable_constant dense<1.0> : tensor<1x1xf32>
  %result = stablehlo.composite "torch.scaled_dot_product_attention" %query, %key, %value {
    composite_attributes = {enable_gqa = false, is_causal = false, scale = 1.0 : f32},
    decomposition = @scalar_attention_plain_f32, version = 1 : i32
  } : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
  check.expect_eq_const(%result, dense<0.0> : tensor<1x1xf32>) : tensor<1x1xf32>
  return
}

// A one-key decomposition retains the producer's safe softmax and P @ V.
func.func private @scalar_attention_plain_f32(%query: tensor<1x1xf32>, %key: tensor<1x1xf32>, %value: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %zero = stablehlo.constant dense<0.0> : tensor<1x1xf32>
  %negative_infinity = stablehlo.constant dense<0xFF800000> : tensor<1x1xf32>
  %score = stablehlo.multiply %query, %key : tensor<1x1xf32>
  %empty = "stablehlo.compare"(%score, %negative_infinity) {comparison_direction = #stablehlo<comparison_direction EQ>, compare_type = #stablehlo<comparison_type FLOAT>} : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xi1>
  %shift = stablehlo.subtract %score, %score : tensor<1x1xf32>
  %exp = stablehlo.exponential %shift : tensor<1x1xf32>
  %probability = stablehlo.select %empty, %zero, %exp : tensor<1x1xi1>, tensor<1x1xf32>
  %result = stablehlo.multiply %probability, %value : tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}

func.func @unmasked_negative_infinity_f16() {
  %query = util.unfoldable_constant dense<0xFC00> : tensor<1x1xf16>
  %key = util.unfoldable_constant dense<1.0> : tensor<1x1xf16>
  %value = util.unfoldable_constant dense<1.0> : tensor<1x1xf16>
  %result = stablehlo.composite "torch.scaled_dot_product_attention" %query, %key, %value {
    composite_attributes = {enable_gqa = false, is_causal = false, scale = 1.0 : f32},
    decomposition = @scalar_attention_plain_f16, version = 1 : i32
  } : (tensor<1x1xf16>, tensor<1x1xf16>, tensor<1x1xf16>) -> tensor<1x1xf16>
  check.expect_eq_const(%result, dense<0.0> : tensor<1x1xf16>) : tensor<1x1xf16>
  return
}

// A one-key decomposition retains the producer's safe softmax and P @ V.
func.func private @scalar_attention_plain_f16(%query: tensor<1x1xf16>, %key: tensor<1x1xf16>, %value: tensor<1x1xf16>) -> tensor<1x1xf16> {
  %zero = stablehlo.constant dense<0.0> : tensor<1x1xf16>
  %negative_infinity = stablehlo.constant dense<0xFC00> : tensor<1x1xf16>
  %score = stablehlo.multiply %query, %key : tensor<1x1xf16>
  %empty = "stablehlo.compare"(%score, %negative_infinity) {comparison_direction = #stablehlo<comparison_direction EQ>, compare_type = #stablehlo<comparison_type FLOAT>} : (tensor<1x1xf16>, tensor<1x1xf16>) -> tensor<1x1xi1>
  %shift = stablehlo.subtract %score, %score : tensor<1x1xf16>
  %exp = stablehlo.exponential %shift : tensor<1x1xf16>
  %probability = stablehlo.select %empty, %zero, %exp : tensor<1x1xi1>, tensor<1x1xf16>
  %result = stablehlo.multiply %probability, %value : tensor<1x1xf16>
  return %result : tensor<1x1xf16>
}

func.func @unmasked_negative_infinity_bf16() {
  %query = util.unfoldable_constant dense<0xFF80> : tensor<1x1xbf16>
  %key = util.unfoldable_constant dense<1.0> : tensor<1x1xbf16>
  %value = util.unfoldable_constant dense<1.0> : tensor<1x1xbf16>
  %result = stablehlo.composite "torch.scaled_dot_product_attention" %query, %key, %value {
    composite_attributes = {enable_gqa = false, is_causal = false, scale = 1.0 : f32},
    decomposition = @scalar_attention_plain_bf16, version = 1 : i32
  } : (tensor<1x1xbf16>, tensor<1x1xbf16>, tensor<1x1xbf16>) -> tensor<1x1xbf16>
  check.expect_eq_const(%result, dense<0.0> : tensor<1x1xbf16>) : tensor<1x1xbf16>
  return
}

// A one-key decomposition retains the producer's safe softmax and P @ V.
func.func private @scalar_attention_plain_bf16(%query: tensor<1x1xbf16>, %key: tensor<1x1xbf16>, %value: tensor<1x1xbf16>) -> tensor<1x1xbf16> {
  %zero = stablehlo.constant dense<0.0> : tensor<1x1xbf16>
  %negative_infinity = stablehlo.constant dense<0xFF80> : tensor<1x1xbf16>
  %score = stablehlo.multiply %query, %key : tensor<1x1xbf16>
  %empty = "stablehlo.compare"(%score, %negative_infinity) {comparison_direction = #stablehlo<comparison_direction EQ>, compare_type = #stablehlo<comparison_type FLOAT>} : (tensor<1x1xbf16>, tensor<1x1xbf16>) -> tensor<1x1xi1>
  %shift = stablehlo.subtract %score, %score : tensor<1x1xbf16>
  %exp = stablehlo.exponential %shift : tensor<1x1xbf16>
  %probability = stablehlo.select %empty, %zero, %exp : tensor<1x1xi1>, tensor<1x1xbf16>
  %result = stablehlo.multiply %probability, %value : tensor<1x1xbf16>
  return %result : tensor<1x1xbf16>
}

func.func @masked_nan_score_f32() {
  %query = util.unfoldable_constant dense<0x7FC00000> : tensor<1x1xf32>
  %key = util.unfoldable_constant dense<1.0> : tensor<1x1xf32>
  %value = util.unfoldable_constant dense<1.0> : tensor<1x1xf32>
  %mask = util.unfoldable_constant dense<false> : tensor<1x1xi1>
  %result = stablehlo.composite "torch.scaled_dot_product_attention" %query, %key, %value, %mask {
    composite_attributes = {enable_gqa = false, is_causal = false, scale = 1.0 : f32},
    decomposition = @scalar_attention_masked_f32, version = 1 : i32
  } : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xi1>) -> tensor<1x1xf32>
  %nan = "stablehlo.compare"(%result, %result) {comparison_direction = #stablehlo<comparison_direction NE>, compare_type = #stablehlo<comparison_type FLOAT>} : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xi1>
  check.expect_eq_const(%nan, dense<true> : tensor<1x1xi1>) : tensor<1x1xi1>
  return
}

// A one-key decomposition retains the producer's safe softmax and P @ V.
func.func private @scalar_attention_masked_f32(%query: tensor<1x1xf32>, %key: tensor<1x1xf32>, %value: tensor<1x1xf32>, %mask: tensor<1x1xi1>) -> tensor<1x1xf32> {
  %zero = stablehlo.constant dense<0.0> : tensor<1x1xf32>
  %negative_infinity = stablehlo.constant dense<0xFF800000> : tensor<1x1xf32>
  %score = stablehlo.multiply %query, %key : tensor<1x1xf32>
  %bias = stablehlo.select %mask, %zero, %negative_infinity : tensor<1x1xi1>, tensor<1x1xf32>
  %masked = stablehlo.add %score, %bias : tensor<1x1xf32>
  %empty = "stablehlo.compare"(%masked, %negative_infinity) {comparison_direction = #stablehlo<comparison_direction EQ>, compare_type = #stablehlo<comparison_type FLOAT>} : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xi1>
  %shift = stablehlo.subtract %masked, %masked : tensor<1x1xf32>
  %exp = stablehlo.exponential %shift : tensor<1x1xf32>
  %probability = stablehlo.select %empty, %zero, %exp : tensor<1x1xi1>, tensor<1x1xf32>
  %result = stablehlo.multiply %probability, %value : tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}

func.func @masked_positive_infinity_f32() {
  %query = util.unfoldable_constant dense<0x7F800000> : tensor<1x1xf32>
  %key = util.unfoldable_constant dense<1.0> : tensor<1x1xf32>
  %value = util.unfoldable_constant dense<1.0> : tensor<1x1xf32>
  %mask = util.unfoldable_constant dense<false> : tensor<1x1xi1>
  %result = stablehlo.composite "torch.scaled_dot_product_attention" %query, %key, %value, %mask {
    composite_attributes = {enable_gqa = false, is_causal = false, scale = 1.0 : f32},
    decomposition = @scalar_attention_masked_f32, version = 1 : i32
  } : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xi1>) -> tensor<1x1xf32>
  %nan = "stablehlo.compare"(%result, %result) {comparison_direction = #stablehlo<comparison_direction NE>, compare_type = #stablehlo<comparison_type FLOAT>} : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xi1>
  check.expect_eq_const(%nan, dense<true> : tensor<1x1xi1>) : tensor<1x1xi1>
  return
}

func.func @masked_nan_value_f32() {
  %query = util.unfoldable_constant dense<1.0> : tensor<1x1xf32>
  %key = util.unfoldable_constant dense<1.0> : tensor<1x1xf32>
  %value = util.unfoldable_constant dense<0x7FC00000> : tensor<1x1xf32>
  %mask = util.unfoldable_constant dense<false> : tensor<1x1xi1>
  %result = stablehlo.composite "torch.scaled_dot_product_attention" %query, %key, %value, %mask {
    composite_attributes = {enable_gqa = false, is_causal = false, scale = 1.0 : f32},
    decomposition = @scalar_attention_masked_f32, version = 1 : i32
  } : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xi1>) -> tensor<1x1xf32>
  %nan = "stablehlo.compare"(%result, %result) {comparison_direction = #stablehlo<comparison_direction NE>, compare_type = #stablehlo<comparison_type FLOAT>} : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xi1>
  check.expect_eq_const(%nan, dense<true> : tensor<1x1xi1>) : tensor<1x1xi1>
  return
}

func.func @masked_nan_score_f16() {
  %query = util.unfoldable_constant dense<0x7E00> : tensor<1x1xf16>
  %key = util.unfoldable_constant dense<1.0> : tensor<1x1xf16>
  %value = util.unfoldable_constant dense<1.0> : tensor<1x1xf16>
  %mask = util.unfoldable_constant dense<false> : tensor<1x1xi1>
  %result = stablehlo.composite "torch.scaled_dot_product_attention" %query, %key, %value, %mask {
    composite_attributes = {enable_gqa = false, is_causal = false, scale = 1.0 : f32},
    decomposition = @scalar_attention_masked_f16, version = 1 : i32
  } : (tensor<1x1xf16>, tensor<1x1xf16>, tensor<1x1xf16>, tensor<1x1xi1>) -> tensor<1x1xf16>
  %nan = "stablehlo.compare"(%result, %result) {comparison_direction = #stablehlo<comparison_direction NE>, compare_type = #stablehlo<comparison_type FLOAT>} : (tensor<1x1xf16>, tensor<1x1xf16>) -> tensor<1x1xi1>
  check.expect_eq_const(%nan, dense<true> : tensor<1x1xi1>) : tensor<1x1xi1>
  return
}

// A one-key decomposition retains the producer's safe softmax and P @ V.
func.func private @scalar_attention_masked_f16(%query: tensor<1x1xf16>, %key: tensor<1x1xf16>, %value: tensor<1x1xf16>, %mask: tensor<1x1xi1>) -> tensor<1x1xf16> {
  %zero = stablehlo.constant dense<0.0> : tensor<1x1xf16>
  %negative_infinity = stablehlo.constant dense<0xFC00> : tensor<1x1xf16>
  %score = stablehlo.multiply %query, %key : tensor<1x1xf16>
  %bias = stablehlo.select %mask, %zero, %negative_infinity : tensor<1x1xi1>, tensor<1x1xf16>
  %masked = stablehlo.add %score, %bias : tensor<1x1xf16>
  %empty = "stablehlo.compare"(%masked, %negative_infinity) {comparison_direction = #stablehlo<comparison_direction EQ>, compare_type = #stablehlo<comparison_type FLOAT>} : (tensor<1x1xf16>, tensor<1x1xf16>) -> tensor<1x1xi1>
  %shift = stablehlo.subtract %masked, %masked : tensor<1x1xf16>
  %exp = stablehlo.exponential %shift : tensor<1x1xf16>
  %probability = stablehlo.select %empty, %zero, %exp : tensor<1x1xi1>, tensor<1x1xf16>
  %result = stablehlo.multiply %probability, %value : tensor<1x1xf16>
  return %result : tensor<1x1xf16>
}

// A partial key tile must not let its zero-padded keys turn -inf scores into
// NaN. Every real key is visible and the producer's safe softmax returns zero.
func.func @negative_infinity_partial_key_tile_bool() {
  %q = util.unfoldable_constant dense<0xFF800000> : tensor<1x2x16xf32>
  %k = util.unfoldable_constant dense<1.0> : tensor<1x3x16xf32>
  %v = util.unfoldable_constant dense<1.0> : tensor<1x3x8xf32>
  %mask = util.unfoldable_constant dense<true> : tensor<2x3xi1>
  %result = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v, %mask {
    composite_attributes = {enable_gqa = false, is_causal = false, scale = 1.0 : f32},
    decomposition = @partial_key_tile_bool_impl, version = 1 : i32
  } : (tensor<1x2x16xf32>, tensor<1x3x16xf32>, tensor<1x3x8xf32>, tensor<2x3xi1>) -> tensor<1x2x8xf32>
  check.expect_eq_const(%result, dense<0.0> : tensor<1x2x8xf32>) : tensor<1x2x8xf32>
  return
}

// A partial key tile must not let its zero-padded keys turn -inf scores into
// NaN. Every real key is visible and the producer's safe softmax returns zero.
func.func @negative_infinity_partial_key_tile_float() {
  %q = util.unfoldable_constant dense<0xFF800000> : tensor<1x2x16xf32>
  %k = util.unfoldable_constant dense<1.0> : tensor<1x3x16xf32>
  %v = util.unfoldable_constant dense<1.0> : tensor<1x3x8xf32>
  %mask = util.unfoldable_constant dense<0.0> : tensor<2x3xf32>
  %result = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v, %mask {
    composite_attributes = {enable_gqa = false, is_causal = false, scale = 1.0 : f32},
    decomposition = @partial_key_tile_float_impl, version = 1 : i32
  } : (tensor<1x2x16xf32>, tensor<1x3x16xf32>, tensor<1x3x8xf32>, tensor<2x3xf32>) -> tensor<1x2x8xf32>
  check.expect_eq_const(%result, dense<0.0> : tensor<1x2x8xf32>) : tensor<1x2x8xf32>
  return
}

func.func private @partial_key_tile_bool_impl(%q: tensor<1x2x16xf32>, %k: tensor<1x3x16xf32>, %v: tensor<1x3x8xf32>, %mask: tensor<2x3xi1>) -> tensor<1x2x8xf32> {
  %zero = stablehlo.constant dense<0.0> : tensor<2x3xf32>
  %negative = stablehlo.constant dense<0xFF800000> : tensor<2x3xf32>
  %bias = stablehlo.select %mask, %zero, %negative : tensor<2x3xi1>, tensor<2x3xf32>
  %result = func.call @partial_key_tile_float_impl(%q, %k, %v, %bias) : (tensor<1x2x16xf32>, tensor<1x3x16xf32>, tensor<1x3x8xf32>, tensor<2x3xf32>) -> tensor<1x2x8xf32>
  return %result : tensor<1x2x8xf32>
}

func.func private @partial_key_tile_float_impl(%q: tensor<1x2x16xf32>, %k: tensor<1x3x16xf32>, %v: tensor<1x3x8xf32>, %mask: tensor<2x3xf32>) -> tensor<1x2x8xf32> {
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %negative = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %true = stablehlo.constant dense<true> : tensor<i1>
  %scores = stablehlo.dot_general %q, %k, batching_dims = [0] x [0], contracting_dims = [2] x [2] : (tensor<1x2x16xf32>, tensor<1x3x16xf32>) -> tensor<1x2x3xf32>
  %bias = stablehlo.broadcast_in_dim %mask, dims = [1, 2] : (tensor<2x3xf32>) -> tensor<1x2x3xf32>
  %masked = stablehlo.add %scores, %bias : tensor<1x2x3xf32>
  %max = stablehlo.reduce(%masked init: %negative) applies stablehlo.maximum across dimensions = [2] : (tensor<1x2x3xf32>, tensor<f32>) -> tensor<1x2xf32>
  %max_broadcast = stablehlo.broadcast_in_dim %max, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2x3xf32>
  %shift = stablehlo.subtract %masked, %max_broadcast : tensor<1x2x3xf32>
  %exp = stablehlo.exponential %shift : tensor<1x2x3xf32>
  %sum = stablehlo.reduce(%exp init: %zero) applies stablehlo.add across dimensions = [2] : (tensor<1x2x3xf32>, tensor<f32>) -> tensor<1x2xf32>
  %sum_broadcast = stablehlo.broadcast_in_dim %sum, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2x3xf32>
  %probability = stablehlo.divide %exp, %sum_broadcast : tensor<1x2x3xf32>
  %negative_broadcast = stablehlo.broadcast_in_dim %negative, dims = [] : (tensor<f32>) -> tensor<1x2x3xf32>
  %is_negative = "stablehlo.compare"(%masked, %negative_broadcast) {comparison_direction = #stablehlo<comparison_direction EQ>, compare_type = #stablehlo<comparison_type FLOAT>} : (tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<1x2x3xi1>
  %empty = stablehlo.reduce(%is_negative init: %true) applies stablehlo.and across dimensions = [2] : (tensor<1x2x3xi1>, tensor<i1>) -> tensor<1x2xi1>
  %empty_broadcast = stablehlo.broadcast_in_dim %empty, dims = [0, 1] : (tensor<1x2xi1>) -> tensor<1x2x3xi1>
  %zero_broadcast = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<1x2x3xf32>
  %safe = stablehlo.select %empty_broadcast, %zero_broadcast, %probability : tensor<1x2x3xi1>, tensor<1x2x3xf32>
  %result = stablehlo.dot_general %safe, %v, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x2x3xf32>, tensor<1x3x8xf32>) -> tensor<1x2x8xf32>
  return %result : tensor<1x2x8xf32>
}
