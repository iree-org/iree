// RUN: iree-opt -split-input-file -verify-diagnostics %s | IreeFileCheck %s

// CHECK-LABEL: @float_passthrough
func @float_passthrough() -> (tensor<4xf32>) {
  %w1 = constant dense<0.5> : tensor<4xf32>
  %w2 = constant dense<1.2> : tensor<4xf32>
  %b1 = constant dense<[-1.0, 1.1, 2.2, -3.0]> : tensor<4xf32>
  %0 = "iree_compress.quant_region"(%w1, %w2, %b1) ({
    ^bb0(%aw1: tensor<4xf32>, %aw2 : tensor<4xf32>, %ab1 : tensor<4xf32>):
      %10 = "xla_hlo.mul"(%aw1, %aw2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %11 = "xla_hlo.add"(%10, %ab1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "iree_compress.return"(%11) : (tensor<4xf32>) -> ()
  }) {config = {logical_kernel = "GENERIC"}} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @quantized_widen
func @quantized_widen() -> (tensor<4xf32>) {
  %w1 = constant dense<0.5> : tensor<4xf32>
  %w2 = constant dense<1.2> : tensor<4xf32>
  %b1 = constant dense<[-1.0, 1.1, 2.2, -3.0]> : tensor<4xf32>
  %qw1 = "quant.qcast"(%w1) : (tensor<4xf32>) -> tensor<4x!quant.any<i8:f32>>
  %0 = "iree_compress.quant_region"(%qw1, %w2, %b1) ({
    ^bb0(%aw1: tensor<4xf32>, %aw2 : tensor<4xf32>, %ab1 : tensor<4xf32>):
      %10 = "xla_hlo.mul"(%aw1, %aw2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %11 = "xla_hlo.add"(%10, %ab1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "iree_compress.return"(%11) : (tensor<4xf32>) -> ()
  }) {config = {logical_kernel = "GENERIC"}} : (tensor<4x!quant.any<i8:f32>>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @quantized_narrow
func @quantized_narrow() -> (tensor<4x!quant.any<i8:f32>>) {
  %w1 = constant dense<0.5> : tensor<4xf32>
  %w2 = constant dense<1.2> : tensor<4xf32>
  %b1 = constant dense<[-1.0, 1.1, 2.2, -3.0]> : tensor<4xf32>
  %qw1 = "quant.qcast"(%w1) : (tensor<4xf32>) -> tensor<4x!quant.any<i8:f32>>
  %0 = "iree_compress.quant_region"(%qw1, %w2, %b1) ({
    ^bb0(%aw1: tensor<4xf32>, %aw2 : tensor<4xf32>, %ab1 : tensor<4xf32>):
      %10 = "xla_hlo.mul"(%aw1, %aw2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %11 = "xla_hlo.add"(%10, %ab1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "iree_compress.return"(%11) : (tensor<4xf32>) -> ()
  }) {config = {logical_kernel = "GENERIC"}} : (tensor<4x!quant.any<i8:f32>>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4x!quant.any<i8:f32>>)
  return %0 : tensor<4x!quant.any<i8:f32>>
}

// -----
func @mismatch_input_expressed_type() -> (tensor<4x!quant.any<i8:f32>>) {
  %w1 = constant dense<0.5> : tensor<4xf16>
  %w2 = constant dense<1.2> : tensor<4xf32>
  %b1 = constant dense<[-1.0, 1.1, 2.2, -3.0]> : tensor<4xf32>
  %qw1 = "quant.qcast"(%w1) : (tensor<4xf16>) -> tensor<4x!quant.any<i8:f16>>
  // @expected-error @+1 {{'iree_compress.quant_region' op incompatible operand to block argument type conversion: 'tensor<4x!quant.any<i8:f16>>' does not implicitly widen to 'tensor<4xf32>'}}
  %0 = "iree_compress.quant_region"(%qw1, %w2, %b1) ({
    ^bb0(%aw1: tensor<4xf32>, %aw2 : tensor<4xf32>, %ab1 : tensor<4xf32>):
      %10 = "xla_hlo.mul"(%aw1, %aw2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %11 = "xla_hlo.add"(%10, %ab1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "iree_compress.return"(%11) : (tensor<4xf32>) -> ()
  }) {config = {logical_kernel = "GENERIC"}} : (tensor<4x!quant.any<i8:f16>>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4x!quant.any<i8:f32>>)
  return %0 : tensor<4x!quant.any<i8:f32>>
}

// -----
func @mismatch_result_expressed_type() -> (tensor<4x!quant.any<i8:f16>>) {
  %w1 = constant dense<0.5> : tensor<4xf32>
  %w2 = constant dense<1.2> : tensor<4xf32>
  %b1 = constant dense<[-1.0, 1.1, 2.2, -3.0]> : tensor<4xf32>
  %qw1 = "quant.qcast"(%w1) : (tensor<4xf32>) -> tensor<4x!quant.any<i8:f32>>
  // @expected-error @+1 {{'iree_compress.quant_region' op incompatible block to result type conversion: 'tensor<4xf32>' does not implicitly narrow to 'tensor<4x!quant.any<i8:f16>>'}}
  %0 = "iree_compress.quant_region"(%qw1, %w2, %b1) ({
    ^bb0(%aw1: tensor<4xf32>, %aw2 : tensor<4xf32>, %ab1 : tensor<4xf32>):
      %10 = "xla_hlo.mul"(%aw1, %aw2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %11 = "xla_hlo.add"(%10, %ab1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "iree_compress.return"(%11) : (tensor<4xf32>) -> ()
  }) {config = {logical_kernel = "GENERIC"}} : (tensor<4x!quant.any<i8:f32>>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4x!quant.any<i8:f16>>)
  return %0 : tensor<4x!quant.any<i8:f16>>
}

// -----
func @mismatch_dims() -> (tensor<4xf32>) {
  %w1 = constant dense<0.5> : tensor<2xf32>
  %w2 = constant dense<1.2> : tensor<4xf32>
  %b1 = constant dense<[-1.0, 1.1, 2.2, -3.0]> : tensor<4xf32>
  // @expected-error @+1 {{'iree_compress.quant_region' op incompatible operand to block argument type conversion: 'tensor<2xf32>' does not implicitly widen to 'tensor<4xf32>'}}
  %0 = "iree_compress.quant_region"(%w1, %w2, %b1) ({
    ^bb0(%aw1: tensor<4xf32>, %aw2 : tensor<4xf32>, %ab1 : tensor<4xf32>):
      %10 = "xla_hlo.mul"(%aw1, %aw2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %11 = "xla_hlo.add"(%10, %ab1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "iree_compress.return"(%11) : (tensor<4xf32>) -> ()
  }) {config = {logical_kernel = "GENERIC"}} : (tensor<2xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  return %0 : tensor<4xf32>
}

// -----
func @mismatch_float_type() -> (tensor<4xf32>) {
  %w1 = constant dense<0.5> : tensor<4xf16>
  %w2 = constant dense<1.2> : tensor<4xf32>
  %b1 = constant dense<[-1.0, 1.1, 2.2, -3.0]> : tensor<4xf32>
  // @expected-error @+1 {{'iree_compress.quant_region' op incompatible operand to block argument type conversion: 'tensor<4xf16>' does not implicitly widen to 'tensor<4xf32>'}}
  %0 = "iree_compress.quant_region"(%w1, %w2, %b1) ({
    ^bb0(%aw1: tensor<4xf32>, %aw2 : tensor<4xf32>, %ab1 : tensor<4xf32>):
      %10 = "xla_hlo.mul"(%aw1, %aw2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %11 = "xla_hlo.add"(%10, %ab1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "iree_compress.return"(%11) : (tensor<4xf32>) -> ()
  }) {config = {logical_kernel = "GENERIC"}} : (tensor<4xf16>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  return %0 : tensor<4xf32>
}
