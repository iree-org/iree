// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --iree-llvmcpu-target-cpu-features=+dotprod --iree-opt-data-tiling --compile-to=global-optimization | FileCheck %s --implicit-check-not=util.initializer --implicit-check-not=arith.extui
// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --iree-llvmcpu-target-cpu-features=+dotprod --iree-opt-data-tiling --iree-opt-const-eval=false --compile-to=global-optimization | FileCheck %s --check-prefix=NO-CONST-EVAL

// On arm64 with dotprod, global optimization rebases the unsigned operands to
// signed storage and folds the shift of the constant weight. Without constant
// evaluation nothing is rebased.

func.func @u8_linear(%x: tensor<2x8xf32>) -> tensor<2x4xf32> {
  %sx = arith.constant 0.5 : f32
  %zx = arith.constant 100 : i8
  %w = arith.constant dense<[[0, 3, 6, 9, 12, 15, 2, 5], [5, 8, 11, 14, 1, 4, 7, 10], [10, 13, 0, 3, 6, 9, 12, 15], [15, 2, 5, 8, 11, 14, 1, 4]]> : tensor<4x8xi8>
  %sw = arith.constant 0.25 : f32
  %zw = arith.constant 8 : i8
  %qi = tensor.empty() : tensor<2x8xi8>
  %xq = iree_linalg_ext.quantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>],
       quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>
      ins(%x, %sx, %zx : tensor<2x8xf32>, f32, i8) outs(%qi : tensor<2x8xi8>) -> tensor<2x8xi8>
  %xi = tensor.empty() : tensor<2x8xf32>
  %xd = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>],
       input_unsigned, zp_unsigned>
      ins(%xq, %sx, %zx : tensor<2x8xi8>, f32, i8) outs(%xi : tensor<2x8xf32>) -> tensor<2x8xf32>
  %wi = tensor.empty() : tensor<4x8xf32>
  %wd = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>],
       quant_min = 0, quant_max = 15, input_unsigned, zp_unsigned>
      ins(%w, %sw, %zw : tensor<4x8xi8>, f32, i8) outs(%wi : tensor<4x8xf32>) -> tensor<4x8xf32>
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<2x4xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<2x4xf32>) -> tensor<2x4xf32>
  %mm = linalg.matmul
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%xd, %wd : tensor<2x8xf32>, tensor<4x8xf32>) outs(%f : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %mm : tensor<2x4xf32>
}

// A linear layer on a 3D activation, as the torch lowering of view, t and mm
// gives it. Only the second QDQ rewrite matches it.
func.func @u8_linear_3d(%x: tensor<1x2x8xf32>) -> tensor<1x2x4xf32> {
  %sx = arith.constant 0.5 : f32
  %zx = arith.constant 100 : i8
  %w = arith.constant dense<[[1, -2, 3, -4, 5, -6, 7, -8], [0, 1, 2, 3, 4, 5, 6, 7], [-1, -2, -3, -4, -5, -6, -7, -8], [7, 6, 5, 4, 3, 2, 1, 0]]> : tensor<4x8xi8>
  %sw = arith.constant 0.25 : f32
  %qi = tensor.empty() : tensor<1x2x8xi8>
  %xq = iree_linalg_ext.quantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
       quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>
      ins(%x, %sx, %zx : tensor<1x2x8xf32>, f32, i8) outs(%qi : tensor<1x2x8xi8>) -> tensor<1x2x8xi8>
  %xi = tensor.empty() : tensor<1x2x8xf32>
  %xd = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
       input_unsigned, zp_unsigned>
      ins(%xq, %sx, %zx : tensor<1x2x8xi8>, f32, i8) outs(%xi : tensor<1x2x8xf32>) -> tensor<1x2x8xf32>
  %xc = tensor.collapse_shape %xd [[0, 1], [2]] : tensor<1x2x8xf32> into tensor<2x8xf32>
  %wi = tensor.empty() : tensor<4x8xf32>
  %wd = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       quant_min = -8, quant_max = 7>
      ins(%w, %sw : tensor<4x8xi8>, f32) outs(%wi : tensor<4x8xf32>) -> tensor<4x8xf32>
  %ti = tensor.empty() : tensor<8x4xf32>
  %wt = linalg.transpose ins(%wd : tensor<4x8xf32>) outs(%ti : tensor<8x4xf32>) permutation = [1, 0]
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<2x4xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<2x4xf32>) -> tensor<2x4xf32>
  %mm = linalg.matmul ins(%xc, %wt : tensor<2x8xf32>, tensor<8x4xf32>) outs(%f : tensor<2x4xf32>) -> tensor<2x4xf32>
  %y = tensor.expand_shape %mm [[0, 1], [2]] output_shape [1, 2, 4] : tensor<2x4xf32> into tensor<1x2x4xf32>
  return %y : tensor<1x2x4xf32>
}

//       CHECK: util.global private @[[$WEIGHT:[a-z_0-9]+]] {{.+}} = dense<{{\[\[}}-8, -5, -2, 1, 4, 7, -6, -3]
// CHECK-LABEL: util.func public @u8_linear(
//       CHECK:   %[[WEIGHT:.+]] = util.global.load immutable @[[$WEIGHT]] : tensor<4x8xi8>
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%{{.+}}, %[[WEIGHT]] : tensor<2x8xi8>, tensor<4x8xi8>)

// NO-CONST-EVAL-LABEL: util.func public @u8_linear(
//       NO-CONST-EVAL:   iree_linalg_ext.quantize_affine
//  NO-CONST-EVAL-SAME:     quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>
