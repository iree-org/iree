// RUN: [[ $IREE_ROCM_DISABLE == 1 ]] || iree-compile --split-input-file --iree-hal-target-backends=rocm --iree-rocm-enable-ukernels=argmax --iree-rocm-target-chip=gfx1100 --compile-to=executable-targets %s | FileCheck %s

// We want to check that uKernel is indeed generated from e2e workflow.

// CHECK: llvm.func @__iree_uk_rocm_argmax_F32I64
// CHECK: llvm.call @__iree_uk_rocm_argmax_F32I64
func.func @argmax_1d_f32i64(%arg0: tensor<1x?xf32>) -> tensor<1x1xi64> {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c-1_i64 = arith.constant -1 : i64
  %15 = tensor.empty() : tensor<1xi64>
  %16 = linalg.fill ins(%c0_i64 : i64) outs(%15 : tensor<1xi64>) -> tensor<1xi64>
  %17 = tensor.empty() : tensor<1xf32>
  %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<1xf32>) -> tensor<1xf32>
  %19:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1x?xf32>) outs(%18, %16 : tensor<1xf32>, tensor<1xi64>) {
  ^bb0(%in: f32, %out: f32, %out_2: i64):
    %20 = linalg.index 1 : index
    %21 = arith.index_cast %20 : index to i64
    %22 = arith.maximumf %in, %out : f32
    %23 = arith.cmpf ogt, %in, %out : f32
    %24 = arith.select %23, %21, %out_2 : i64
    linalg.yield %22, %24 : f32, i64
  } -> (tensor<1xf32>, tensor<1xi64>)
  %expanded_1 = tensor.expand_shape %19#1 [[0, 1]] : tensor<1xi64> into tensor<1x1xi64>
  return %expanded_1 : tensor<1x1xi64>
}

// -----

// CHECK: llvm.func @__iree_uk_rocm_argmax_F16I64
// CHECK: llvm.call @__iree_uk_rocm_argmax_F16I64
func.func @argmax_1d_f16i64(%arg0: tensor<1x?xf16>) -> tensor<1x1xi64> {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFC00 : f16
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c-1_i64 = arith.constant -1 : i64
  %15 = tensor.empty() : tensor<1xi64>
  %16 = linalg.fill ins(%c0_i64 : i64) outs(%15 : tensor<1xi64>) -> tensor<1xi64>
  %17 = tensor.empty() : tensor<1xf16>
  %18 = linalg.fill ins(%cst : f16) outs(%17 : tensor<1xf16>) -> tensor<1xf16>
  %19:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1x?xf16>) outs(%18, %16 : tensor<1xf16>, tensor<1xi64>) {
  ^bb0(%in: f16, %out: f16, %out_2: i64):
    %20 = linalg.index 1 : index
    %21 = arith.index_cast %20 : index to i64
    %22 = arith.maximumf %in, %out : f16
    %23 = arith.cmpf ogt, %in, %out : f16
    %24 = arith.select %23, %21, %out_2 : i64
    linalg.yield %22, %24 : f16, i64
  } -> (tensor<1xf16>, tensor<1xi64>)
  %expanded_1 = tensor.expand_shape %19#1 [[0, 1]] : tensor<1xi64> into tensor<1x1xi64>
  return %expanded_1 : tensor<1x1xi64>
}


// -----

// CHECK: llvm.func @__iree_uk_rocm_argmax_F32I64
// CHECK: llvm.call @__iree_uk_rocm_argmax_F32I64
func.func @argmax_2d_f32i64(%arg0: tensor<16x?xf32>) -> tensor<16x1xi64> {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c-1_i64 = arith.constant -1 : i64
  %15 = tensor.empty() : tensor<16xi64>
  %16 = linalg.fill ins(%c0_i64 : i64) outs(%15 : tensor<16xi64>) -> tensor<16xi64>
  %17 = tensor.empty() : tensor<16xf32>
  %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<16xf32>) -> tensor<16xf32>
  %19:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<16x?xf32>) outs(%18, %16 : tensor<16xf32>, tensor<16xi64>) {
  ^bb0(%in: f32, %out: f32, %out_2: i64):
    %20 = linalg.index 1 : index
    %21 = arith.index_cast %20 : index to i64
    %22 = arith.maximumf %in, %out : f32
    %23 = arith.cmpf ogt, %in, %out : f32
    %24 = arith.select %23, %21, %out_2 : i64
    linalg.yield %22, %24 : f32, i64
  } -> (tensor<16xf32>, tensor<16xi64>)
  %expanded_1 = tensor.expand_shape %19#1 [[0, 1]] : tensor<16xi64> into tensor<16x1xi64>
  return %expanded_1 : tensor<16x1xi64>
}
