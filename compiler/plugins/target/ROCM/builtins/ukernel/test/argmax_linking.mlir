// RUN: [[ $IREE_ROCM_DISABLE == 1 ]] || iree-compile --split-input-file --iree-hal-target-backends=rocm --iree-rocm-enable-ukernels=all --iree-rocm-target-chip=gfx1100 --compile-to=executable-targets %s | FileCheck %s

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

// -----

// CHECK: llvm.func @__iree_uk_rocm_argmax_F32I32
// CHECK: llvm.call @__iree_uk_rocm_argmax_F32I32
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @argmax_3d_dyn_f32i32(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0xFF800000 : f32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %0 = tensor.empty(%dim, %dim_1) : tensor<?x?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %2 = tensor.empty(%dim, %dim_1) : tensor<?x?xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<?x?x?xf32>) outs(%3, %1 : tensor<?x?xf32>, tensor<?x?xi32>) {
  ^bb0(%in: f32, %out: f32, %out_2: i32):
    %6 = linalg.index 2 : index
    %7 = arith.index_cast %6 : index to i32
    %8 = arith.maximumf %in, %out : f32
    %9 = arith.cmpf ogt, %in, %out : f32
    %10 = arith.select %9, %7, %out_2 : i32
    linalg.yield %8, %10 : f32, i32
  } -> (tensor<?x?xf32>, tensor<?x?xi32>)
  %5 = arith.sitofp %4#1 : tensor<?x?xi32> to tensor<?x?xf32>
  return %5 : tensor<?x?xf32>
}

// -----

// CHECK: llvm.func @__iree_uk_rocm_argmax_F32I64
// CHECK: llvm.call @__iree_uk_rocm_argmax_F32I64
func.func @argmax_3d_dyn_f32i64(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0xFF800000 : f32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %0 = tensor.empty(%dim, %dim_1) : tensor<?x?xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<?x?xi64>) -> tensor<?x?xi64>
  %2 = tensor.empty(%dim, %dim_1) : tensor<?x?xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<?x?x?xf32>) outs(%3, %1 : tensor<?x?xf32>, tensor<?x?xi64>) {
  ^bb0(%in: f32, %out: f32, %out_2: i64):
    %6 = linalg.index 2 : index
    %7 = arith.index_cast %6 : index to i64
    %8 = arith.maximumf %in, %out : f32
    %9 = arith.cmpf ogt, %in, %out : f32
    %10 = arith.select %9, %7, %out_2 : i64
    linalg.yield %8, %10 : f32, i64
  } -> (tensor<?x?xf32>, tensor<?x?xi64>)
  %5 = arith.sitofp %4#1 : tensor<?x?xi64> to tensor<?x?xf32>
  return %5 : tensor<?x?xf32>
}

// -----

// CHECK: llvm.func @__iree_uk_rocm_argmax_F16I32
// CHECK: llvm.call @__iree_uk_rocm_argmax_F16I32
func.func @argmax_3d_dyn_f16i32(%arg0: tensor<?x?x?xf16>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0xFC00 : f16
  %cst_0 = arith.constant 0.000000e+00 : f16
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf16>
  %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf16>
  %0 = tensor.empty(%dim, %dim_1) : tensor<?x?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %2 = tensor.empty(%dim, %dim_1) : tensor<?x?xf16>
  %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<?x?x?xf16>) outs(%3, %1 : tensor<?x?xf16>, tensor<?x?xi32>) {
  ^bb0(%in: f16, %out: f16, %out_2: i32):
    %6 = linalg.index 2 : index
    %7 = arith.index_cast %6 : index to i32
    %8 = arith.maximumf %in, %out : f16
    %9 = arith.cmpf ogt, %in, %out : f16
    %10 = arith.select %9, %7, %out_2 : i32
    linalg.yield %8, %10 : f16, i32
  } -> (tensor<?x?xf16>, tensor<?x?xi32>)
  %5 = arith.sitofp %4#1 : tensor<?x?xi32> to tensor<?x?xf32>
  return %5 : tensor<?x?xf32>
}

// -----

// CHECK: llvm.func @__iree_uk_rocm_argmax_F16I64
// CHECK: llvm.call @__iree_uk_rocm_argmax_F16I64
func.func @argmax_3d_dyn_f16i64(%arg0: tensor<?x?x?xf16>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0xFC00 : f16
  %cst_0 = arith.constant 0.000000e+00 : f16
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf16>
  %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf16>
  %0 = tensor.empty(%dim, %dim_1) : tensor<?x?xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<?x?xi64>) -> tensor<?x?xi64>
  %2 = tensor.empty(%dim, %dim_1) : tensor<?x?xf16>
  %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<?x?x?xf16>) outs(%3, %1 : tensor<?x?xf16>, tensor<?x?xi64>) {
  ^bb0(%in: f16, %out: f16, %out_2: i64):
    %6 = linalg.index 2 : index
    %7 = arith.index_cast %6 : index to i64
    %8 = arith.maximumf %in, %out : f16
    %9 = arith.cmpf ogt, %in, %out : f16
    %10 = arith.select %9, %7, %out_2 : i64
    linalg.yield %8, %10 : f16, i64
  } -> (tensor<?x?xf16>, tensor<?x?xi64>)
  %5 = arith.sitofp %4#1 : tensor<?x?xi64> to tensor<?x?xf32>
  return %5 : tensor<?x?xf32>
}
