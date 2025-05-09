// RUN: iree-compile --split-input-file --iree-hal-target-device=hip --iree-hip-enable-ukernels=all --iree-hip-target=gfx1100 --compile-to=executable-targets %s | FileCheck %s

// We want to check that uKernel is indeed generated from e2e workflow.

// CHECK: llvm.func @iree_uk_amdgpu_argmax_f32i64
// CHECK: llvm.call @iree_uk_amdgpu_argmax_f32i64
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
  %expanded_1 = tensor.expand_shape %19#1 [[0, 1]] output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>
  return %expanded_1 : tensor<1x1xi64>
}

// -----

// CHECK: llvm.func @iree_uk_amdgpu_argmax_f16i64
// CHECK: llvm.call @iree_uk_amdgpu_argmax_f16i64
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
  %expanded_1 = tensor.expand_shape %19#1 [[0, 1]]  output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>
  return %expanded_1 : tensor<1x1xi64>
}

// -----

// CHECK: llvm.func @iree_uk_amdgpu_argmax_bf16i32
// CHECK: llvm.call @iree_uk_amdgpu_argmax_bf16i32
func.func @argmax_1d_bf16i32(%arg0: tensor<1x?xbf16>) -> tensor<1x1xi32> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0xFF80 : bf16  // -inf in bf16
  %c0 = arith.constant 0 : index

  %init_val = tensor.empty() : tensor<1xbf16>
  %init_idx = tensor.empty() : tensor<1xi32>
  %filled_val = linalg.fill ins(%cst : bf16) outs(%init_val : tensor<1xbf16>) -> tensor<1xbf16>
  %filled_idx = linalg.fill ins(%c0_i32 : i32) outs(%init_idx : tensor<1xi32>) -> tensor<1xi32>

  %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%arg0 : tensor<1x?xbf16>) outs(%filled_val, %filled_idx : tensor<1xbf16>, tensor<1xi32>) {
    ^bb0(%in: bf16, %cur_val: bf16, %cur_idx: i32):
      %i = linalg.index 1 : index
      %i32 = arith.index_cast %i : index to i32
      %max = arith.maximumf %in, %cur_val : bf16
      %pred = arith.cmpf ogt, %in, %cur_val : bf16
      %new_idx = arith.select %pred, %i32, %cur_idx : i32
      linalg.yield %max, %new_idx : bf16, i32
  } -> (tensor<1xbf16>, tensor<1xi32>)

  %expanded = tensor.expand_shape %result#1 [[0, 1]] output_shape [1, 1] : tensor<1xi32> into tensor<1x1xi32>
  return %expanded : tensor<1x1xi32>
}

// -----

// CHECK: llvm.func @iree_uk_amdgpu_argmax_bf16i64
// CHECK: llvm.call @iree_uk_amdgpu_argmax_bf16i64
func.func @argmax_1d_bf16i64(%arg0: tensor<1x?xbf16>) -> tensor<1x1xi64> {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF80 : bf16  // -inf in bf16
  %c0 = arith.constant 0 : index

  %init_val = tensor.empty() : tensor<1xbf16>
  %init_idx = tensor.empty() : tensor<1xi64>
  %filled_val = linalg.fill ins(%cst : bf16) outs(%init_val : tensor<1xbf16>) -> tensor<1xbf16>
  %filled_idx = linalg.fill ins(%c0_i64 : i64) outs(%init_idx : tensor<1xi64>) -> tensor<1xi64>

  %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%arg0 : tensor<1x?xbf16>) outs(%filled_val, %filled_idx : tensor<1xbf16>, tensor<1xi64>) {
    ^bb0(%in: bf16, %cur_val: bf16, %cur_idx: i64):
      %i = linalg.index 1 : index
      %i64 = arith.index_cast %i : index to i64
      %max = arith.maximumf %in, %cur_val : bf16
      %pred = arith.cmpf ogt, %in, %cur_val : bf16
      %new_idx = arith.select %pred, %i64, %cur_idx : i64
      linalg.yield %max, %new_idx : bf16, i64
  } -> (tensor<1xbf16>, tensor<1xi64>)

  %expanded = tensor.expand_shape %result#1 [[0, 1]] output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>
  return %expanded : tensor<1x1xi64>
}

// -----

// CHECK: llvm.func @iree_uk_amdgpu_argmax_bf16i32
// CHECK: llvm.call @iree_uk_amdgpu_argmax_bf16i32
func.func @argmax_2d_bf16i32(%arg0: tensor<16x?xbf16>) -> tensor<16x1xi32> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0xFF80 : bf16  // -inf for bf16

  %init_val = tensor.empty() : tensor<16xbf16>
  %init_idx = tensor.empty() : tensor<16xi32>
  %filled_val = linalg.fill ins(%cst : bf16) outs(%init_val : tensor<16xbf16>) -> tensor<16xbf16>
  %filled_idx = linalg.fill ins(%c0_i32 : i32) outs(%init_idx : tensor<16xi32>) -> tensor<16xi32>

  %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%arg0 : tensor<16x?xbf16>) outs(%filled_val, %filled_idx : tensor<16xbf16>, tensor<16xi32>) {
    ^bb0(%in: bf16, %cur_val: bf16, %cur_idx: i32):
      %j = linalg.index 1 : index
      %j32 = arith.index_cast %j : index to i32
      %max = arith.maximumf %in, %cur_val : bf16
      %pred = arith.cmpf ogt, %in, %cur_val : bf16
      %sel_idx = arith.select %pred, %j32, %cur_idx : i32
      linalg.yield %max, %sel_idx : bf16, i32
  } -> (tensor<16xbf16>, tensor<16xi32>)

  %expanded = tensor.expand_shape %result#1 [[0, 1]] output_shape [16, 1] : tensor<16xi32> into tensor<16x1xi32>
  return %expanded : tensor<16x1xi32>
}

// -----

// CHECK: llvm.func @iree_uk_amdgpu_argmax_bf16i64
// CHECK: llvm.call @iree_uk_amdgpu_argmax_bf16i64
func.func @argmax_2d_bf16i64(%arg0: tensor<16x?xbf16>) -> tensor<16x1xi64> {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF80 : bf16  // -inf for bf16
  %c0 = arith.constant 0 : index

  %init_idx = tensor.empty() : tensor<16xi64>
  %init_val = tensor.empty() : tensor<16xbf16>
  %filled_idx = linalg.fill ins(%c0_i64 : i64) outs(%init_idx : tensor<16xi64>) -> tensor<16xi64>
  %filled_val = linalg.fill ins(%cst : bf16) outs(%init_val : tensor<16xbf16>) -> tensor<16xbf16>

  %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%arg0 : tensor<16x?xbf16>) outs(%filled_val, %filled_idx : tensor<16xbf16>, tensor<16xi64>) {
    ^bb0(%in: bf16, %cur_val: bf16, %cur_idx: i64):
      %j = linalg.index 1 : index
      %j64 = arith.index_cast %j : index to i64
      %max = arith.maximumf %in, %cur_val : bf16
      %pred = arith.cmpf ogt, %in, %cur_val : bf16
      %sel_idx = arith.select %pred, %j64, %cur_idx : i64
      linalg.yield %max, %sel_idx : bf16, i64
  } -> (tensor<16xbf16>, tensor<16xi64>)

  %expanded = tensor.expand_shape %result#1 [[0, 1]] output_shape [16, 1] : tensor<16xi64> into tensor<16x1xi64>
  return %expanded : tensor<16x1xi64>
}

// -----

// CHECK: llvm.func @iree_uk_amdgpu_argmax_f32i64
// CHECK: llvm.call @iree_uk_amdgpu_argmax_f32i64
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
  %expanded_1 = tensor.expand_shape %19#1 [[0, 1]] output_shape [16, 1] : tensor<16xi64> into tensor<16x1xi64>
  return %expanded_1 : tensor<16x1xi64>
}

// -----

// CHECK: llvm.func @iree_uk_amdgpu_argmax_f32i32
// CHECK: llvm.call @iree_uk_amdgpu_argmax_f32i32
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

// CHECK: llvm.func @iree_uk_amdgpu_argmax_f32i64
// CHECK: llvm.call @iree_uk_amdgpu_argmax_f32i64
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

// CHECK: llvm.func @iree_uk_amdgpu_argmax_bf16i32
// CHECK: llvm.call @iree_uk_amdgpu_argmax_bf16i32
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @argmax_3d_dyn_bf16i32(%arg0: tensor<?x?x?xbf16>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0xFF80 : bf16  // -inf for bf16

  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?x?xbf16>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x?x?xbf16>

  %init_val = tensor.empty(%dim0, %dim1) : tensor<?x?xbf16>
  %init_idx = tensor.empty(%dim0, %dim1) : tensor<?x?xi32>

  %filled_val = linalg.fill ins(%cst : bf16) outs(%init_val : tensor<?x?xbf16>) -> tensor<?x?xbf16>
  %filled_idx = linalg.fill ins(%c0_i32 : i32) outs(%init_idx : tensor<?x?xi32>) -> tensor<?x?xi32>

  %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%arg0 : tensor<?x?x?xbf16>) outs(%filled_val, %filled_idx : tensor<?x?xbf16>, tensor<?x?xi32>) {
    ^bb0(%in: bf16, %cur_val: bf16, %cur_idx: i32):
      %k = linalg.index 2 : index
      %k_i32 = arith.index_cast %k : index to i32
      %max = arith.maximumf %in, %cur_val : bf16
      %pred = arith.cmpf ogt, %in, %cur_val : bf16
      %sel_idx = arith.select %pred, %k_i32, %cur_idx : i32
      linalg.yield %max, %sel_idx : bf16, i32
  } -> (tensor<?x?xbf16>, tensor<?x?xi32>)

  %result_f32 = arith.sitofp %result#1 : tensor<?x?xi32> to tensor<?x?xf32>
  return %result_f32 : tensor<?x?xf32>
}

// -----

// CHECK: llvm.func @iree_uk_amdgpu_argmax_bf16i64
// CHECK: llvm.call @iree_uk_amdgpu_argmax_bf16i64
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @argmax_3d_dyn_bf16i64(%arg0: tensor<?x?x?xbf16>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF80 : bf16  // -inf for bf16

  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?x?xbf16>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x?x?xbf16>

  %init_val = tensor.empty(%dim0, %dim1) : tensor<?x?xbf16>
  %init_idx = tensor.empty(%dim0, %dim1) : tensor<?x?xi64>

  %filled_val = linalg.fill ins(%cst : bf16) outs(%init_val : tensor<?x?xbf16>) -> tensor<?x?xbf16>
  %filled_idx = linalg.fill ins(%c0_i64 : i64) outs(%init_idx : tensor<?x?xi64>) -> tensor<?x?xi64>

  %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%arg0 : tensor<?x?x?xbf16>) outs(%filled_val, %filled_idx : tensor<?x?xbf16>, tensor<?x?xi64>) {
    ^bb0(%in: bf16, %cur_val: bf16, %cur_idx: i64):
      %k = linalg.index 2 : index
      %k_i64 = arith.index_cast %k : index to i64
      %max = arith.maximumf %in, %cur_val : bf16
      %pred = arith.cmpf ogt, %in, %cur_val : bf16
      %sel_idx = arith.select %pred, %k_i64, %cur_idx : i64
      linalg.yield %max, %sel_idx : bf16, i64
  } -> (tensor<?x?xbf16>, tensor<?x?xi64>)

  %result_f32 = arith.sitofp %result#1 : tensor<?x?xi64> to tensor<?x?xf32>
  return %result_f32 : tensor<?x?xf32>
}

// -----

// CHECK: llvm.func @iree_uk_amdgpu_argmax_f16i32
// CHECK: llvm.call @iree_uk_amdgpu_argmax_f16i32
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

// CHECK: llvm.func @iree_uk_amdgpu_argmax_f16i64
// CHECK: llvm.call @iree_uk_amdgpu_argmax_f16i64
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
