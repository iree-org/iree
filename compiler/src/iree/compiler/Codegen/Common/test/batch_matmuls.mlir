// RUN: iree-opt %s \
// RUN: --iree-codegen-transform-dialect-library=%p/batch_matmul_match_spec.mlir \
// RUN: --iree-transform-dialect-interpreter \
// RUN: --split-input-file --verify-diagnostics

!lhs = tensor<128x80x32xf32>
!rhs = tensor<128x32x320xf32>
!res = tensor<128x80x320xf32>

func.func @batch_matmul(%arg0: !lhs, %arg1: !rhs, %arg2: !res) -> !res {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : !res
  // expected-remark @below {{fill}}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !res) -> !res
  // expected-remark @below {{batch matmul}}
  %2 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } ins(%arg0, %arg1 : !lhs, !rhs) outs(%1 : !res) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %3 = arith.mulf %arg3, %arg4 : f32
    %4 = arith.addf %arg5, %3 : f32
    linalg.yield %4 : f32
  } -> !res
  return %2 : !res
}

// -----

!lhs = tensor<128x80x32xf32>
!rhs = tensor<128x32x320xf32>
!res = tensor<128x80x320xf32>

func.func @batch_matmul(%arg0: !lhs, %arg1: !rhs, %arg2: !res) -> !res {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : !res
  // expected-remark @below {{fill}}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !res) -> !res
  // expected-remark @below {{batch matmul}}
  %2 = linalg.batch_matmul ins(%arg0, %arg1 : !lhs, !rhs) outs(%1 : !res) -> !res
  return %2 : !res
}

// -----

!lhs = tensor<80x128x32xf32>
!rhs = tensor<128x32x320xf32>
!res = tensor<80x320x128xf32>

func.func @batch_matmul(%arg0: !lhs, %arg1: !rhs, %arg2: !res) -> !res {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : !res
  // expected-remark @below {{fill}}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !res) -> !res
  // expected-remark @below {{batch matmul}}
  %2 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } ins(%arg0, %arg1 : !lhs, !rhs) outs(%1 : !res) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %3 = arith.mulf %arg3, %arg4 : f32
    %4 = arith.addf %arg5, %3 : f32
    linalg.yield %4 : f32
  } -> !res
  return %2 : !res
}
