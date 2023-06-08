// RUN: iree-opt --split-input-file --iree-decompose-complex %s | FileCheck %s

func.func @fill_test() -> tensor<2x3xcomplex<f32>> {
    %cst = complex.constant [0.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x3xcomplex<f32>>
    %1 = linalg.fill ins(%cst : complex<f32>) outs(%0 : tensor<2x3xcomplex<f32>>) -> tensor<2x3xcomplex<f32>>
    func.return %1 : tensor<2x3xcomplex<f32>>
}


// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: @fill_test
// CHECK:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x3xcomplex<f32>>
// CHECK:   %[[GENERIC:.+]] = linalg.generic {
// CHECK-SAME: indexing_maps = [#[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: outs(%[[EMPTY]] : tensor<2x3xcomplex<f32>>)
// CHECK:     %[[CREATE:.+]] = complex.create %[[CST]], %[[CST]]
// CHECK:     linalg.yield %[[CREATE]]
// CHECK:   return %[[GENERIC]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func public @linalg_generic_test(%arg0: tensor<2x3xcomplex<f32>>) -> (tensor<2x3xcomplex<f32>>) {
  %cst = complex.constant [-1.000000e+00 : f32, 1.000000e+00 : f32] : complex<f32>
  %0 = tensor.empty() : tensor<2x3xcomplex<f32>>
  %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<2x3xcomplex<f32>>) outs(%0 : tensor<2x3xcomplex<f32>>) {
  ^bb0(%in: complex<f32>, %out: complex<f32>):
    %3 = linalg.index 0 : index
    %4 = linalg.index 1 : index
    %5 = arith.index_cast %3 : index to i32
    %6 = arith.index_cast %4 : index to i32
    %7 = arith.uitofp %5 : i32 to f32
    %8 = arith.uitofp %6 : i32 to f32
    %9 = complex.create %7, %8 : complex<f32>
    %10 = complex.mul %cst, %9 : complex<f32>
    %11 = complex.mul %in, %10 : complex<f32>
    linalg.yield %11 : complex<f32>
  } -> tensor<2x3xcomplex<f32>>
  return %2 : tensor<2x3xcomplex<f32>>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: @linalg_generic_test
// CHECK: %[[C0:.+]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[C1:.+]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<2x3xcomplex<f32>>
// CHECK: %[[GENERIC:.+]] = linalg.generic 
// CHECK-SAME: indexing_maps = [#map, #map]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins(%arg0 : tensor<2x3xcomplex<f32>>)
// CHECK-SAME:outs(%[[EMPTY]] : tensor<2x3xcomplex<f32>>)
// CHECK: ^bb0(%[[IN:.+]]: complex<f32>, %[[OUT:.+]]: complex<f32>):
// CHECK:   %[[IDX0:.+]] = linalg.index 0 : index
// CHECK:   %[[IDX1:.+]] = linalg.index 1 : index
// CHECK:   %[[CST0:.+]] = arith.index_cast %[[IDX0]] : index to i32
// CHECK:   %[[CST1:.+]] = arith.index_cast %[[IDX1]] : index to i32
// CHECK:   %[[FP0:.+]] = arith.uitofp %[[CST0]]
// CHECK:   %[[FP1:.+]] = arith.uitofp %[[CST1]]
// CHECK:   %[[CMPLX0:.+]] = complex.create %[[FP0]], %[[FP1]] : complex<f32>
// CHECK:   %[[CMPLX1:.+]] = complex.create %[[C0]], %[[C1]] : complex<f32>
// CHECK:   %[[MUL0:.+]] = complex.mul %[[CMPLX1]], %[[CMPLX0]] : complex<f32>
// CHECK:   %[[MUL1:.+]] = complex.mul %[[IN]], %[[MUL0]]
// CHECK:   linalg.yield %[[MUL1]]
// CHECK: } -> tensor<2x3xcomplex<f32>>
// CHECK: return %[[GENERIC]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func public @linalg_dual_generic_test(%arg0: tensor<2x3xcomplex<f32>>, %arg1: tensor<2x3xcomplex<f32>>) -> (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) {
  %cst = complex.constant [-1.000000e+00 : f32, 1.000000e+00 : f32] : complex<f32>
  %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%arg0 : tensor<2x3xcomplex<f32>>) {
  ^bb0(%out: complex<f32>):
    %mul = complex.mul %out, %cst : complex<f32>
    linalg.yield %mul : complex<f32>
  } -> tensor<2x3xcomplex<f32>>
  %3 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%arg1 : tensor<2x3xcomplex<f32>>) {
  ^bb0(%out: complex<f32>):
    %mul = complex.div %out, %cst : complex<f32>
    linalg.yield %mul : complex<f32>
  } -> tensor<2x3xcomplex<f32>>
  return %2, %3 : tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>

}

// CHECK-LABEL: @linalg_dual_generic
// CHECK: linalg.generic
// CHECK: complex.create
// CHECK: complex.mul
// CHECK: linalg.generic
// CHECK: complex.create
// CHECK: complex.div
