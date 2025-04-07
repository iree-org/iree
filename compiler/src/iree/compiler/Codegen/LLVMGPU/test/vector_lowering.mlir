// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmgpu-vector-lowering{unroll=0},canonicalize,cse))" --split-input-file %s | FileCheck %s --check-prefix=NOUNROLL
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmgpu-vector-lowering{unroll=1},canonicalize,cse))" --split-input-file %s | FileCheck %s --check-prefix=UNROLL

module {
  func.func @broadcast_read_lowering(%arg0: memref<4096x32xf16>) -> vector<1x8xf16> {
    %cst_1 = arith.constant 0.000000e+00 : f16
    %0 = gpu.thread_id  x
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %broadcast_read = vector.transfer_read %arg0[%workgroup_id_x, %0], %cst_1 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, 0)>} : memref<4096x32xf16>, vector<1x8xf16>
    return %broadcast_read : vector<1x8xf16>
  }
}
// NOUNROLL-LABEL: func.func @broadcast_read_lowering
//  NOUNROLL-SAME: (%[[ARG0:.+]]: memref<4096x32xf16>)
//       NOUNROLL:   %[[LOAD:.+]] = vector.load %[[ARG0]]{{.*}} : memref<4096x32xf16>
//       NOUNROLL:   %[[ELEM:.+]] = vector.extract %[[LOAD]][0] : f16 from vector<1xf16>
//       NOUNROLL:   %[[SPLAT:.+]] = vector.splat %[[ELEM]] : vector<8xf16>
//       NOUNROLL:   %[[INSERT:.+]] = vector.broadcast %[[SPLAT]] : vector<8xf16> to vector<1x8xf16>
//       NOUNROLL:   return %[[INSERT]]

// -----

module {
  func.func @contraction_masked(%lhs: vector<3xf16>, %rhs: vector<2x3xf16>, %acc: vector<2xf32>, %mask: vector<3x2xi1>) -> vector<2xf32> {
    %ret = vector.mask %mask { vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1)>], iterator_types = ["reduction", "parallel"], kind = #vector.kind<add>} %lhs, %rhs, %acc : vector<3xf16>, vector<2x3xf16> into vector<2xf32> } : vector<3x2xi1> -> vector<2xf32>
    return %ret: vector<2xf32>
  }
}

// NOUNROLL-LABEL: func.func @contraction_masked
//  NOUNROLL-SAME: %[[LHS:.+]]: vector<3xf16>, %[[RHS:.+]]: vector<2x3xf16>,
//  NOUNROLL-SAME: %[[ACC:.+]]: vector<2xf32>, %[[MASK:.+]]: vector<3x2xi1>
//       NOUNROLL:   %[[TPRHS:.+]] = vector.transpose %[[RHS]], [1, 0] : vector<2x3xf16> to vector<3x2xf16>
//       NOUNROLL:   %[[RHS_EXTRACT:.+]] = vector.extract %[[TPRHS]][0] : vector<2xf16> from vector<3x2xf16>
//       NOUNROLL:   %[[LHS_EXTRACT:.+]] = vector.extract %[[LHS]][0] : f16 from vector<3xf16>
//       NOUNROLL:   %[[RHS_CAST:.+]] = arith.extf %[[RHS_EXTRACT]] : vector<2xf16> to vector<2xf32>
//       NOUNROLL:   %[[LHS_CAST:.+]] = arith.extf %[[LHS_EXTRACT]] : f16 to f32
//       NOUNROLL:   %[[MASK_EXTRACT:.+]] = vector.extract %[[MASK]][0] : vector<2xi1> from vector<3x2xi1>
//       NOUNROLL:   %[[LHS_SPLAT:.+]] = vector.splat %[[LHS_CAST]] : vector<2xf32>
//       NOUNROLL:   %[[FMA:.+]] = vector.fma %[[RHS_CAST]], %[[LHS_SPLAT]], %[[ACC]] : vector<2xf32>
//       NOUNROLL:   arith.select %[[MASK_EXTRACT]], %[[FMA]], %[[ACC]] : vector<2xi1>, vector<2xf32>

// With unrolling, the transpose gets decomposed into transposes on slices in the final dimension of the result.
//
// UNROLL-LABEL: func.func @contraction_masked
//  UNROLL-SAME: %[[LHS:.+]]: vector<3xf16>, %[[RHS:.+]]: vector<2x3xf16>, %[[ACC:.+]]: vector<2xf32>, %[[MASK:.+]]: vector<3x2xi1>
//       UNROLL:   %[[STRIDED_SLICE_0:.+]] = vector.extract_strided_slice %[[RHS]] {offsets = [0, 0], sizes = [2, 1], strides = [1, 1]} : vector<2x3xf16> to vector<2x1xf16>
//       UNROLL:   %[[TRANSPOSE_0:.+]] = vector.transpose %[[STRIDED_SLICE_0]], [1, 0] : vector<2x1xf16> to vector<1x2xf16>
//       UNROLL:   %[[STRIDED_SLICE_1:.+]] = vector.extract_strided_slice %[[RHS]] {offsets = [0, 1], sizes = [2, 1], strides = [1, 1]} : vector<2x3xf16> to vector<2x1xf16>
//       UNROLL:                         vector.transpose %[[STRIDED_SLICE_1]], [1, 0] : vector<2x1xf16> to vector<1x2xf16>
//       UNROLL:   %[[STRIDED_SLICE_2:.+]] = vector.extract_strided_slice %[[RHS]] {offsets = [0, 2], sizes = [2, 1], strides = [1, 1]} : vector<2x3xf16> to vector<2x1xf16>
//       UNROLL:                         vector.transpose %[[STRIDED_SLICE_2]], [1, 0] : vector<2x1xf16> to vector<1x2xf16>
//       UNROLL:   %[[LHS_EXTRACT_0:.+]] = vector.extract %[[TRANSPOSE_0]][0] : vector<2xf16> from vector<1x2xf16>
//       UNROLL:   %[[CAST_0:.+]] = arith.extf %[[LHS_EXTRACT_0]] : vector<2xf16> to vector<2xf32>


// -----

func.func @test_unroll(%arg0: vector<4x1x1x1x4xf32>, %arg1: vector<4x1x1x1x4xf32>) -> vector<2x1x1xf16> {
  %cst = arith.constant dense<0.000000e+00> : vector<2x1x1xf16>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<2x4x1x1x1x4xf32>
  %0 = vector.insert %arg0, %cst_0 [0] : vector<4x1x1x1x4xf32> into vector<2x4x1x1x1x4xf32>
  %1 = vector.insert %arg1, %0 [1] : vector<4x1x1x1x4xf32> into vector<2x4x1x1x1x4xf32>
  %2 = arith.truncf %1 : vector<2x4x1x1x1x4xf32> to vector<2x4x1x1x1x4xf16>
  %3 = vector.multi_reduction <maximumf>, %2, %cst [1, 3, 5] : vector<2x4x1x1x1x4xf16> to vector<2x1x1xf16>
  return %3 : vector<2x1x1xf16>
}

// The arith.tuncf gets unrolled to 8 truncf operations on 4 elements
//    UNROLL-LABEL: func.func @test_unroll
//     UNROLL-SAME: (%[[ARG0:.+]]: vector<4x1x1x1x4xf32>, %[[ARG1:.+]]: vector<4x1x1x1x4xf32>) -> vector<2x1x1xf16>
//          UNROLL:   %[[EXTRACT_0:.+]] = vector.extract %[[ARG0]][0, 0, 0, 0] : vector<4xf32> from vector<4x1x1x1x4xf32>
//          UNROLL:   %[[RESHAPE_0:.+]] = vector.shape_cast %[[EXTRACT_0]] : vector<4xf32> to vector<1x1x1x1x1x4xf32>
//          UNROLL:     arith.truncf %[[RESHAPE_0]] : vector<1x1x1x1x1x4xf32> to vector<1x1x1x1x1x4xf16>
//  UNROLL-COUNT-7:     arith.truncf {{.*}} : vector<1x1x1x1x1x4xf32> to vector<1x1x1x1x1x4xf16>
//      UNROLL-NOT:     arith.truncf
// UNROLL-COUNT-16:     arith.maximumf {{.*}} : vector<2xf16>
//      UNROLL-NOT:     arith.maximumf
//          UNROLL:     return {{.*}} : vector<2x1x1xf16>
