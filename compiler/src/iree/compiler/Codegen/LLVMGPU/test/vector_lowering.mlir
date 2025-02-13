// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmgpu-vector-lowering,canonicalize,cse))" --split-input-file %s | FileCheck %s

module {
  func.func @broadcast_read_lowering(%arg0: memref<4096x32xf16>) -> vector<1x8xf16> {
    %cst_1 = arith.constant 0.000000e+00 : f16
    %0 = gpu.thread_id  x
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %broadcast_read = vector.transfer_read %arg0[%workgroup_id_x, %0], %cst_1 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, 0)>} : memref<4096x32xf16>, vector<1x8xf16>
    return %broadcast_read : vector<1x8xf16>
  }
}
// CHECK-LABEL: func.func @broadcast_read_lowering
//  CHECK-SAME: (%[[ARG0:.+]]: memref<4096x32xf16>)
//  CHECK: %[[LOAD:.+]] = vector.load %[[ARG0]]{{.*}} : memref<4096x32xf16>
//  CHECK: %[[ELEM:.+]] = vector.extract %[[LOAD]][0] : f16 from vector<1xf16>
//  CHECK: %[[SPLAT:.+]] = vector.splat %[[ELEM]] : vector<8xf16>
//  CHECK: %[[INSERT:.+]] = vector.broadcast %[[SPLAT]] : vector<8xf16> to vector<1x8xf16>
//  CHECK: return %[[INSERT]]

// -----

module {
  func.func @contraction_masked(%lhs: vector<3xf16>, %rhs: vector<2x3xf16>, %acc: vector<2xf32>, %mask: vector<3x2xi1>) -> vector<2xf32> {
    %ret = vector.mask %mask { vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1)>], iterator_types = ["reduction", "parallel"], kind = #vector.kind<add>} %lhs, %rhs, %acc : vector<3xf16>, vector<2x3xf16> into vector<2xf32> } : vector<3x2xi1> -> vector<2xf32>
    return %ret: vector<2xf32>
  }
}

// CHECK-LABEL: func.func @contraction_masked
// CHECK-SAME: %[[LHS:.+]]: vector<3xf16>, %[[RHS:.+]]: vector<2x3xf16>, %[[ACC:.+]]: vector<2xf32>, %[[MASK:.+]]: vector<3x2xi1>
// CHECK: %[[TPRHS:.+]] = vector.transpose %[[RHS]], [1, 0] : vector<2x3xf16> to vector<3x2xf16>
// CHECK: %[[RHS_EXTRACT:.+]] = vector.extract %[[TPRHS]][0] : vector<2xf16> from vector<3x2xf16>
// CHECK: %[[LHS_EXTRACT:.+]] = vector.extract %[[LHS]][0] : f16 from vector<3xf16>
// CHECK: %[[RHS_CAST:.+]] = arith.extf %[[RHS_EXTRACT]] : vector<2xf16> to vector<2xf32>
// CHECK: %[[LHS_CAST:.+]] = arith.extf %[[LHS_EXTRACT]] : f16 to f32
// CHECK: %[[MASK_EXTRACT:.+]] = vector.extract %[[MASK]][0] : vector<2xi1> from vector<3x2xi1>
// CHECK: %[[LHS_SPLAT:.+]] = vector.splat %[[LHS_CAST]] : vector<2xf32>
// CHECK: %[[FMA:.+]] = vector.fma %[[RHS_CAST]], %[[LHS_SPLAT]], %[[ACC]] : vector<2xf32>
// CHECK: arith.select %[[MASK_EXTRACT]], %[[FMA]], %[[ACC]] : vector<2xi1>, vector<2xf32>
