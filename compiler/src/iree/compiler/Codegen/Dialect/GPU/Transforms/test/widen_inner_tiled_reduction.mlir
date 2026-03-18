// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-gpu-widen-inner-tiled-reduction))' --split-input-file | FileCheck %s

// F16 variant (layout group 1, f32 ACC, addf collapse).
func.func @vdmfma_f16(
    %lhs: vector<8xf16>, %rhs: vector<16xf16>, %acc: vector<2xf32>,
    %lb: index, %ub: index, %step: index) -> vector<2xf32> {
  %result = scf.for %iv = %lb to %ub step %step iter_args(%iter_acc = %acc) -> vector<2xf32> {
    %mma = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%iter_acc) {
      indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>],
      iterator_types = [],
      kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x64_F16>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true>
    } : vector<8xf16>, vector<16xf16> into vector<2xf32>
    scf.yield %mma : vector<2xf32>
  }
  return %result : vector<2xf32>
}

// CHECK-LABEL: func @vdmfma_f16
//  CHECK-SAME:   %[[LHS:.+]]: vector<8xf16>, %[[RHS:.+]]: vector<16xf16>, %[[ACC:.+]]: vector<2xf32>
//       CHECK:   %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<2xf32>
//       CHECK:   %[[EXPAND:.+]] = vector.interleave %[[ACC]], %[[ZERO]] : vector<2xf32>
//       CHECK:   %[[FOR:.+]] = scf.for {{.*}} iter_args(%[[ITER:.+]] = %[[EXPAND]]) -> (vector<4xf32>)
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ITER]])
//  CHECK-SAME:       kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x64_F16>
//  CHECK-SAME:       semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true, promotedAcc = true>
//  CHECK-SAME:       : vector<8xf16>, vector<16xf16> into vector<4xf32>
//       CHECK:     scf.yield %[[MMA]] : vector<4xf32>
//       CHECK:   %[[EVENS:.+]], %[[ODDS:.+]] = vector.deinterleave %[[FOR]] : vector<4xf32> -> vector<2xf32>
//       CHECK:   %[[COLLAPSED:.+]] = arith.addf %[[EVENS]], %[[ODDS]] : vector<2xf32>
//       CHECK:   return %[[COLLAPSED]] : vector<2xf32>

// -----

// BF16 variant (layout group 1, f32 ACC, addf collapse).
func.func @vdmfma_bf16(
    %lhs: vector<8xbf16>, %rhs: vector<16xbf16>, %acc: vector<2xf32>,
    %lb: index, %ub: index, %step: index) -> vector<2xf32> {
  %result = scf.for %iv = %lb to %ub step %step iter_args(%iter_acc = %acc) -> vector<2xf32> {
    %mma = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%iter_acc) {
      indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>],
      iterator_types = [],
      kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x64_BF16>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true>
    } : vector<8xbf16>, vector<16xbf16> into vector<2xf32>
    scf.yield %mma : vector<2xf32>
  }
  return %result : vector<2xf32>
}

// CHECK-LABEL: func @vdmfma_bf16
//       CHECK:   vector.interleave %{{.+}}, %{{.+}} : vector<2xf32>
//       CHECK:   scf.for {{.*}} -> (vector<4xf32>)
//       CHECK:     iree_codegen.inner_tiled
//  CHECK-SAME:       kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x64_BF16>
//  CHECK-SAME:       promotedAcc = true
//  CHECK-SAME:       : vector<8xbf16>, vector<16xbf16> into vector<4xf32>
//       CHECK:   vector.deinterleave %{{.+}} : vector<4xf32> -> vector<2xf32>
//       CHECK:   arith.addf {{.*}} : vector<2xf32>

// -----

// I8 variant (layout group 2, i32 ACC, addi collapse).
func.func @vdmfma_i8(
    %lhs: vector<16xi8>, %rhs: vector<32xi8>, %acc: vector<2xi32>,
    %lb: index, %ub: index, %step: index) -> vector<2xi32> {
  %result = scf.for %iv = %lb to %ub step %step iter_args(%iter_acc = %acc) -> vector<2xi32> {
    %mma = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%iter_acc) {
      indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>],
      iterator_types = [],
      kind = #iree_gpu.virtual_mma_layout<VDMFMA_I32_8x16x128_I8>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true>
    } : vector<16xi8>, vector<32xi8> into vector<2xi32>
    scf.yield %mma : vector<2xi32>
  }
  return %result : vector<2xi32>
}

// CHECK-LABEL: func @vdmfma_i8
//  CHECK-SAME:   %[[LHS:.+]]: vector<16xi8>, %[[RHS:.+]]: vector<32xi8>, %[[ACC:.+]]: vector<2xi32>
//       CHECK:   %[[ZERO:.+]] = arith.constant dense<0> : vector<2xi32>
//       CHECK:   %[[EXPAND:.+]] = vector.interleave %[[ACC]], %[[ZERO]] : vector<2xi32>
//       CHECK:   %[[FOR:.+]] = scf.for {{.*}} iter_args(%[[ITER:.+]] = %[[EXPAND]]) -> (vector<4xi32>)
//       CHECK:     iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ITER]])
//  CHECK-SAME:       kind = #iree_gpu.virtual_mma_layout<VDMFMA_I32_8x16x128_I8>
//  CHECK-SAME:       promotedAcc = true
//  CHECK-SAME:       : vector<16xi8>, vector<32xi8> into vector<4xi32>
//       CHECK:   %[[EVENS:.+]], %[[ODDS:.+]] = vector.deinterleave %[[FOR]] : vector<4xi32> -> vector<2xi32>
//       CHECK:   %[[COLLAPSED:.+]] = arith.addi %[[EVENS]], %[[ODDS]] : vector<2xi32>
//       CHECK:   return %[[COLLAPSED]] : vector<2xi32>

// -----

// F8E4M3FNUZ variant (layout group 2, f32 ACC, addf collapse).
func.func @vdmfma_f8(
    %lhs: vector<16xf8E4M3FNUZ>, %rhs: vector<32xf8E4M3FNUZ>, %acc: vector<2xf32>,
    %lb: index, %ub: index, %step: index) -> vector<2xf32> {
  %result = scf.for %iv = %lb to %ub step %step iter_args(%iter_acc = %acc) -> vector<2xf32> {
    %mma = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%iter_acc) {
      indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>],
      iterator_types = [],
      kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x128_F8E4M3FNUZ>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true>
    } : vector<16xf8E4M3FNUZ>, vector<32xf8E4M3FNUZ> into vector<2xf32>
    scf.yield %mma : vector<2xf32>
  }
  return %result : vector<2xf32>
}

// CHECK-LABEL: func @vdmfma_f8
//       CHECK:   vector.interleave %{{.+}}, %{{.+}} : vector<2xf32>
//       CHECK:   scf.for {{.*}} -> (vector<4xf32>)
//       CHECK:     iree_codegen.inner_tiled
//  CHECK-SAME:       kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x128_F8E4M3FNUZ>
//  CHECK-SAME:       promotedAcc = true
//  CHECK-SAME:       : vector<16xf8E4M3FNUZ>, vector<32xf8E4M3FNUZ> into vector<4xf32>
//       CHECK:   vector.deinterleave %{{.+}} : vector<4xf32> -> vector<2xf32>
//       CHECK:   arith.addf {{.*}} : vector<2xf32>
