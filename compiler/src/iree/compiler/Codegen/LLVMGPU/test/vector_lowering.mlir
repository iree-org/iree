// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmgpu-vector-lowering,canonicalize,cse))" --split-input-file %s | FileCheck %s


//module {
//  func.func @broadcast_read_lowering(%arg0: memref<4096x32xf16>) -> vector<1x8xf16> {
//    %cst_1 = arith.constant 0.000000e+00 : f16
//    %0 = gpu.thread_id  x
//    %workgroup_id_x = hal.interface.workgroup.id[0] : index
//    %broadcast_read = vector.transfer_read %arg0[%workgroup_id_x, %0], %cst_1 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, 0)>} : memref<4096x32xf16>, vector<1x8xf16>
//    return %broadcast_read : vector<1x8xf16>
//  }
//}
//// CHECK-LABEL: func.func @broadcast_read_lowering
////  CHECK-SAME: (%[[ARG0:.+]]: memref<4096x32xf16>)
////  CHECK: %[[LOAD:.+]] = vector.load %[[ARG0]]{{.*}} : memref<4096x32xf16>
////  CHECK: %[[ELEM:.+]] = vector.extract %[[LOAD]][0] : f16 from vector<1xf16>
////  CHECK: %[[INSERT:.+]] = vector.broadcast %[[ELEM]] : f16 to vector<1x8xf16>
////  CHECK: return %[[INSERT]]
//
//// -----
//
//module {
//  func.func @contraction_masked(%lhs: vector<3xf16>, %rhs: vector<2x3xf16>, %acc: vector<2xf32>, %mask: vector<3x2xi1>) -> vector<2xf32> {
//    %ret = vector.mask %mask { vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1)>], iterator_types = ["reduction", "parallel"], kind = #vector.kind<add>} %lhs, %rhs, %acc : vector<3xf16>, vector<2x3xf16> into vector<2xf32> } : vector<3x2xi1> -> vector<2xf32>
//    return %ret: vector<2xf32>
//  }
//}
//
//// CHECK-LABEL: func.func @contraction_masked
//// CHECK-SAME: %[[LHS:.+]]: vector<3xf16>, %[[RHS:.+]]: vector<2x3xf16>, %[[ACC:.+]]: vector<2xf32>, %[[MASK:.+]]: vector<3x2xi1>
//// CHECK: %[[TPRHS:.+]] = vector.transpose %[[RHS]], [1, 0] : vector<2x3xf16> to vector<3x2xf16>
//// CHECK: %[[RHS_EXTRACT:.+]] = vector.extract %[[TPRHS]][0] : vector<2xf16> from vector<3x2xf16>
//// CHECK: %[[LHS_EXTRACT:.+]] = vector.extract %[[LHS]][0] : f16 from vector<3xf16>
//// CHECK: %[[RHS_CAST:.+]] = arith.extf %[[RHS_EXTRACT]] : vector<2xf16> to vector<2xf32>
//// CHECK: %[[LHS_CAST:.+]] = arith.extf %[[LHS_EXTRACT]] : f16 to f32
//// CHECK: %[[MASK_EXTRACT:.+]] = vector.extract %[[MASK]][0] : vector<2xi1> from vector<3x2xi1>
//// CHECK: %[[LHS_BROADCAST:.+]] = vector.broadcast %[[LHS_CAST]] : f32 to vector<2xf32>
//// CHECK: %[[FMA:.+]] = vector.fma %[[RHS_CAST]], %[[LHS_BROADCAST]], %[[ACC]] : vector<2xf32>
//// CHECK: arith.select %[[MASK_EXTRACT]], %[[FMA]], %[[ACC]] : vector<2xi1>, vector<2xf32>
//
//// -----
//
//// Test to check if masked transfer reads masked on an outer dimension lowers
//// to:
////  cond = dim >= id
////  vector.maskedload mem[..., id, ...] broadcast(cond)
//func.func @partial_masked_transfer_read(%mem : memref<16x?x32xf16>) -> vector<1x2x16xf16> {
//  %cst = ub.poison : f16
//  %c0 = arith.constant 0 : index
//  %c1 = arith.constant 1 : index
//  %c16 = arith.constant 16 : index
//  %dim = memref.dim %mem, %c1 : memref<16x?x32xf16>
//  %mask = vector.create_mask %c1, %dim, %c16 : vector<1x2x16xi1>
//  %read = vector.transfer_read %mem[%c0, %c0, %c0], %cst, %mask {in_bounds = [true, true, true]} : memref<16x?x32xf16>, vector<1x2x16xf16>
//  return %read : vector<1x2x16xf16>
//}
//
//// CHECK-LABEL: func.func @partial_masked_transfer_read
//// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
//// CHECK-DAG: %[[DIM:.+]] = memref.dim
//// CHECK-DAG: %[[COND:.+]] = arith.cmpi sgt, %[[DIM]], %[[C0]] : index
//// CHECK-DAG: %[[COND1:.+]] = arith.cmpi sgt, %[[DIM]], %[[C1]] : index
//// CHECK-DAG: %[[MASK:.+]] = vector.broadcast %[[COND]] : i1 to vector<16xi1>
//// CHECK-DAG: %[[MASK1:.+]] = vector.broadcast %[[COND1]] : i1 to vector<16xi1>
//// CHECK: vector.maskedload %{{.*}}[%[[C0]], %[[C0]], %[[C0]]], %[[MASK]]
//// CHECK: vector.maskedload %{{.*}}[%[[C0]], %[[C1]], %[[C0]]], %[[MASK1]]
//
//// -----
//
//// Test to check if masked transfer reads masked on an outer dimension lowers
//// to:
////  cond = dim >= id
////  vector.maskedload mem[..., id, ...] broadcast(cond)
//func.func @partial_masked_transfer_read(%mem : memref<16x?x32xf16>) -> vector<1x2x16xf16> {
//  %cst = ub.poison : f16
//  %c0 = arith.constant 0 : index
//  %c1 = arith.constant 1 : index
//  %c16 = arith.constant 16 : index
//  %dim = memref.dim %mem, %c1 : memref<16x?x32xf16>
//  %mask = vector.create_mask %c1, %dim, %c16 : vector<1x2x16xi1>
//  %read = vector.transfer_read %mem[%c0, %c0, %c0], %cst, %mask {in_bounds = [true, true, true]} : memref<16x?x32xf16>, vector<1x2x16xf16>
//  return %read : vector<1x2x16xf16>
//}
//
//// CHECK-LABEL: func.func @partial_masked_transfer_read
//// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
//// CHECK-DAG: %[[DIM:.+]] = memref.dim
//// CHECK-DAG: %[[COND:.+]] = arith.cmpi sgt, %[[DIM]], %[[C0]] : index
//// CHECK-DAG: %[[COND1:.+]] = arith.cmpi sgt, %[[DIM]], %[[C1]] : index
//// CHECK-DAG: %[[MASK:.+]] = vector.broadcast %[[COND]] : i1 to vector<16xi1>
//// CHECK-DAG: %[[MASK1:.+]] = vector.broadcast %[[COND1]] : i1 to vector<16xi1>
//// CHECK: vector.maskedload %{{.*}}[%[[C0]], %[[C0]], %[[C0]]], %[[MASK]]
//// CHECK: vector.maskedload %{{.*}}[%[[C0]], %[[C1]], %[[C0]]], %[[MASK1]]
//
//// -----
//
//// Test multi_reduction lowering.
//// TODO(#21483): Detect reduction(fma(a, b, 0.0)) and fold it into a series of FMAs.
//
//func.func @multi_reduction_f32(%a: vector<2x1x8xf32>, %b: vector<2x1x8xf32>) -> vector<2x1xf32> {
//  %cst_4 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
//  %cst_5 = arith.constant dense<0.000000e+00> : vector<2x1x8xf32>
//  %22 = arith.mulf %a, %b : vector<2x1x8xf32>
//  %23 = arith.addf %22, %cst_5 : vector<2x1x8xf32>
//  %24 = vector.multi_reduction <add>, %23, %cst_4 [2] : vector<2x1x8xf32> to vector<2x1xf32>
//  return %24 : vector<2x1xf32>
//}
//
//// CHECK-LABEL: func.func @multi_reduction_f32
//// CHECK-DAG:  %[[C0:.+]]  = arith.constant 0.000000e+00 : f32
//// CHECK-DAG:  %[[V0:.+]]  = arith.constant dense<0.000000e+00> : vector<2x1x8xf32>
//// CHECK:      %[[FMA:.+]] = math.fma %{{.*}}, %{{.*}}, %[[V0]] fastmath<contract> : vector<2x1x8xf32>
//// CHECK:      %[[E0:.+]]  = vector.extract %[[FMA]][0, 0] : vector<8xf32> from vector<2x1x8xf32>
//// CHECK:                    vector.reduction <add>, %[[E0]], %[[C0]] : vector<8xf32> into f32
//// CHECK:      %[[E1:.+]]  = vector.extract %[[FMA]][1, 0] : vector<8xf32> from vector<2x1x8xf32>
//// CHECK:                    vector.reduction <add>, %[[E1]], %[[C0]] : vector<8xf32> into f32
//
//func.func @multi_reduction_no_uplift(%a: vector<2x1x8xf32>, %b: vector<2x1x8xf32>) -> vector<2x1xf32> {
//  %cst_4 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
//  %cst_5 = arith.constant dense<0.000000e+00> : vector<2x1x8xf32>
//  %22 = arith.mulf %a, %b fastmath<fast>: vector<2x1x8xf32>
//  %23 = arith.addf %22, %cst_5 : vector<2x1x8xf32>
//  %24 = vector.multi_reduction <add>, %23, %cst_4 [2] : vector<2x1x8xf32> to vector<2x1xf32>
//  return %24 : vector<2x1xf32>
//}
//
//// CHECK-LABEL: func.func @multi_reduction_no_uplift
//// CHECK-DAG:  %[[C0:.+]]  = arith.constant 0.000000e+00 : f32
//// CHECK-DAG:  %[[V0:.+]]  = arith.constant dense<0.000000e+00> : vector<2x1x8xf32>
//// CHECK:      %[[MUL:.+]] = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : vector<2x1x8xf32>
//// CHECK:      %[[ADD:.+]] = arith.addf %[[MUL]], %[[V0]] : vector<2x1x8xf32>
//// CHECK:      %[[E0:.+]]  = vector.extract %[[ADD]][0, 0] : vector<8xf32> from vector<2x1x8xf32>
//// CHECK:                    vector.reduction <add>, %[[E0]], %[[C0]] : vector<8xf32> into f32
//// CHECK:      %[[E1:.+]]  = vector.extract %[[ADD]][1, 0] : vector<8xf32> from vector<2x1x8xf32>
//// CHECK:                    vector.reduction <add>, %[[E1]], %[[C0]] : vector<8xf32> into f32
//

// func.func @multi_reduction_f32(%a: vector<4x1x8xf32>, %b: vector<4x1x8xf32>) -> vector<4x1xf32> {
//   %cst_4 = arith.constant dense<0.000000e+00> : vector<4x1xf32>
//   %cst_5 = arith.constant dense<0.000000e+00> : vector<4x1x8xf32>
//   %23 = math.fma %a, %b, %cst_5 fastmath<contract> : vector<4x1x8xf32>
//   %24 = vector.multi_reduction <add>, %23, %cst_4 [2] : vector<4x1x8xf32> to vector<4x1xf32>
//   return %24 : vector<4x1xf32>
// }


// func.func @multi_reduction_f32(%a: vector<4x1x8xf32>, %b: vector<4x1x8xf32>) -> vector<4x1xf32> {
//   %cst_4 = arith.constant dense<0.000000e+00> : vector<4x1xf32>
//   %cst_5 = arith.constant dense<0.000000e+00> : vector<4x1x8xf32>
//   %23 = math.fma %a, %b, %cst_5 fastmath<contract> : vector<4x1x8xf32>
//   %24 = vector.multi_reduction <add>, %23, %cst_4 [2] : vector<4x1x8xf32> to vector<4x1xf32>
//   return %24 : vector<4x1xf32>
// }

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @multi_reduction_f32(%arg0: vector<2x1x8xf32>, %arg1: vector<2x1x8xf32>) -> vector<2x1xf32> {
    %cst = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %0 = vector.contract {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %arg0, %arg1, %cst : vector<2x1x8xf32>, vector<2x1x8xf32> into vector<2x1xf32>
    return %0 : vector<2x1xf32>
  }
}

// #lhs = affine_map<(d0, d1, d2) -> (d1, d2, d0)>  // LHS axes = [d1, d2, d0]
// #rhs = affine_map<(d0, d1, d2) -> (d0, d1, d2)>  // RHS axes = [d0, d1, d2] (identity)
// #acc = affine_map<(d0, d1, d2) -> (d0, d1)>

// module {
//   func.func @case_loopToAxis_ne_i(%A: vector<1x4x2xf32>, %B: vector<2x1x4xf32>)
//       -> vector<2x1xf32> {
//     %c = arith.constant dense<0.0> : vector<2x1xf32>
//     %0 = vector.contract
//           { indexing_maps = [#lhs, #rhs, #acc],
//             iterator_types = ["parallel", "parallel", "reduction"],
//             kind = #vector.kind<add> }
//           %A, %B, %c
//           : vector<1x4x2xf32>, vector<2x1x4xf32> into vector<2x1xf32>
//     return %0 : vector<2x1xf32>
//   }
// }

// #lhs = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
// #rhs = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
// #res = affine_map<(d0, d1, d2) -> (d0, d1)>

// func.func @both_non_identity_both_mismatch(
//     %A: vector<3x4x2xf32>,
//     %B: vector<4x3x2xf32>) -> vector<3x2xf32> {
//   %c0 = arith.constant dense<0.000000e+00> : vector<3x2xf32>
//   %out = vector.contract
//            { indexing_maps = [#lhs, #rhs, #res],
//              iterator_types = ["parallel","parallel","reduction"],
//              kind = #vector.kind<add> }
//            %A, %B, %c0
//          : vector<3x4x2xf32>, vector<4x3x2xf32> into vector<3x2xf32>
//   return %out : vector<3x2xf32>
// }


// func.func @foo(%a: vector<4x1x1x1x1x1x1x1x8xf32>, %b: vector<4x1x1x1x1x1x1x1x8xf32>, %c: vector<4x1x1x1x1x1xf32>) -> vector<4x1x1x1x1x1xf32> {
//   // %cst_4 = arith.constant dense<1.000000e+00> : vector<4x1xf32>
//   %1 = arith.mulf %a, %b : vector<4x1x1x1x1x1x1x1x8xf32>
//   %24 = vector.multi_reduction <add>, %1, %c [2, 5, 8] : vector<4x2x1x1x1x1x1x1x8xf32> to vector<4x2x1x1x1x1xf32>
//   return %24 : vector<4x2x1x1x1x1xf32>
// }

// #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5, d6, d7, d8)>
// #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d3, d4, d6, d7)>
// module {
//   func.func @foo(%arg0: vector<4x2x1x1x1x1x1x1x8xf32>, %arg1: vector<4x2x1x1x1x1x1x1x8xf32>, %arg2: vector<4x2x1x1x1x1xf32>) -> vector<4x2x1x1x1x1xf32> {
//     %0 = vector.contract {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<4x2x1x1x1x1x1x1x8xf32>, vector<4x2x1x1x1x1x1x1x8xf32> into vector<4x2x1x1x1x1xf32>
//     return %0 : vector<4x2x1x1x1x1xf32>
//   }
// }