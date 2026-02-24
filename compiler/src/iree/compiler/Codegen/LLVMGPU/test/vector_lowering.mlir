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
//  CHECK: %[[INSERT:.+]] = vector.broadcast %[[ELEM]] : f16 to vector<1x8xf16>
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
// CHECK: %[[LHS_BROADCAST:.+]] = vector.broadcast %[[LHS_CAST]] : f32 to vector<2xf32>
// CHECK: %[[FMA:.+]] = vector.fma %[[RHS_CAST]], %[[LHS_BROADCAST]], %[[ACC]] : vector<2xf32>
// CHECK: arith.select %[[MASK_EXTRACT]], %[[FMA]], %[[ACC]] : vector<2xi1>, vector<2xf32>

// -----

// Test to check if masked transfer reads masked on an outer dimension lowers
// to:
//  cond = dim >= id
//  vector.maskedload mem[..., id, ...] broadcast(cond)
func.func @partial_masked_transfer_read(%mem : memref<16x?x32xf16>) -> vector<1x2x16xf16> {
  %cst = ub.poison : f16
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %dim = memref.dim %mem, %c1 : memref<16x?x32xf16>
  %mask = vector.create_mask %c1, %dim, %c16 : vector<1x2x16xi1>
  %read = vector.transfer_read %mem[%c0, %c0, %c0], %cst, %mask {in_bounds = [true, true, true]} : memref<16x?x32xf16>, vector<1x2x16xf16>
  return %read : vector<1x2x16xf16>
}

// CHECK-LABEL: func.func @partial_masked_transfer_read
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[DIM:.+]] = memref.dim
// CHECK-DAG: %[[COND:.+]] = arith.cmpi sgt, %[[DIM]], %[[C0]] : index
// CHECK-DAG: %[[COND1:.+]] = arith.cmpi sgt, %[[DIM]], %[[C1]] : index
// CHECK-DAG: %[[MASK:.+]] = vector.broadcast %[[COND]] : i1 to vector<16xi1>
// CHECK-DAG: %[[MASK1:.+]] = vector.broadcast %[[COND1]] : i1 to vector<16xi1>
// CHECK: vector.maskedload %{{.*}}[%[[C0]], %[[C0]], %[[C0]]], %[[MASK]]
// CHECK: vector.maskedload %{{.*}}[%[[C0]], %[[C1]], %[[C0]]], %[[MASK1]]

// -----

// Test multi_reduction lowering.

func.func @multi_reduction_f32(%a: vector<2x1x8xf32>, %b: vector<2x1x8xf32>) -> vector<2x1xf32> {
  %cst_4 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
  %cst_5 = arith.constant dense<0.000000e+00> : vector<2x1x8xf32>
  %22 = arith.mulf %a, %b : vector<2x1x8xf32>
  %23 = arith.addf %22, %cst_5 : vector<2x1x8xf32>
  %24 = vector.multi_reduction <add>, %23, %cst_4 [2] : vector<2x1x8xf32> to vector<2x1xf32>
  return %24 : vector<2x1xf32>
}

// CHECK-LABEL: func.func @multi_reduction_f32
// CHECK-SAME: %[[ARG0:.+]]: vector<2x1x8xf32>, %[[ARG1:.+]]: vector<2x1x8xf32>)
// CHECK-DAG: %[[LHS0:.+]] = vector.extract %[[ARG0]][0, 0]
// CHECK-DAG: %[[RHS0:.+]] = vector.extract %[[ARG1]][0, 0]
// CHECK-DAG: %[[LHS1:.+]] = vector.extract %[[ARG0]][1, 0]
// CHECK-DAG: %[[RHS1:.+]] = vector.extract %[[ARG1]][1, 0]
// CHECK-DAG: %[[FMA1:.+]] = math.fma %[[LHS0]], %[[RHS0]], %{{.*}} fastmath<contract> : vector<8xf32>
// CHECK-DAG: %[[FMA2:.+]] = math.fma %[[LHS1]], %[[RHS1]], %{{.*}} fastmath<contract> : vector<8xf32>
// CHECK-DAG: %[[RED1:.+]] = vector.reduction <add>, %[[FMA1]], %{{.*}} : vector<8xf32> into f32
// CHECK-DAG: %[[RED2:.+]] = vector.reduction <add>, %[[FMA2]], %{{.*}} : vector<8xf32> into f32
// CHECK: vector.from_elements %[[RED1]], %[[RED2]] : vector<2x1xf32>

// -----


func.func @multi_reduction_no_uplift(%a: vector<2x1x8xf32>, %b: vector<2x1x8xf32>) -> vector<2x1xf32> {
  %cst_4 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
  %cst_5 = arith.constant dense<0.000000e+00> : vector<2x1x8xf32>
  %22 = arith.mulf %a, %b fastmath<fast>: vector<2x1x8xf32>
  %23 = arith.addf %22, %cst_5 : vector<2x1x8xf32>
  %24 = vector.multi_reduction <add>, %23, %cst_4 [2] : vector<2x1x8xf32> to vector<2x1xf32>
  return %24 : vector<2x1xf32>
}

// CHECK-LABEL: func.func @multi_reduction_no_uplift
// CHECK-SAME: %[[ARG0:.+]]: vector<2x1x8xf32>, %[[ARG1:.+]]: vector<2x1x8xf32>)
// CHECK-DAG: %[[LHS0:.+]] = vector.extract %[[ARG0]][0, 0]
// CHECK-DAG: %[[RHS0:.+]] = vector.extract %[[ARG1]][0, 0]
// CHECK-DAG: %[[LHS1:.+]] = vector.extract %[[ARG0]][1, 0]
// CHECK-DAG: %[[RHS1:.+]] = vector.extract %[[ARG1]][1, 0]
// CHECK-DAG: arith.mulf %[[LHS0]], %[[RHS0]] fastmath<fast> : vector<8xf32>
// CHECK-DAG: arith.mulf %[[LHS1]], %[[RHS1]] fastmath<fast> : vector<8xf32>
// CHECK-NOT: math.fma

// -----

func.func @multi_reduction_f32_to_single_chain_fma(%a: vector<2x1x8xf32>, %b: vector<2x1x8xf32>) -> vector<2x1xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<2x1xf32>
  %0 = arith.mulf %a, %b : vector<2x1x8xf32>
  %1 = vector.multi_reduction <add>, %0, %cst [2] : vector<2x1x8xf32> to vector<2x1xf32>
  return %1 : vector<2x1xf32>
}

// CHECK-LABEL: func.func @multi_reduction_f32_to_single_chain_fma
// CHECK:  %[[C0:.+]]  = arith.constant dense<0.000000e+00> : vector<2xf32>
// CHECK:  %[[FMA0:.*]] = math.fma %{{.*}}, %{{.*}}, %[[C0]] : vector<2xf32>
// CHECK:  %[[FMA1:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA0]] : vector<2xf32>
// CHECK:  %[[FMA2:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA1]] : vector<2xf32>
// CHECK:  %[[FMA3:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA2]] : vector<2xf32>
// CHECK:  %[[FMA4:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA3]] : vector<2xf32>
// CHECK:  %[[FMA5:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA4]] : vector<2xf32>
// CHECK:  %[[FMA6:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA5]] : vector<2xf32>
// CHECK:  %[[FMA7:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA6]] : vector<2xf32>
// CHECK:  return %{{.*}} : vector<2x1xf32>

// -----

func.func @multi_reduction_f16_to_single_chain_fma(%a: vector<2x1x8xf16>, %b: vector<2x1x8xf16>) -> vector<2x1xf16> {
  %cst = arith.constant dense<0.000000e+00> : vector<2x1xf16>
  %0 = arith.mulf %a, %b : vector<2x1x8xf16>
  %1 = vector.multi_reduction <add>, %0, %cst [2] : vector<2x1x8xf16> to vector<2x1xf16>
  return %1 : vector<2x1xf16>
}

// CHECK-LABEL: func.func @multi_reduction_f16_to_single_chain_fma
// CHECK:  %[[C0:.+]]  = arith.constant dense<0.000000e+00> : vector<2xf16>
// CHECK:  %[[FMA0:.*]] = math.fma %{{.*}}, %{{.*}}, %[[C0]] : vector<2xf16>
// CHECK:  %[[FMA1:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA0]] : vector<2xf16>
// CHECK:  %[[FMA2:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA1]] : vector<2xf16>
// CHECK:  %[[FMA3:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA2]] : vector<2xf16>
// CHECK:  %[[FMA4:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA3]] : vector<2xf16>
// CHECK:  %[[FMA5:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA4]] : vector<2xf16>
// CHECK:  %[[FMA6:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA5]] : vector<2xf16>
// CHECK:  %[[FMA7:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA6]] : vector<2xf16>
// CHECK:  return %{{.*}} : vector<2x1xf16>

// -----

func.func @multi_reduction_f32_to_double_chain_fma(%a: vector<4x1x8xf32>, %b: vector<4x1x8xf32>) -> vector<4x1xf32> {
 %cst = arith.constant dense<0.000000e+00> : vector<4x1xf32>
 %0 = arith.mulf %a, %b : vector<4x1x8xf32>
 %1 = vector.multi_reduction <add>, %0, %cst [2] : vector<4x1x8xf32> to vector<4x1xf32>
 return %1 : vector<4x1xf32>
}

// CHECK-LABEL: func.func @multi_reduction_f32_to_double_chain_fma
// CHECK:  %[[C0:.+]]  = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK:  %[[FMA0:.*]] = math.fma %{{.*}}, %{{.*}}, %[[C0]] : vector<4xf32>
// CHECK:  %[[FMA1:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA0]] : vector<4xf32>
// CHECK:  %[[FMA2:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA1]] : vector<4xf32>
// CHECK:  %[[FMA3:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA2]] : vector<4xf32>
// CHECK:  %[[FMA4:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA3]] : vector<4xf32>
// CHECK:  %[[FMA5:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA4]] : vector<4xf32>
// CHECK:  %[[FMA6:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA5]] : vector<4xf32>
// CHECK:  %[[FMA7:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA6]] : vector<4xf32>
// CHECK:  return %{{.*}} : vector<4x1xf32>

// -----

#lhs = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#rhs = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#res = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @general_contract_add_to_chain_fma(
    %A: vector<3x4x2xf32>,
    %B: vector<4x3x2xf32>) -> vector<3x2xf32> {
  %c0 = arith.constant dense<0.000000e+00> : vector<3x2xf32>
  %out = vector.contract
           { indexing_maps = [#lhs, #rhs, #res],
             iterator_types = ["parallel","parallel","reduction"],
             kind = #vector.kind<add> }
           %A, %B, %c0
         : vector<3x4x2xf32>, vector<4x3x2xf32> into vector<3x2xf32>
  return %out : vector<3x2xf32>
}

// CHECK-LABEL: func.func @general_contract_add_to_chain_fma
// CHECK:  %[[C0:.+]] = arith.constant dense<0.000000e+00> : vector<6xf32>
// CHECK:  %[[FMA0:.*]] = math.fma {{.*}}, {{.*}}, %[[C0]] : vector<6xf32>
// CHECK:  %[[FMA1:.*]] = math.fma {{.*}}, {{.*}}, %[[FMA0]] : vector<6xf32>
// CHECK:  %[[FMA2:.*]] = math.fma {{.*}}, {{.*}}, %[[FMA1]] : vector<6xf32>
// CHECK:  return %{{.*}} : vector<3x2xf32>

// -----

// Only float-point types should be lowered to fmas.
#lhs = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#rhs = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#res = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @int_contract_add_no_chain_fma(
    %A: vector<3x4x2xi32>,
    %B: vector<4x3x2xi32>) -> vector<3x2xi32> {
  %c0 = arith.constant dense<0> : vector<3x2xi32>
  %out = vector.contract
           { indexing_maps = [#lhs, #rhs, #res],
             iterator_types = ["parallel","parallel","reduction"],
             kind = #vector.kind<add> }
           %A, %B, %c0
         : vector<3x4x2xi32>, vector<4x3x2xi32> into vector<3x2xi32>
  return %out : vector<3x2xi32>
}

// CHECK-LABEL: func.func @int_contract_add_no_chain_fma
// CHECK-NOT:  math.fma
// CHECK:  return %{{.*}} : vector<3x2xi32>

// -----

// Test unrolling of a 2D transfer_gather representing an embedding lookup:
// outer dim is gathered (indices), inner dim is contiguous.

func.func @transfer_gather_unroll_embedding_lookup(
  %source: memref<4096x64xf16>,
  %indices: vector<4xindex>) -> vector<4x64xf16> {
  %cst = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [%indices : vector<4xindex>], %cst {
    indexing_maps = [affine_map<(d0, d1)[s0] -> (s0, d1)>,
                     affine_map<(d0, d1)[s0] -> (d0)>]
  } : memref<4096x64xf16>, vector<4x64xf16>
  return %out : vector<4x64xf16>
}

// After unrolling + canonicalization, the 2D gather becomes 4 contiguous loads.
// CHECK-LABEL: func.func @transfer_gather_unroll_embedding_lookup
// CHECK-NOT: transfer_gather
// CHECK-COUNT-4: vector.load
// CHECK-NOT: transfer_gather

// -----

// Test unrolling of a masked 2D transfer_gather.
// Same embedding lookup shape but with a mask on the result.

func.func @transfer_gather_unroll_masked(
  %source: memref<4096x64xf16>,
  %indices: vector<4xindex>,
  %mask: vector<4x64xi1>) -> vector<4x64xf16> {
  %cst = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [%indices : vector<4xindex>], %cst, %mask {
    indexing_maps = [affine_map<(d0, d1)[s0] -> (s0, d1)>,
                     affine_map<(d0, d1)[s0] -> (d0)>,
                     affine_map<(d0, d1)[s0] -> (d0, d1)>]
  } : memref<4096x64xf16>, vector<4x64xf16>, vector<4x64xi1>
  return %out : vector<4x64xf16>
}

// After unrolling, mask slices are passed to each sub-gather.
// The masked rank-1 gathers lower to vector.maskedload ops.
// CHECK-LABEL: func.func @transfer_gather_unroll_masked
// CHECK-NOT: transfer_gather
// CHECK-COUNT-4: vector.maskedload
// CHECK-NOT: transfer_gather

// -----

// Test unrolling of a 3D transfer_gather with a transposed 2D index vector.
// The first two output dims (d0=4, d1=8) are both gathered via a single
// index vec of shape 8x4 (note: d1 before d0, i.e. "transposed").
// The inner dim (d2=64) is contiguous.

func.func @transfer_gather_unroll_transposed_index(
  %source: memref<4096x64xf16>,
  %indices: vector<8x4xindex>) -> vector<4x8x64xf16> {
  %cst = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [%indices : vector<8x4xindex>], %cst {
    indexing_maps = [affine_map<(d0, d1, d2)[s0] -> (s0, d2)>,
                     affine_map<(d0, d1, d2)[s0] -> (d1, d0)>]
  } : memref<4096x64xf16>, vector<4x8x64xf16>
  return %out : vector<4x8x64xf16>
}

// After two rounds of unrolling (d0=4 then d1=8) + canonicalization,
// the 3D gather becomes 4*8=32 contiguous loads.
// CHECK-LABEL: func.func @transfer_gather_unroll_transposed_index
// CHECK-NOT: transfer_gather
// CHECK-COUNT-32: vector.load
// CHECK-NOT: transfer_gather
