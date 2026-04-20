// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmgpu-legalize-nd-vectors),util.func(iree-llvmgpu-legalize-nd-vectors))" \
// RUN:   --split-input-file %s | FileCheck %s

func.func @addf_2d(%arg0: vector<2x4xf32>, %arg1: vector<2x4xf32>) -> vector<2x4xf32> {
  %0 = arith.addf %arg0, %arg1 : vector<2x4xf32>
  return %0 : vector<2x4xf32>
}
// CHECK-LABEL: func.func @addf_2d
//  CHECK-SAME:   (%[[A0:.+]]: vector<4xf32>, %[[A1:.+]]: vector<4xf32>, %[[B0:.+]]: vector<4xf32>, %[[B1:.+]]: vector<4xf32>)
//  CHECK-SAME:   -> (vector<4xf32>, vector<4xf32>)
//       CHECK:   %[[R0:.+]] = arith.addf %[[A0]], %[[B0]] : vector<4xf32>
//       CHECK:   %[[R1:.+]] = arith.addf %[[A1]], %[[B1]] : vector<4xf32>
//       CHECK:   return %[[R0]], %[[R1]] : vector<4xf32>, vector<4xf32>

// -----

func.func @negf_3d(%arg0: vector<2x3x4xf32>) -> vector<2x3x4xf32> {
  %0 = arith.negf %arg0 : vector<2x3x4xf32>
  return %0 : vector<2x3x4xf32>
}
// CHECK-LABEL: func.func @negf_3d
// CHECK-COUNT-6: arith.negf %{{.*}} : vector<4xf32>
//       CHECK:   return {{.*}} : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>

// -----

func.func @constant_splat() -> vector<2x4xf32> {
  %0 = arith.constant dense<1.0> : vector<2x4xf32>
  return %0 : vector<2x4xf32>
}
// CHECK-LABEL: func.func @constant_splat
//       CHECK:   %[[C0:.+]] = arith.constant dense<1.000000e+00> : vector<4xf32>
//       CHECK:   %[[C1:.+]] = arith.constant dense<1.000000e+00> : vector<4xf32>
//       CHECK:   return %[[C0]], %[[C1]] : vector<4xf32>, vector<4xf32>

// -----

func.func @constant_nonsplat() -> vector<2x3xf32> {
  %0 = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : vector<2x3xf32>
  return %0 : vector<2x3xf32>
}
// CHECK-LABEL: func.func @constant_nonsplat
//       CHECK:   %[[C0:.+]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : vector<3xf32>
//       CHECK:   %[[C1:.+]] = arith.constant dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : vector<3xf32>
//       CHECK:   return %[[C0]], %[[C1]] : vector<3xf32>, vector<3xf32>

// -----

// Extracting one inner 1-D slice from a 2-D vector selects the right flat vector.
func.func @extract_inner_slice(%v: vector<3x4xf32>) -> vector<4xf32> {
  %0 = vector.extract %v[1] : vector<4xf32> from vector<3x4xf32>
  return %0 : vector<4xf32>
}
// CHECK-LABEL: func.func @extract_inner_slice
//  CHECK-SAME:   (%[[V0:.+]]: vector<4xf32>, %[[V1:.+]]: vector<4xf32>, %[[V2:.+]]: vector<4xf32>)
//       CHECK:   return %[[V1]] : vector<4xf32>

// -----

// Extracting a scalar from a 2-D vector: select the 1-D vector, then extract the element.
func.func @extract_scalar(%v: vector<2x4xf32>) -> f32 {
  %0 = vector.extract %v[1, 2] : f32 from vector<2x4xf32>
  return %0 : f32
}
// CHECK-LABEL: func.func @extract_scalar
//  CHECK-SAME:   (%[[V0:.+]]: vector<4xf32>, %[[V1:.+]]: vector<4xf32>)
//       CHECK:   %[[S:.+]] = vector.extract %[[V1]][2] : f32 from vector<4xf32>
//       CHECK:   return %[[S]] : f32

// -----

// Extracting a 2-D sub-vector from a 3-D vector selects the first group of flat vectors.
func.func @extract_3d(%v: vector<2x3x4xf32>) -> vector<3x4xf32> {
  %0 = vector.extract %v[0] : vector<3x4xf32> from vector<2x3x4xf32>
  return %0 : vector<3x4xf32>
}
// CHECK-LABEL: func.func @extract_3d
//  CHECK-SAME:   (%[[V0:.+]]: vector<4xf32>, %[[V1:.+]]: vector<4xf32>, %[[V2:.+]]: vector<4xf32>, %[[V3:.+]]: vector<4xf32>, %[[V4:.+]]: vector<4xf32>, %[[V5:.+]]: vector<4xf32>)
//  CHECK-SAME:   -> (vector<4xf32>, vector<4xf32>, vector<4xf32>)
//       CHECK:   return %[[V0]], %[[V1]], %[[V2]] : vector<4xf32>, vector<4xf32>, vector<4xf32>

// -----

// Inserting a 1-D vector into a 2-D vector replaces the corresponding flat vector.
func.func @insert_1d(%src: vector<4xf32>, %dst: vector<3x4xf32>) -> vector<3x4xf32> {
  %0 = vector.insert %src, %dst[2] : vector<4xf32> into vector<3x4xf32>
  return %0 : vector<3x4xf32>
}
// CHECK-LABEL: func.func @insert_1d
//  CHECK-SAME:   (%[[SRC:.+]]: vector<4xf32>, %[[D0:.+]]: vector<4xf32>, %[[D1:.+]]: vector<4xf32>, %[[D2:.+]]: vector<4xf32>)
//       CHECK:   return %[[D0]], %[[D1]], %[[SRC]] : vector<4xf32>, vector<4xf32>, vector<4xf32>

// -----

// Inserting a scalar into a 2-D vector: insert into the selected 1-D vector at the inner index.
func.func @insert_scalar(%val: f32, %dst: vector<2x4xf32>) -> vector<2x4xf32> {
  %0 = vector.insert %val, %dst[0, 3] : f32 into vector<2x4xf32>
  return %0 : vector<2x4xf32>
}
// CHECK-LABEL: func.func @insert_scalar
//  CHECK-SAME:   (%[[VAL:.+]]: f32, %[[D0:.+]]: vector<4xf32>, %[[D1:.+]]: vector<4xf32>)
//       CHECK:   %[[NEW:.+]] = vector.insert %[[VAL]], %[[D0]] [3] : f32 into vector<4xf32>
//       CHECK:   return %[[NEW]], %[[D1]] : vector<4xf32>, vector<4xf32>

// -----

// Transpose that only permutes outer dims: just reorders the flat 1-D vectors.
// vector<2x3x4xf32> with perm [1,0,2] -> vector<3x2x4xf32>
// Source layout (row-major): [0]=(0,0), [1]=(0,1), [2]=(0,2), [3]=(1,0), [4]=(1,1), [5]=(1,2)
// Result layout:             [0]=(0,0), [1]=(0,1), [2]=(1,0), [3]=(1,1), [4]=(2,0), [5]=(2,1)
// So result order is: src[0], src[3], src[1], src[4], src[2], src[5]
func.func @transpose_outer_only(%v: vector<2x3x4xf32>) -> vector<3x2x4xf32> {
  %0 = vector.transpose %v, [1, 0, 2] : vector<2x3x4xf32> to vector<3x2x4xf32>
  return %0 : vector<3x2x4xf32>
}
// CHECK-LABEL: func.func @transpose_outer_only
//  CHECK-SAME:   (%[[V0:.+]]: vector<4xf32>, %[[V1:.+]]: vector<4xf32>, %[[V2:.+]]: vector<4xf32>, %[[V3:.+]]: vector<4xf32>, %[[V4:.+]]: vector<4xf32>, %[[V5:.+]]: vector<4xf32>)
//       CHECK:   return %[[V0]], %[[V3]], %[[V1]], %[[V4]], %[[V2]], %[[V5]]

// -----

// Transpose that permutes the inner dim: decomposes to scalars and reassembles.
// vector<2x3xf32> with perm [1,0] -> vector<3x2xf32>
func.func @transpose_inner(%v: vector<2x3xf32>) -> vector<3x2xf32> {
  %0 = vector.transpose %v, [1, 0] : vector<2x3xf32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}
// CHECK-LABEL: func.func @transpose_inner
//  CHECK-SAME:   (%[[V0:.+]]: vector<3xf32>, %[[V1:.+]]: vector<3xf32>)
//       CHECK:   %[[S0:.+]]:3 = vector.to_elements %[[V0]] : vector<3xf32>
//       CHECK:   %[[S1:.+]]:3 = vector.to_elements %[[V1]] : vector<3xf32>
//       CHECK:   %[[R0:.+]] = vector.from_elements %[[S0]]#0, %[[S1]]#0 : vector<2xf32>
//       CHECK:   %[[R1:.+]] = vector.from_elements %[[S0]]#1, %[[S1]]#1 : vector<2xf32>
//       CHECK:   %[[R2:.+]] = vector.from_elements %[[S0]]#2, %[[S1]]#2 : vector<2xf32>
//       CHECK:   return %[[R0]], %[[R1]], %[[R2]] : vector<2xf32>, vector<2xf32>, vector<2xf32>

// -----

// shape_cast with same inner dim: passthrough of flat vectors.
func.func @shape_cast_same_inner(%v: vector<2x3x4xf32>) -> vector<6x4xf32> {
  %0 = vector.shape_cast %v : vector<2x3x4xf32> to vector<6x4xf32>
  return %0 : vector<6x4xf32>
}
// CHECK-LABEL: func.func @shape_cast_same_inner
//  CHECK-SAME:   (%[[V0:.+]]: vector<4xf32>, %[[V1:.+]]: vector<4xf32>, %[[V2:.+]]: vector<4xf32>, %[[V3:.+]]: vector<4xf32>, %[[V4:.+]]: vector<4xf32>, %[[V5:.+]]: vector<4xf32>)
//   CHECK-NOT:   vector.to_elements
//       CHECK:   return %[[V0]], %[[V1]], %[[V2]], %[[V3]], %[[V4]], %[[V5]]

// -----

// shape_cast with different inner dim: flatten to scalars and regroup.
func.func @shape_cast_diff_inner(%v: vector<2x6xf32>) -> vector<3x4xf32> {
  %0 = vector.shape_cast %v : vector<2x6xf32> to vector<3x4xf32>
  return %0 : vector<3x4xf32>
}
// CHECK-LABEL: func.func @shape_cast_diff_inner
//  CHECK-SAME:   (%[[V0:.+]]: vector<6xf32>, %[[V1:.+]]: vector<6xf32>)
//       CHECK:   %[[S0:.+]]:6 = vector.to_elements %[[V0]] : vector<6xf32>
//       CHECK:   %[[S1:.+]]:6 = vector.to_elements %[[V1]] : vector<6xf32>
//       CHECK:   %[[R0:.+]] = vector.from_elements %[[S0]]#0, %[[S0]]#1, %[[S0]]#2, %[[S0]]#3 : vector<4xf32>
//       CHECK:   %[[R1:.+]] = vector.from_elements %[[S0]]#4, %[[S0]]#5, %[[S1]]#0, %[[S1]]#1 : vector<4xf32>
//       CHECK:   %[[R2:.+]] = vector.from_elements %[[S1]]#2, %[[S1]]#3, %[[S1]]#4, %[[S1]]#5 : vector<4xf32>
//       CHECK:   return %[[R0]], %[[R1]], %[[R2]]

// -----

// shape_cast from 2-D to 1-D: flatten to scalars and reassemble into a single vector.
func.func @shape_cast_to_1d(%v: vector<2x4xf32>) -> vector<8xf32> {
  %0 = vector.shape_cast %v : vector<2x4xf32> to vector<8xf32>
  return %0 : vector<8xf32>
}
// CHECK-LABEL: func.func @shape_cast_to_1d
//  CHECK-SAME:   (%[[V0:.+]]: vector<4xf32>, %[[V1:.+]]: vector<4xf32>)
//       CHECK:   %[[S0:.+]]:4 = vector.to_elements %[[V0]] : vector<4xf32>
//       CHECK:   %[[S1:.+]]:4 = vector.to_elements %[[V1]] : vector<4xf32>
//       CHECK:   %[[R:.+]] = vector.from_elements %[[S0]]#0, %[[S0]]#1, %[[S0]]#2, %[[S0]]#3, %[[S1]]#0, %[[S1]]#1, %[[S1]]#2, %[[S1]]#3 : vector<8xf32>
//       CHECK:   return %[[R]] : vector<8xf32>

// -----

// Broadcast a scalar to a 2-D vector: broadcast to 1-D, reuse for both results.
func.func @broadcast_scalar(%s: f32) -> vector<2x4xf32> {
  %0 = vector.broadcast %s : f32 to vector<2x4xf32>
  return %0 : vector<2x4xf32>
}
// CHECK-LABEL: func.func @broadcast_scalar
//  CHECK-SAME:   (%[[S:.+]]: f32)
//       CHECK:   %[[B:.+]] = vector.broadcast %[[S]] : f32 to vector<4xf32>
//       CHECK:   return %[[B]], %[[B]] : vector<4xf32>, vector<4xf32>

// -----

// Bitcast on a 2-D vector: each 1-D vector is bitcast individually.
func.func @bitcast_2d(%v: vector<2x4xf32>) -> vector<2x4xi32> {
  %0 = vector.bitcast %v : vector<2x4xf32> to vector<2x4xi32>
  return %0 : vector<2x4xi32>
}
// CHECK-LABEL: func.func @bitcast_2d
//  CHECK-SAME:   (%[[V0:.+]]: vector<4xf32>, %[[V1:.+]]: vector<4xf32>)
//       CHECK:   %[[R0:.+]] = vector.bitcast %[[V0]] : vector<4xf32> to vector<4xi32>
//       CHECK:   %[[R1:.+]] = vector.bitcast %[[V1]] : vector<4xf32> to vector<4xi32>
//       CHECK:   return %[[R0]], %[[R1]] : vector<4xi32>, vector<4xi32>

// -----

// ub.poison on a 2-D vector is split into multiple 1-D poisons.
func.func @poison_2d() -> vector<2x4xf32> {
  %0 = ub.poison : vector<2x4xf32>
  return %0 : vector<2x4xf32>
}
// CHECK-LABEL: func.func @poison_2d
//       CHECK:   %[[P0:.+]] = ub.poison : vector<4xf32>
//       CHECK:   %[[P1:.+]] = ub.poison : vector<4xf32>
//       CHECK:   return %[[P0]], %[[P1]] : vector<4xf32>, vector<4xf32>

// -----

// to_elements on a 2-D vector: decompose each 1-D vector separately.
func.func @to_elements_2d(%v: vector<2x3xf32>) -> (f32, f32, f32, f32, f32, f32) {
  %0:6 = vector.to_elements %v : vector<2x3xf32>
  return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 : f32, f32, f32, f32, f32, f32
}
// CHECK-LABEL: func.func @to_elements_2d
//  CHECK-SAME:   (%[[V0:.+]]: vector<3xf32>, %[[V1:.+]]: vector<3xf32>)
//       CHECK:   %[[S0:.+]]:3 = vector.to_elements %[[V0]] : vector<3xf32>
//       CHECK:   %[[S1:.+]]:3 = vector.to_elements %[[V1]] : vector<3xf32>
//       CHECK:   return %[[S0]]#0, %[[S0]]#1, %[[S0]]#2, %[[S1]]#0, %[[S1]]#1, %[[S1]]#2

// -----

// from_elements producing a 2-D vector: chunk scalars into 1-D from_elements.
func.func @from_elements_2d(%a: f32, %b: f32, %c: f32, %d: f32, %e: f32, %f: f32) -> vector<2x3xf32> {
  %0 = vector.from_elements %a, %b, %c, %d, %e, %f : vector<2x3xf32>
  return %0 : vector<2x3xf32>
}
// CHECK-LABEL: func.func @from_elements_2d
//  CHECK-SAME:   (%[[A:.+]]: f32, %[[B:.+]]: f32, %[[C:.+]]: f32, %[[D:.+]]: f32, %[[E:.+]]: f32, %[[F:.+]]: f32)
//       CHECK:   %[[R0:.+]] = vector.from_elements %[[A]], %[[B]], %[[C]] : vector<3xf32>
//       CHECK:   %[[R1:.+]] = vector.from_elements %[[D]], %[[E]], %[[F]] : vector<3xf32>
//       CHECK:   return %[[R0]], %[[R1]] : vector<3xf32>, vector<3xf32>

// -----

// scf.for with n-D vector iter_args: iter_args and yields become multiple 1-D vectors.
func.func @scf_for_nd(%init: vector<2x4xf32>, %arg: vector<2x4xf32>) -> vector<2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (vector<2x4xf32>) {
    %sum = arith.addf %acc, %arg : vector<2x4xf32>
    scf.yield %sum : vector<2x4xf32>
  }
  return %res : vector<2x4xf32>
}
// CHECK-LABEL: func.func @scf_for_nd
//  CHECK-SAME:   (%[[I0:.+]]: vector<4xf32>, %[[I1:.+]]: vector<4xf32>, %[[A0:.+]]: vector<4xf32>, %[[A1:.+]]: vector<4xf32>)
//       CHECK:   %[[FOR:.+]]:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC0:.+]] = %[[I0]], %[[ACC1:.+]] = %[[I1]]) -> (vector<4xf32>, vector<4xf32>)
//       CHECK:     %[[S0:.+]] = arith.addf %[[ACC0]], %[[A0]] : vector<4xf32>
//       CHECK:     %[[S1:.+]] = arith.addf %[[ACC1]], %[[A1]] : vector<4xf32>
//       CHECK:     scf.yield %[[S0]], %[[S1]] : vector<4xf32>, vector<4xf32>
//       CHECK:   return %[[FOR]]#0, %[[FOR]]#1 : vector<4xf32>, vector<4xf32>

// -----

// nvgpu.mma.sync is legal despite having n-D vectors; materializations bridge
// the 1-D converted values and the n-D op interface.
func.func @mma_sync_legal(%a: vector<4x2xf16>, %b: vector<2x2xf16>, %c: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = nvgpu.mma.sync(%a, %b, %c) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
  return %0 : vector<2x2xf32>
}
// CHECK-LABEL: func.func @mma_sync_legal
//       CHECK:   nvgpu.mma.sync({{.*}}) {mmaShape = [16, 8, 16]}
//  CHECK-SAME:     : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
//       CHECK:   return {{.*}} : vector<2xf32>, vector<2xf32>

// -----

func.func @delinearize_2d_vector_unroll(%vec: vector<2x2xindex>) -> (vector<2x2xindex>, vector<2x2xindex>) {
  %0:2 = affine.delinearize_index %vec into (4, 8) : vector<2x2xindex>, vector<2x2xindex>
  return %0#0, %0#1 : vector<2x2xindex>, vector<2x2xindex>
}
// CHECK-LABEL: func.func @delinearize_2d_vector_unroll
//  CHECK-SAME:   (%[[V0:.+]]: vector<2xindex>, %[[V1:.+]]: vector<2xindex>)
//  CHECK-SAME:   -> (vector<2xindex>, vector<2xindex>, vector<2xindex>, vector<2xindex>)
//       CHECK:   %[[R0:.+]]:2 = affine.delinearize_index %[[V0]] into (4, 8) : vector<2xindex>, vector<2xindex>
//       CHECK:   %[[R1:.+]]:2 = affine.delinearize_index %[[V1]] into (4, 8) : vector<2xindex>, vector<2xindex>
//       CHECK:   return %[[R0]]#0, %[[R1]]#0, %[[R0]]#1, %[[R1]]#1 : vector<2xindex>, vector<2xindex>, vector<2xindex>, vector<2xindex>

// -----

func.func @linearize_2d_vector_unroll(%v0: vector<2x2xindex>, %v1: vector<2x2xindex>) -> vector<2x2xindex> {
  %0 = affine.linearize_index [%v0, %v1] by (4, 8) : vector<2x2xindex>
  return %0 : vector<2x2xindex>
}
// CHECK-LABEL: func.func @linearize_2d_vector_unroll
//  CHECK-SAME:   (%[[V0_0:.+]]: vector<2xindex>, %[[V0_1:.+]]: vector<2xindex>, %[[V1_0:.+]]: vector<2xindex>, %[[V1_1:.+]]: vector<2xindex>)
//  CHECK-SAME:   -> (vector<2xindex>, vector<2xindex>)
//       CHECK:   %[[R0:.+]] = affine.linearize_index [%[[V0_0]], %[[V1_0]]] by (4, 8) : vector<2xindex>
//       CHECK:   %[[R1:.+]] = affine.linearize_index [%[[V0_1]], %[[V1_1]]] by (4, 8) : vector<2xindex>
//       CHECK:   return %[[R0]], %[[R1]] : vector<2xindex>, vector<2xindex>

// -----

util.func @util_func_addf_2d(%arg0: vector<2x4xf32>, %arg1: vector<2x4xf32>) -> vector<2x4xf32> {
  %0 = arith.addf %arg0, %arg1 : vector<2x4xf32>
  util.return %0 : vector<2x4xf32>
}
// CHECK-LABEL: util.func public @util_func_addf_2d
//  CHECK-SAME:   (%[[A0:.+]]: vector<4xf32>, %[[A1:.+]]: vector<4xf32>, %[[B0:.+]]: vector<4xf32>, %[[B1:.+]]: vector<4xf32>)
//  CHECK-SAME:   -> (vector<4xf32>, vector<4xf32>)
//       CHECK:   %[[R0:.+]] = arith.addf %[[A0]], %[[B0]] : vector<4xf32>
//       CHECK:   %[[R1:.+]] = arith.addf %[[A1]], %[[B1]] : vector<4xf32>
//       CHECK:   util.return %[[R0]], %[[R1]] : vector<4xf32>, vector<4xf32>

// -----

func.func @negative_vector_multi_reduction_rank_one(%arg0: vector<2xf32>, %acc: f32) -> f32 {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0] : vector<2xf32> to f32
    return %0 : f32
}

// CHECK-LABEL: func.func @negative_vector_multi_reduction_rank_one
//       CHECK:   vector.multi_reduction
//       CHECK:   return
