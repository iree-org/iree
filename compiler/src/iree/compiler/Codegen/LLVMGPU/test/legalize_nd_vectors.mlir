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

// transfer_read with 2-D result: unrolled into rank-1 reads with adjusted offsets.
func.func @transfer_read_2d(%A: memref<?x?x?xf32>, %a: index, %b: index, %c: index, %padding: f32) -> vector<5x4xf32> {
  %vec = vector.transfer_read %A[%a, %b, %c], %padding {in_bounds = [true, false]} : memref<?x?x?xf32>, vector<5x4xf32>
  return %vec : vector<5x4xf32>
}
// CHECK-DAG:   #[[$ID:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   #[[$P1:.*]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   #[[$P2:.*]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   #[[$P3:.*]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-DAG:   #[[$P4:.*]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-LABEL: func.func @transfer_read_2d
//  CHECK-SAME:   (%[[A:.+]]: memref<?x?x?xf32>, %[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index, %[[PAD:.+]]: f32)
//  CHECK-SAME:   -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
//       CHECK:   %[[OFF1_0:.+]] = affine.apply #[[$ID]](%[[IDX1]])
//       CHECK:   %[[OFF2_0:.+]] = affine.apply #[[$ID]](%[[IDX2]])
//       CHECK:   %[[V0:.+]] = vector.transfer_read %[[A]][%[[IDX0]], %[[OFF1_0]], %[[OFF2_0]]], %[[PAD]] : memref<?x?x?xf32>, vector<4xf32>
//       CHECK:   %[[OFF1_1:.+]] = affine.apply #[[$P1]](%[[IDX1]])
//       CHECK:   %[[OFF2_1:.+]] = affine.apply #[[$ID]](%[[IDX2]])
//       CHECK:   %[[V1:.+]] = vector.transfer_read %[[A]][%[[IDX0]], %[[OFF1_1]], %[[OFF2_1]]], %[[PAD]] : memref<?x?x?xf32>, vector<4xf32>
//       CHECK:   %[[OFF1_2:.+]] = affine.apply #[[$P2]](%[[IDX1]])
//       CHECK:   %[[OFF2_2:.+]] = affine.apply #[[$ID]](%[[IDX2]])
//       CHECK:   %[[V2:.+]] = vector.transfer_read %[[A]][%[[IDX0]], %[[OFF1_2]], %[[OFF2_2]]], %[[PAD]] : memref<?x?x?xf32>, vector<4xf32>
//       CHECK:   %[[OFF1_3:.+]] = affine.apply #[[$P3]](%[[IDX1]])
//       CHECK:   %[[OFF2_3:.+]] = affine.apply #[[$ID]](%[[IDX2]])
//       CHECK:   %[[V3:.+]] = vector.transfer_read %[[A]][%[[IDX0]], %[[OFF1_3]], %[[OFF2_3]]], %[[PAD]] : memref<?x?x?xf32>, vector<4xf32>
//       CHECK:   %[[OFF1_4:.+]] = affine.apply #[[$P4]](%[[IDX1]])
//       CHECK:   %[[OFF2_4:.+]] = affine.apply #[[$ID]](%[[IDX2]])
//       CHECK:   %[[V4:.+]] = vector.transfer_read %[[A]][%[[IDX0]], %[[OFF1_4]], %[[OFF2_4]]], %[[PAD]] : memref<?x?x?xf32>, vector<4xf32>
//       CHECK:   return %[[V0]], %[[V1]], %[[V2]], %[[V3]], %[[V4]] : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>

// -----

// transfer_read with 2-D mask: each unrolled read gets the corresponding 1-D mask slice.
func.func @transfer_read_2d_masked(%A: memref<?x?x?xf32>, %a: index, %b: index, %c: index, %padding: f32, %mask: vector<2x4xi1>) -> vector<2x4xf32> {
  %vec = vector.transfer_read %A[%a, %b, %c], %padding, %mask {in_bounds = [true, false]} : memref<?x?x?xf32>, vector<2x4xf32>
  return %vec : vector<2x4xf32>
}
// CHECK-LABEL: func.func @transfer_read_2d_masked
//  CHECK-SAME:   (%[[A:.+]]: memref<?x?x?xf32>, %[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index, %[[PAD:.+]]: f32, %[[M0:.+]]: vector<4xi1>, %[[M1:.+]]: vector<4xi1>)
//  CHECK-SAME:   -> (vector<4xf32>, vector<4xf32>)
//       CHECK:   %[[OFF1_0:.+]] = affine.apply #[[$ID]](%[[IDX1]])
//       CHECK:   %[[OFF2_0:.+]] = affine.apply #[[$ID]](%[[IDX2]])
//       CHECK:   %[[V0:.+]] = vector.transfer_read %[[A]][%[[IDX0]], %[[OFF1_0]], %[[OFF2_0]]], %[[PAD]], %[[M0]]
//  CHECK-SAME:     : memref<?x?x?xf32>, vector<4xf32>
//       CHECK:   %[[OFF1_1:.+]] = affine.apply #[[$P1]](%[[IDX1]])
//       CHECK:   %[[OFF2_1:.+]] = affine.apply #[[$ID]](%[[IDX2]])
//       CHECK:   %[[V1:.+]] = vector.transfer_read %[[A]][%[[IDX0]], %[[OFF1_1]], %[[OFF2_1]]], %[[PAD]], %[[M1]]
//  CHECK-SAME:     : memref<?x?x?xf32>, vector<4xf32>
//       CHECK:   return %[[V0]], %[[V1]] : vector<4xf32>, vector<4xf32>

// -----

// transfer_write with 2-D vector on memref: unrolled into rank-1 writes with adjusted offsets.
func.func @transfer_write_2d(%vec: vector<3x4xf32>, %A: memref<?x?x?xf32>, %a: index, %b: index, %c: index) {
  vector.transfer_write %vec, %A[%a, %b, %c] {in_bounds = [true, false]} : vector<3x4xf32>, memref<?x?x?xf32>
  return
}
// CHECK-LABEL: func.func @transfer_write_2d
//  CHECK-SAME:   (%[[V0:.+]]: vector<4xf32>, %[[V1:.+]]: vector<4xf32>, %[[V2:.+]]: vector<4xf32>, %[[A:.+]]: memref<?x?x?xf32>, %[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index)
//       CHECK:   %[[OFF1_0:.+]] = affine.apply #[[$ID]](%[[IDX1]])
//       CHECK:   %[[OFF2_0:.+]] = affine.apply #[[$ID]](%[[IDX2]])
//       CHECK:   vector.transfer_write %[[V0]], %[[A]][%[[IDX0]], %[[OFF1_0]], %[[OFF2_0]]] : vector<4xf32>, memref<?x?x?xf32>
//       CHECK:   %[[OFF1_1:.+]] = affine.apply #[[$P1]](%[[IDX1]])
//       CHECK:   %[[OFF2_1:.+]] = affine.apply #[[$ID]](%[[IDX2]])
//       CHECK:   vector.transfer_write %[[V1]], %[[A]][%[[IDX0]], %[[OFF1_1]], %[[OFF2_1]]] : vector<4xf32>, memref<?x?x?xf32>
//       CHECK:   %[[OFF1_2:.+]] = affine.apply #[[$P2]](%[[IDX1]])
//       CHECK:   %[[OFF2_2:.+]] = affine.apply #[[$ID]](%[[IDX2]])
//       CHECK:   vector.transfer_write %[[V2]], %[[A]][%[[IDX0]], %[[OFF1_2]], %[[OFF2_2]]] : vector<4xf32>, memref<?x?x?xf32>
//       CHECK:   return

// -----

// transfer_write with 2-D mask: each unrolled write gets the corresponding 1-D mask slice.
func.func @transfer_write_2d_masked(%vec: vector<2x4xf32>, %A: memref<?x?x?xf32>, %a: index, %b: index, %c: index, %mask: vector<2x4xi1>) {
  vector.transfer_write %vec, %A[%a, %b, %c], %mask {in_bounds = [true, false]} : vector<2x4xf32>, memref<?x?x?xf32>
  return
}
// CHECK-LABEL: func.func @transfer_write_2d_masked
//  CHECK-SAME:   (%[[V0:.+]]: vector<4xf32>, %[[V1:.+]]: vector<4xf32>, %[[A:.+]]: memref<?x?x?xf32>, %[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index, %[[M0:.+]]: vector<4xi1>, %[[M1:.+]]: vector<4xi1>)
//       CHECK:   %[[OFF1_0:.+]] = affine.apply #[[$ID]](%[[IDX1]])
//       CHECK:   %[[OFF2_0:.+]] = affine.apply #[[$ID]](%[[IDX2]])
//       CHECK:   vector.transfer_write %[[V0]], %[[A]][%[[IDX0]], %[[OFF1_0]], %[[OFF2_0]]], %[[M0]]
//  CHECK-SAME:     : vector<4xf32>, memref<?x?x?xf32>
//       CHECK:   %[[OFF1_1:.+]] = affine.apply #[[$P1]](%[[IDX1]])
//       CHECK:   %[[OFF2_1:.+]] = affine.apply #[[$ID]](%[[IDX2]])
//       CHECK:   vector.transfer_write %[[V1]], %[[A]][%[[IDX0]], %[[OFF1_1]], %[[OFF2_1]]], %[[M1]]
//  CHECK-SAME:     : vector<4xf32>, memref<?x?x?xf32>
//       CHECK:   return

// -----

// transfer_read with OOB outer dim: generates scf.if bounds checks with
// padding fallback. Outer dim (vector dim 0) maps to memref dim 0 and is
// not guaranteed in-bounds.
func.func @transfer_read_2d_oob(%A: memref<?x?xf32>, %i: index, %j: index, %pad: f32) -> vector<2x4xf32> {
  %vec = vector.transfer_read %A[%i, %j], %pad {in_bounds = [false, true]} : memref<?x?xf32>, vector<2x4xf32>
  return %vec : vector<2x4xf32>
}
// CHECK-LABEL: func.func @transfer_read_2d_oob
//  CHECK-SAME:   (%[[A:.+]]: memref<?x?xf32>, %[[I:.+]]: index, %[[J:.+]]: index, %[[PAD:.+]]: f32)
//  CHECK-SAME:   -> (vector<4xf32>, vector<4xf32>)
//       CHECK:   %[[PADVEC:.+]] = vector.broadcast %[[PAD]] : f32 to vector<4xf32>
//       CHECK:   %[[I0:.+]] = affine.apply #[[$ID]](%[[I]])
//       CHECK:   %[[J0:.+]] = affine.apply #[[$ID]](%[[J]])
//       CHECK:   %[[DIM0:.+]] = memref.dim %[[A]], %{{.+}} : memref<?x?xf32>
//       CHECK:   %[[CMP0:.+]] = arith.cmpi slt, %[[I0]], %[[DIM0]] : index
//       CHECK:   %[[R0:.+]] = scf.if %[[CMP0]] -> (vector<4xf32>) {
//       CHECK:     vector.transfer_read %[[A]][%[[I0]], %[[J0]]], %[[PAD]] {in_bounds = [true]}
//       CHECK:     scf.yield
//       CHECK:   } else {
//       CHECK:     scf.yield %[[PADVEC]]
//       CHECK:   }
//       CHECK:   %[[I1:.+]] = affine.apply #[[$P1]](%[[I]])
//       CHECK:   %[[J1:.+]] = affine.apply #[[$ID]](%[[J]])
//       CHECK:   %[[DIM1:.+]] = memref.dim %[[A]], %{{.+}} : memref<?x?xf32>
//       CHECK:   %[[CMP1:.+]] = arith.cmpi slt, %[[I1]], %[[DIM1]] : index
//       CHECK:   %[[R1:.+]] = scf.if %[[CMP1]] -> (vector<4xf32>) {
//       CHECK:     vector.transfer_read %[[A]][%[[I1]], %[[J1]]], %[[PAD]] {in_bounds = [true]}
//       CHECK:     scf.yield
//       CHECK:   } else {
//       CHECK:     scf.yield %[[PADVEC]]
//       CHECK:   }
//       CHECK:   return %[[R0]], %[[R1]]

// -----

// transfer_write with OOB outer dim on memref: generates scf.if to
// conditionally skip the write when the outer index is out of bounds.
func.func @transfer_write_2d_oob(%vec: vector<2x4xf32>, %A: memref<?x?xf32>, %i: index, %j: index) {
  vector.transfer_write %vec, %A[%i, %j] {in_bounds = [false, true]} : vector<2x4xf32>, memref<?x?xf32>
  return
}
// CHECK-LABEL: func.func @transfer_write_2d_oob
//  CHECK-SAME:   (%[[V0:.+]]: vector<4xf32>, %[[V1:.+]]: vector<4xf32>, %[[A:.+]]: memref<?x?xf32>, %[[I:.+]]: index, %[[J:.+]]: index)
//       CHECK:   %[[I0:.+]] = affine.apply #[[$ID]](%[[I]])
//       CHECK:   %[[J0:.+]] = affine.apply #[[$ID]](%[[J]])
//       CHECK:   %[[DIM0:.+]] = memref.dim %[[A]], %{{.+}} : memref<?x?xf32>
//       CHECK:   %[[CMP0:.+]] = arith.cmpi slt, %[[I0]], %[[DIM0]] : index
//       CHECK:   scf.if %[[CMP0]] {
//       CHECK:     vector.transfer_write %[[V0]], %[[A]][%[[I0]], %[[J0]]] {in_bounds = [true]}
//       CHECK:   }
//       CHECK:   %[[I1:.+]] = affine.apply #[[$P1]](%[[I]])
//       CHECK:   %[[J1:.+]] = affine.apply #[[$ID]](%[[J]])
//       CHECK:   %[[DIM1:.+]] = memref.dim %[[A]], %{{.+}} : memref<?x?xf32>
//       CHECK:   %[[CMP1:.+]] = arith.cmpi slt, %[[I1]], %[[DIM1]] : index
//       CHECK:   scf.if %[[CMP1]] {
//       CHECK:     vector.transfer_write %[[V1]], %[[A]][%[[I1]], %[[J1]]] {in_bounds = [true]}
//       CHECK:   }
//       CHECK:   return

// -----

// transfer_write on tensor: each unrolled rank-1 write chains the SSA result
// as the destination for the next write.
func.func @transfer_write_tensor(%vec: vector<3x4xf32>, %dest: tensor<?x?xf32>, %i: index, %j: index) -> tensor<?x?xf32> {
  %res = vector.transfer_write %vec, %dest[%i, %j] {in_bounds = [true, true]} : vector<3x4xf32>, tensor<?x?xf32>
  return %res : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @transfer_write_tensor
//  CHECK-SAME:   (%[[V0:.+]]: vector<4xf32>, %[[V1:.+]]: vector<4xf32>, %[[V2:.+]]: vector<4xf32>, %[[DEST:.+]]: tensor<?x?xf32>, %[[I:.+]]: index, %[[J:.+]]: index)
//  CHECK-SAME:   -> tensor<?x?xf32>
//       CHECK:   %[[W0:.+]] = vector.transfer_write %[[V0]], %[[DEST]][{{.*}}] {in_bounds = [true]} : vector<4xf32>, tensor<?x?xf32>
//       CHECK:   %[[W1:.+]] = vector.transfer_write %[[V1]], %[[W0]][{{.*}}] {in_bounds = [true]} : vector<4xf32>, tensor<?x?xf32>
//       CHECK:   %[[W2:.+]] = vector.transfer_write %[[V2]], %[[W1]][{{.*}}] {in_bounds = [true]} : vector<4xf32>, tensor<?x?xf32>
//       CHECK:   return %[[W2]] : tensor<?x?xf32>

// -----

// transfer_read with all dims in-bounds: no bounds check, inner in_bounds is [true].
func.func @transfer_read_all_in_bounds(%A: memref<?x?xf32>, %i: index, %j: index, %pad: f32) -> vector<2x4xf32> {
  %vec = vector.transfer_read %A[%i, %j], %pad {in_bounds = [true, true]} : memref<?x?xf32>, vector<2x4xf32>
  return %vec : vector<2x4xf32>
}
// CHECK-LABEL: func.func @transfer_read_all_in_bounds
//  CHECK-SAME:   (%[[A:.+]]: memref<?x?xf32>, %[[I:.+]]: index, %[[J:.+]]: index, %[[PAD:.+]]: f32)
//   CHECK-NOT:   scf.if
//   CHECK-NOT:   memref.dim
//       CHECK:   vector.transfer_read %[[A]]{{.*}} {in_bounds = [true]} : memref<?x?xf32>, vector<4xf32>
//       CHECK:   vector.transfer_read %[[A]]{{.*}} {in_bounds = [true]} : memref<?x?xf32>, vector<4xf32>
//       CHECK:   return {{.*}} : vector<4xf32>, vector<4xf32>

// -----

// transfer_read with 3-D result: unrolled into 2*3=6 rank-1 reads.
func.func @transfer_read_3d(%A: memref<?x?x?x?xf32>, %a: index, %b: index, %c: index, %d: index, %pad: f32) -> vector<2x3x4xf32> {
  %vec = vector.transfer_read %A[%a, %b, %c, %d], %pad {in_bounds = [true, true, true]} : memref<?x?x?x?xf32>, vector<2x3x4xf32>
  return %vec : vector<2x3x4xf32>
}
// CHECK-LABEL: func.func @transfer_read_3d
//  CHECK-SAME:   -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
//   CHECK-NOT:   scf.if
//  CHECK-COUNT-6: vector.transfer_read {{.*}} : memref<?x?x?x?xf32>, vector<4xf32>
//       CHECK:   return {{.*}} : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>

// -----

// Rank-1 transfer_read is left untouched by the pattern.
func.func @negative_transfer_read_1d(%A: memref<?xf32>, %i: index, %pad: f32) -> vector<4xf32> {
  %vec = vector.transfer_read %A[%i], %pad {in_bounds = [true]} : memref<?xf32>, vector<4xf32>
  return %vec : vector<4xf32>
}
// CHECK-LABEL: func.func @negative_transfer_read_1d
//  CHECK-SAME:   (%[[A:.+]]: memref<?xf32>, %[[I:.+]]: index, %[[PAD:.+]]: f32) -> vector<4xf32>
//       CHECK:   %[[V:.+]] = vector.transfer_read %[[A]][%[[I]]], %[[PAD]] {in_bounds = [true]} : memref<?xf32>, vector<4xf32>
//       CHECK:   return %[[V]] : vector<4xf32>

// -----

// Rank-1 transfer_write is left untouched by the pattern.
func.func @negative_transfer_write_1d(%vec: vector<4xf32>, %A: memref<?xf32>, %i: index) {
  vector.transfer_write %vec, %A[%i] {in_bounds = [true]} : vector<4xf32>, memref<?xf32>
  return
}
// CHECK-LABEL: func.func @negative_transfer_write_1d
//  CHECK-SAME:   (%[[V:.+]]: vector<4xf32>, %[[A:.+]]: memref<?xf32>, %[[I:.+]]: index)
//       CHECK:   vector.transfer_write %[[V]], %[[A]][%[[I]]] {in_bounds = [true]} : vector<4xf32>, memref<?xf32>
//       CHECK:   return

// -----

// transfer_read with OOB outer dim and a mask: the scf.if guard wraps the
// masked read, and the else branch yields the padding vector.
func.func @transfer_read_oob_masked(%A: memref<?x?xf32>, %i: index, %j: index, %pad: f32, %mask: vector<2x4xi1>) -> vector<2x4xf32> {
  %vec = vector.transfer_read %A[%i, %j], %pad, %mask {in_bounds = [false, true]} : memref<?x?xf32>, vector<2x4xf32>
  return %vec : vector<2x4xf32>
}
// CHECK-LABEL: func.func @transfer_read_oob_masked
//  CHECK-SAME:   (%[[A:.+]]: memref<?x?xf32>, %[[I:.+]]: index, %[[J:.+]]: index, %[[PAD:.+]]: f32, %[[M0:.+]]: vector<4xi1>, %[[M1:.+]]: vector<4xi1>)
//       CHECK:   %[[PADVEC:.+]] = vector.broadcast %[[PAD]] : f32 to vector<4xf32>
//       CHECK:   %[[DIM:.+]] = memref.dim %[[A]], %{{.+}} : memref<?x?xf32>
//       CHECK:   %[[CMP0:.+]] = arith.cmpi slt, %{{.+}}, %[[DIM]] : index
//       CHECK:   %[[R0:.+]] = scf.if %[[CMP0]] -> (vector<4xf32>) {
//       CHECK:     vector.transfer_read %[[A]]{{.*}}, %[[PAD]], %[[M0]] {in_bounds = [true]}
//       CHECK:     scf.yield
//       CHECK:   } else {
//       CHECK:     scf.yield %[[PADVEC]]
//       CHECK:   }
//       CHECK:   %[[CMP1:.+]] = arith.cmpi slt, %{{.+}}, %{{.+}} : index
//       CHECK:   %[[R1:.+]] = scf.if %[[CMP1]] -> (vector<4xf32>) {
//       CHECK:     vector.transfer_read %[[A]]{{.*}}, %[[PAD]], %[[M1]] {in_bounds = [true]}
//       CHECK:     scf.yield
//       CHECK:   } else {
//       CHECK:     scf.yield %[[PADVEC]]
//       CHECK:   }
//       CHECK:   return %[[R0]], %[[R1]]

// -----

// transfer_write on tensor with OOB outer dim: generates scf.if that yields
// the updated tensor or the original when the index is out of bounds.
func.func @transfer_write_tensor_oob(%vec: vector<2x4xf32>, %dest: tensor<?x?xf32>, %i: index, %j: index) -> tensor<?x?xf32> {
  %res = vector.transfer_write %vec, %dest[%i, %j] {in_bounds = [false, true]} : vector<2x4xf32>, tensor<?x?xf32>
  return %res : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @transfer_write_tensor_oob
//  CHECK-SAME:   (%[[V0:.+]]: vector<4xf32>, %[[V1:.+]]: vector<4xf32>, %[[DEST:.+]]: tensor<?x?xf32>, %[[I:.+]]: index, %[[J:.+]]: index)
//  CHECK-SAME:   -> tensor<?x?xf32>
//       CHECK:   %[[I0:.+]] = affine.apply #[[$ID]](%[[I]])
//       CHECK:   %[[DIM0:.+]] = tensor.dim %[[DEST]], %{{.+}} : tensor<?x?xf32>
//       CHECK:   %[[CMP0:.+]] = arith.cmpi slt, %[[I0]], %[[DIM0]] : index
//       CHECK:   %[[R0:.+]] = scf.if %[[CMP0]] -> (tensor<?x?xf32>) {
//       CHECK:     %[[W0:.+]] = vector.transfer_write %[[V0]], %[[DEST]][{{.*}}] {in_bounds = [true]}
//       CHECK:     scf.yield %[[W0]] : tensor<?x?xf32>
//       CHECK:   } else {
//       CHECK:     scf.yield %[[DEST]] : tensor<?x?xf32>
//       CHECK:   }
//       CHECK:   %[[I1:.+]] = affine.apply #[[$P1]](%[[I]])
//       CHECK:   %[[DIM1:.+]] = tensor.dim %[[DEST]], %{{.+}} : tensor<?x?xf32>
//       CHECK:   %[[CMP1:.+]] = arith.cmpi slt, %[[I1]], %[[DIM1]] : index
//       CHECK:   %[[R1:.+]] = scf.if %[[CMP1]] -> (tensor<?x?xf32>) {
//       CHECK:     %[[W1:.+]] = vector.transfer_write %[[V1]], %[[R0]][{{.*}}] {in_bounds = [true]}
//       CHECK:     scf.yield %[[W1]] : tensor<?x?xf32>
//       CHECK:   } else {
//       CHECK:     scf.yield %[[R0]] : tensor<?x?xf32>
//       CHECK:   }
//       CHECK:   return %[[R1]] : tensor<?x?xf32>
