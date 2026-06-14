// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmgpu-lower-1d-vector-ops))" \
// RUN:   --split-input-file %s | FileCheck %s

// transfer_read/write on memref with in_bounds are lowered to vector.load/store.
// CHECK-LABEL: func @transfer_to_load_store(
// CHECK-SAME:    %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:    %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:    %[[RES:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>, vector<4xf32>
// CHECK-NEXT:    vector.store %[[RES]], %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>, vector<4xf32>
// CHECK-NEXT:    return %[[RES]] : vector<4xf32>
func.func @transfer_to_load_store(%mem : memref<8x8xf32>, %idx : index) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 {in_bounds = [true]} : memref<8x8xf32>, vector<4xf32>
  vector.transfer_write %res, %mem[%idx, %idx] {in_bounds = [true]} : vector<4xf32>, memref<8x8xf32>
  return %res : vector<4xf32>
}

// -----

// Scalar-sized vector (vector<1xf32>) on dynamic memref.
// CHECK-LABEL: func @transfer_scalar(
// CHECK-SAME:    %[[MEM:.*]]: memref<?x?xf32>,
// CHECK-SAME:    %[[IDX:.*]]: index) -> vector<1xf32> {
// CHECK-NEXT:    %[[RES:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<?x?xf32>, vector<1xf32>
// CHECK-NEXT:    return %[[RES]] : vector<1xf32>
func.func @transfer_scalar(%mem : memref<?x?xf32>, %idx : index) -> vector<1xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 {in_bounds = [true]} : memref<?x?xf32>, vector<1xf32>
  return %res : vector<1xf32>
}

// -----

// Non-default but unit-stride layout still lowers.
// CHECK-LABEL: func @transfer_nondefault_layout(
// CHECK-SAME:    %[[MEM:.*]]: memref<8x8xf32, #{{.*}}>,
// CHECK-SAME:    %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:    %[[RES:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32, #{{.*}}>, vector<4xf32>
// CHECK-NEXT:    vector.store %[[RES]], %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32, #{{.*}}>, vector<4xf32>
// CHECK-NEXT:    return %[[RES]] : vector<4xf32>

#layout = affine_map<(d0, d1) -> (d0*16 + d1)>
func.func @transfer_nondefault_layout(%mem : memref<8x8xf32, #layout>, %idx : index) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 {in_bounds = [true]} : memref<8x8xf32, #layout>, vector<4xf32>
  vector.transfer_write %res, %mem[%idx, %idx] {in_bounds = [true]} : vector<4xf32>, memref<8x8xf32, #layout>
  return %res : vector<4xf32>
}

// -----

// Out-of-bounds (no in_bounds attr) should NOT be lowered.
// CHECK-LABEL: func @transfer_not_inbounds(
// CHECK-SAME:    %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:    %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:    %[[CF0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[RES:.*]] = vector.transfer_read %[[MEM]][%[[IDX]], %[[IDX]]], %[[CF0]] : memref<8x8xf32>, vector<4xf32>
// CHECK-NEXT:    vector.transfer_write %[[RES]], %[[MEM]][%[[IDX]], %[[IDX]]] : vector<4xf32>, memref<8x8xf32>
// CHECK-NEXT:    return %[[RES]] : vector<4xf32>
func.func @transfer_not_inbounds(%mem : memref<8x8xf32>, %idx : index) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 : memref<8x8xf32>, vector<4xf32>
  vector.transfer_write %res, %mem[%idx, %idx] : vector<4xf32>, memref<8x8xf32>
  return %res : vector<4xf32>
}

// -----

// Non-identity permutation map is unrolled to scf.for with memref.load.
// CHECK-LABEL: func @transfer_perm_map(
// CHECK-SAME:    %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:    %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK:         %[[FOR:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[CST]]) -> (vector<4xf32>) {
// CHECK:           %[[ADDR:.*]] = affine.apply {{.*}}(%[[IV]])[%[[IDX]]]
// CHECK:           %[[LOAD:.*]] = memref.load %[[MEM]][%[[ADDR]], %[[IDX]]] : memref<8x8xf32>
// CHECK:           %[[INS:.*]] = vector.insert %[[LOAD]], %[[ACC]] [%[[IV]]] : f32 into vector<4xf32>
// CHECK:           scf.yield %[[INS]] : vector<4xf32>
// CHECK:         }
// CHECK:         return %[[FOR]] : vector<4xf32>
func.func @transfer_perm_map(%mem : memref<8x8xf32>, %idx : index) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<8x8xf32>, vector<4xf32>
  return %res : vector<4xf32>
}

// -----

// Masked rank-1 transfer_read/write lower to maskedload/maskedstore.
// CHECK-LABEL: func @transfer_masked(
// CHECK-SAME:    %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:    %[[IDX:.*]]: index,
// CHECK-SAME:    %[[MASK:.*]]: vector<4xi1>) -> vector<4xf32> {
// CHECK-NEXT:    %[[FILL:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-NEXT:    %[[RES:.*]] = vector.maskedload %[[MEM]][%[[IDX]], %[[IDX]]], %[[MASK]], %[[FILL]] : memref<8x8xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
// CHECK-NEXT:    vector.maskedstore %[[MEM]][%[[IDX]], %[[IDX]]], %[[MASK]], %[[RES]] : memref<8x8xf32>, vector<4xi1>, vector<4xf32>
// CHECK-NEXT:    return %[[RES]] : vector<4xf32>
func.func @transfer_masked(%mem : memref<8x8xf32>, %idx : index, %mask : vector<4xi1>) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0, %mask {in_bounds = [true]} : memref<8x8xf32>, vector<4xf32>
  vector.transfer_write %res, %mem[%idx, %idx], %mask {in_bounds = [true]} : vector<4xf32>, memref<8x8xf32>
  return %res : vector<4xf32>
}

// -----

// Tensor source should NOT be lowered (only memref is supported).
// CHECK-LABEL: func @transfer_tensor(
// CHECK:         vector.transfer_read
func.func @transfer_tensor(%src : tensor<8x8xf32>, %idx : index) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %src[%idx, %idx], %cf0 {in_bounds = [true]} : tensor<8x8xf32>, vector<4xf32>
  return %res : vector<4xf32>
}

// -----

// Rank-0 transfer_read is lowered to vector.load.
// CHECK-LABEL: func @transfer_read_rank0(
// CHECK-SAME:    %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:    %[[IDX:.*]]: index) -> vector<f32> {
// CHECK-NEXT:    %[[RES:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>, vector<f32>
// CHECK-NEXT:    return %[[RES]] : vector<f32>
func.func @transfer_read_rank0(%mem : memref<8x8xf32>, %idx : index) -> vector<f32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 {in_bounds = []} : memref<8x8xf32>, vector<f32>
  return %res : vector<f32>
}

// -----

// Rank-0 transfer_write is lowered to vector.store.
// CHECK-LABEL: func @transfer_write_rank0(
// CHECK-SAME:    %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:    %[[IDX:.*]]: index,
// CHECK-SAME:    %[[VEC:.*]]: vector<f32>) {
// CHECK-NEXT:    vector.store %[[VEC]], %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>, vector<f32>
// CHECK-NEXT:    return
func.func @transfer_write_rank0(%mem : memref<8x8xf32>, %idx : index, %vec : vector<f32>) {
  vector.transfer_write %vec, %mem[%idx, %idx] {in_bounds = []} : vector<f32>, memref<8x8xf32>
  return
}

// -----

// 1-D vector.multi_reduction is lowered to vector.reduction.
// CHECK-LABEL: func @one_dim_reduction(
// CHECK-SAME:    %[[INPUT:.+]]: vector<8xf32>, %[[ACC:.+]]: f32
func.func @one_dim_reduction(%arg0: vector<8xf32>, %acc: f32) -> f32 {
  // CHECK: %[[RESULT:.+]] = vector.reduction <add>, %[[INPUT]], %[[ACC]] : vector<8xf32> into f32
  %0 = vector.multi_reduction <add>, %arg0, %acc [0] : vector<8xf32> to f32
  // CHECK: return %[[RESULT]]
  return %0 : f32
}

// -----

// Rank-1 transfer_gather with a vector.step index vec folds to vector.load.
// CHECK-LABEL: func @transfer_gather_step_to_load(
// CHECK-SAME:    %[[MEM:.*]]: memref<8x16xf32>,
// CHECK-SAME:    %[[O0:.*]]: index,
// CHECK-SAME:    %[[O1:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:    %[[RES:.*]] = vector.load %[[MEM]][%[[O0]], %[[O1]]] : memref<8x16xf32>, vector<4xf32>
// CHECK-NEXT:    return %[[RES]] : vector<4xf32>
func.func @transfer_gather_step_to_load(%mem : memref<8x16xf32>, %o0 : index, %o1 : index) -> vector<4xf32> {
  %pad = arith.constant 0.0 : f32
  %idx = vector.step : vector<4xindex>
  %res = iree_vector_ext.transfer_gather %mem[%o0, %o1]
      [%idx : vector<4xindex>], %pad {
        indexing_maps = [
          affine_map<(d0)[s0] -> (0, s0)>,
          affine_map<(d0)[s0] -> (d0)>
        ]
      } : memref<8x16xf32>, vector<4xf32>
  return %res : vector<4xf32>
}

// -----

// Rank-1 transfer_scatter with a vector.step index vec folds to vector.store.
// CHECK-LABEL: func @transfer_scatter_step_to_store(
// CHECK-SAME:    %[[MEM:.*]]: memref<8x16xf32>,
// CHECK-SAME:    %[[O0:.*]]: index, %[[O1:.*]]: index,
// CHECK-SAME:    %[[VEC:.*]]: vector<4xf32>) {
// CHECK-NEXT:    vector.store %[[VEC]], %[[MEM]][%[[O0]], %[[O1]]] : memref<8x16xf32>, vector<4xf32>
// CHECK-NEXT:    return
func.func @transfer_scatter_step_to_store(%mem : memref<8x16xf32>, %o0 : index, %o1 : index, %vec : vector<4xf32>) {
  %idx = vector.step : vector<4xindex>
  iree_vector_ext.transfer_scatter %vec into %mem[%o0, %o1]
      [%idx : vector<4xindex>] {
        indexing_maps = [
          affine_map<(d0)[s0] -> (0, s0)>,
          affine_map<(d0)[s0] -> (d0)>
        ]
      } : vector<4xf32>, memref<8x16xf32>
  return
}

// -----

// Masked rank-1 transfer_gather with a vector.step index vec folds to vector.maskedload.
// CHECK-LABEL: func @transfer_gather_step_masked(
// CHECK-SAME:    %[[MEM:.*]]: memref<8x16xf32>,
// CHECK-SAME:    %[[O0:.*]]: index, %[[O1:.*]]: index,
// CHECK-SAME:    %[[MASK:.*]]: vector<4xi1>) -> vector<4xf32> {
// CHECK-NEXT:    %[[FILL:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-NEXT:    %[[RES:.*]] = vector.maskedload %[[MEM]][%[[O0]], %[[O1]]], %[[MASK]], %[[FILL]] : memref<8x16xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
// CHECK-NEXT:    return %[[RES]] : vector<4xf32>
func.func @transfer_gather_step_masked(%mem : memref<8x16xf32>, %o0 : index, %o1 : index, %mask : vector<4xi1>) -> vector<4xf32> {
  %pad = arith.constant 0.0 : f32
  %idx = vector.step : vector<4xindex>
  %res = iree_vector_ext.transfer_gather %mem[%o0, %o1]
      [%idx : vector<4xindex>], %pad, %mask {
        indexing_maps = [
          affine_map<(d0)[s0] -> (0, s0)>,
          affine_map<(d0)[s0] -> (d0)>,
          affine_map<(d0)[s0] -> (d0)>
        ]
      } : memref<8x16xf32>, vector<4xf32>, vector<4xi1>
  return %res : vector<4xf32>
}

// -----

// Masked rank-1 transfer_scatter with a vector.step index vec folds to vector.maskedstore.
// CHECK-LABEL: func @transfer_scatter_step_masked(
// CHECK-SAME:    %[[MEM:.*]]: memref<8x16xf32>,
// CHECK-SAME:    %[[O0:.*]]: index, %[[O1:.*]]: index,
// CHECK-SAME:    %[[VEC:.*]]: vector<4xf32>,
// CHECK-SAME:    %[[MASK:.*]]: vector<4xi1>) {
// CHECK-NEXT:    vector.maskedstore %[[MEM]][%[[O0]], %[[O1]]], %[[MASK]], %[[VEC]] : memref<8x16xf32>, vector<4xi1>, vector<4xf32>
// CHECK-NEXT:    return
func.func @transfer_scatter_step_masked(%mem : memref<8x16xf32>, %o0 : index, %o1 : index, %vec : vector<4xf32>, %mask : vector<4xi1>) {
  %idx = vector.step : vector<4xindex>
  iree_vector_ext.transfer_scatter %vec into %mem[%o0, %o1]
      [%idx : vector<4xindex>], %mask {
        indexing_maps = [
          affine_map<(d0)[s0] -> (0, s0)>,
          affine_map<(d0)[s0] -> (d0)>,
          affine_map<(d0)[s0] -> (d0)>
        ]
      } : vector<4xf32>, memref<8x16xf32>, vector<4xi1>
  return
}
