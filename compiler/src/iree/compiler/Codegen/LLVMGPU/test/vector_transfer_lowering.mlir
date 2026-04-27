// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-vector-transfer-lowering))" --split-input-file %s | FileCheck %s

func.func @broadcast_read_lowering(%arg0: memref<4096x32xf16>) -> vector<1x8xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %thread_id_x = gpu.thread_id x
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %0 = vector.transfer_read %arg0[%workgroup_id_x, %thread_id_x], %cst {in_bounds = [true]} : memref<4096x32xf16>, vector<1xf16>
  %1 = vector.extract %0[0] : f16 from vector<1xf16>
  %2 = vector.broadcast %1 : f16 to vector<1x8xf16>
  return %2 : vector<1x8xf16>
}

// CHECK-LABEL: func.func @broadcast_read_lowering
//  CHECK-SAME: (%[[ARG0:.+]]: memref<4096x32xf16>)
//  CHECK: %[[LOAD:.+]] = vector.load %[[ARG0]]{{.*}} : memref<4096x32xf16>
//  CHECK: %[[ELEM:.+]] = vector.extract %[[LOAD]][0] : f16 from vector<1xf16>
//  CHECK: %[[INSERT:.+]] = vector.broadcast %[[ELEM]] : f16 to vector<1x8xf16>
//  CHECK: return %[[INSERT]]

// -----

func.func @transfer_gather_unroll_embedding_lookup(%arg0: memref<4096x64xf16>, %arg1: vector<4xindex>) -> (vector<64xf16>) {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = vector.extract %arg1[0] : index from vector<4xindex>
  %1 = vector.transfer_read %arg0[%0, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  return %1 : vector<64xf16>
}

// Gather becomes a contiguous loads.
// CHECK-LABEL: func.func @transfer_gather_unroll_embedding_lookup
//   CHECK-NOT:   transfer_read
//       CHECK:   vector.load
//   CHECK-NOT:   transfer_read

// -----

func.func @transfer_gather_unroll_masked(%arg0: memref<4096x64xf16>, %arg1: vector<4xindex>, %arg2: vector<64xi1>) -> (vector<64xf16>) {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = vector.extract %arg1[0] : index from vector<4xindex>
  %1 = vector.transfer_read %arg0[%0, %c0], %cst, %arg2 {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  return %1 : vector<64xf16>
}

// The masked rank-1 gathers lower to vector.maskedload ops.
// CHECK-LABEL: func.func @transfer_gather_unroll_masked
// CHECK-NOT: transfer_read
//     CHECK: vector.maskedload
// CHECK-NOT: transfer_read

// -----

func.func @transfer_gather_unroll_transposed_index(%arg0: memref<4096x64xf16>, %arg1: vector<4xindex>) -> (vector<64xf16>) {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = vector.extract %arg1[0] : index from vector<4xindex>
  %1 = vector.transfer_read %arg0[%0, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  return %1 : vector<64xf16>
}

// CHECK-LABEL: func.func @transfer_gather_unroll_transposed_index
// CHECK-NOT: transfer_read
//     CHECK: vector.load
// CHECK-NOT: transfer_read

// -----

func.func @transfer_scatter_unroll_embedding_write(%arg0: memref<4096x64xf16>, %arg1: vector<64xf16>, %arg2: vector<4xindex>) {
  %c0 = arith.constant 0 : index
  %0 = vector.extract %arg2[0] : index from vector<4xindex>
  vector.transfer_write %arg1, %arg0[%0, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  return
}

// CHECK-LABEL: func.func @transfer_scatter_unroll_embedding_write
// CHECK-COUNT: vector.store {{.+}} : memref<4096x64xf16>, vector<64xf16>

// -----

func.func @transfer_scatter_unroll_masked(%arg0: memref<4096x64xf16>, %arg1: vector<64xf16>, %arg2: vector<4xindex>,  %arg3: vector<64xi1>) {
  %c0 = arith.constant 0 : index
  %0 = vector.extract %arg2[0] : index from vector<4xindex>
  vector.transfer_write %arg1, %arg0[%0, %c0], %arg3 {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  return
}

// CHECK-LABEL: func.func @transfer_scatter_unroll_masked
// CHECK-COUNT: vector.maskedstore {{.+}} : memref<4096x64xf16>, vector<64xi1>, vector<64xf16>

// -----

func.func @transfer_scatter_unroll_tensor(%arg0: tensor<4096x64xf16>, %arg1: vector<64xf16>,  %arg2: vector<4xindex>) -> tensor<4096x64xf16> {
  %c0 = arith.constant 0 : index
  %0 = vector.extract %arg2[0] : index from vector<4xindex>
  %1 = vector.transfer_write %arg1, %arg0[%0, %c0] {in_bounds = [true]} : vector<64xf16>, tensor<4096x64xf16>
  return %1 : tensor<4096x64xf16>
}

// CHECK-LABEL: func.func @transfer_scatter_unroll_tensor
// CHECK-COUNT: vector.transfer_write {{.+}} : vector<64xf16>, tensor<4096x64xf16>

// -----

func.func @transfer_scatter_unroll_transposed_index(%arg0: memref<4096x64xf16>, %arg1: vector<64xf16>, %arg2: vector<4xindex>) {
  %c0 = arith.constant 0 : index
  %0 = vector.extract %arg2[0] : index from vector<4xindex>
  vector.transfer_write %arg1, %arg0[%0, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  return
}

// CHECK-LABEL: func.func @transfer_scatter_unroll_transposed_index
// CHECK-COUNT: vector.store {{.+}} : memref<4096x64xf16>, vector<64xf16>
