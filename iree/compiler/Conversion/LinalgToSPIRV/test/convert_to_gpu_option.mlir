// RUN: iree-opt -iree-codegen-convert-to-gpu -iree-codegen-use-legacy-conv-lowering=false -canonicalize -cse -split-input-file %s | IreeFileCheck %s

#map0 = affine_map<(d0)[s0] -> (1, -d0 + s0)>
#map1 = affine_map<(d0)[s0, s1] -> (s0 + 4, -d0 + s1)>
#map2 = affine_map<(d0)[s0, s1] -> (s0 + 32, -d0 + s1)>
#map3 = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>
#map4 = affine_map<(d0)[s0] -> (4, -d0 + s0)>
#map5 = affine_map<(d0)[s0] -> (32, -d0 + s0)>

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @conv_no_padding(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) attributes {spv.entry_point_abi = {local_size = dense<[32, 4, 1]> : vector<3xi32>}} {
    %c4 = constant 4 : index
    %c32 = constant 32 : index
    %c2 = constant 2 : index
    %c0 = constant 0 : index
    %c3 = constant 3 : index
    %c1 = constant 1 : index
    %0 = dim %arg1, %c0 : memref<?x?x?x?xf32>
    %1 = dim %arg1, %c1 : memref<?x?x?x?xf32>
    %2 = dim %arg1, %c2 : memref<?x?x?x?xf32>
    %3 = dim %arg2, %c1 : memref<?x?x?x?xf32>
    %4 = dim %arg2, %c2 : memref<?x?x?x?xf32>
    scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%0, %3, %4) step (%c1, %c4, %c32) {
      %5 = affine.min #map0(%arg3)[%0]
      %6 = affine.min #map1(%arg4)[%1, %1]
      %7 = affine.min #map2(%arg5)[%2, %2]
      %8 = dim %arg1, %c3 : memref<?x?x?x?xf32>
      %9 = subview %arg1[%arg3, %arg4, %arg5, 0] [%5, %6, %7, %8] [1, 1, 1, 1]  : memref<?x?x?x?xf32> to memref<?x?x?x?xf32, #map3>
      %10 = dim %arg2, %c0 : memref<?x?x?x?xf32>
      %11 = affine.min #map0(%arg3)[%10]
      %12 = affine.min #map4(%arg4)[%3]
      %13 = affine.min #map5(%arg5)[%4]
      %14 = dim %arg2, %c3 : memref<?x?x?x?xf32>
      %15 = subview %arg2[%arg3, %arg4, %arg5, 0] [%11, %12, %13, %14] [1, 1, 1, 1]  : memref<?x?x?x?xf32> to memref<?x?x?x?xf32, #map3>
      linalg.conv(%arg0, %9, %15) {__internal_linalg_transform__ = "workitem", dilations = [1, 1], strides = [1, 1]} : memref<?x?x?x?xf32>, memref<?x?x?x?xf32, #map3>, memref<?x?x?x?xf32, #map3>
      scf.yield
    }
    return
  }
}

// CHECK-LABEL: func @conv_no_padding
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//   CHECK-DAG:   %[[C4:.+]] = constant 4 : index
//   CHECK-DAG:   %[[C32:.+]] = constant 32 : index
//   CHECK-DAG:   %[[C2:.+]] = constant 2 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[UB0:.+]] = dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:   %[[UB1:.+]] = dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[UB2:.+]] = dim %[[ARG1]], %[[C2]]
//   CHECK-DAG:   %[[UB3:.+]] = dim %[[ARG2]], %[[C1]]
//   CHECK-DAG:   %[[UB4:.+]] = dim %[[ARG2]], %[[C2]]
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-DAG:   %[[BIDZ:.+]] = "gpu.block_id"() {dimension = "z"}
//   CHECK-DAG:   %[[BOFFSETY:.+]] = muli %[[BIDY]], %[[C4]]
//   CHECK-DAG:   %[[BOFFSETX:.+]] = muli %[[BIDX]], %[[C32]]
//       CHECK:   %[[SV1:.+]] = subview %[[ARG1]][%[[BIDZ]], %[[BOFFSETY]], %[[BOFFSETX]], 0]
//       CHECK:   %[[SV2:.+]] = subview %[[ARG2]][%[[BIDZ]], %[[BOFFSETY]], %[[BOFFSETX]], 0]
//   CHECK-DAG:   %[[TIDX:.+]] = "gpu.thread_id"() {dimension = "x"}
//   CHECK-DAG:   %[[TIDY:.+]] = "gpu.thread_id"() {dimension = "y"}
//   CHECK-DAG:   %[[TIDZ:.+]] = "gpu.thread_id"() {dimension = "z"}
//       CHECK:   %[[INBOUNDSZ:.+]] = cmpi "slt", %[[TIDZ]], %{{.+}}
//       CHECK:   %[[INBOUNDSY:.+]] = cmpi "slt", %[[TIDY]], %{{.+}}
//       CHECK:   %[[T35:.+]] = and %[[INBOUNDSZ]], %[[INBOUNDSY]]
//       CHECK:   %[[INBOUNDSX:.+]] = cmpi "slt", %[[TIDX]], %{{.+}}
//       CHECK:   %[[INBOUNDS:.+]] = and %[[T35]], %[[INBOUNDSX]]
//       CHECK:   scf.if %[[INBOUNDS]]
//       CHECK:     scf.for
//       CHECK:       scf.for
//       CHECK:         scf.for
//       CHECK:           scf.for
//   CHECK-NOT:             linalg.conv
