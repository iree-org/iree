// RUN: iree-opt -iree-codegen-convert-to-gpu -canonicalize -cse -split-input-file  -iree-codegen-constrained-workgroup-count %s | IreeFileCheck %s

#map0 = affine_map<(d0)[s0] -> (8, -d0 + s0)>
#map1 = affine_map<(d0)[s0] -> (4, -d0 + s0)>
#map2 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) attributes {spv.entry_point_abi = {local_size = dense<[8, 8, 1]> : vector<3xi32>}} {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c4 = constant 4 : index
    %c8 = constant 8 : index
    %0 = dim %arg0, %c0 : memref<?x?xf32>
    %1 = dim %arg0, %c1 : memref<?x?xf32>
    %2 = dim %arg1, %c1 : memref<?x?xf32>
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%0, %2) step (%c8, %c8) {
      scf.for %arg5 = %c0 to %1 step %c4 {
        %3 = affine.min #map0(%arg3)[%0]
        %4 = affine.min #map1(%arg5)[%1]
        %5 = subview %arg0[%arg3, %arg5] [%3, %4] [1, 1]  : memref<?x?xf32> to memref<?x?xf32, #map2>
        %6 = dim %arg1, %c0 : memref<?x?xf32>
        %7 = affine.min #map1(%arg5)[%6]
        %8 = affine.min #map0(%arg4)[%2]
        %9 = subview %arg1[%arg5, %arg4] [%7, %8] [1, 1]  : memref<?x?xf32> to memref<?x?xf32, #map2>
        %10 = dim %arg2, %c0 : memref<?x?xf32>
        %11 = affine.min #map0(%arg3)[%10]
        %12 = dim %arg2, %c1 : memref<?x?xf32>
        %13 = affine.min #map0(%arg4)[%12]
        %14 = subview %arg2[%arg3, %arg4] [%11, %13] [1, 1]  : memref<?x?xf32> to memref<?x?xf32, #map2>
        linalg.matmul %5, %9, %14 {__internal_linalg_transform__ = "workitem"} : (memref<?x?xf32, #map2>, memref<?x?xf32, #map2>, memref<?x?xf32, #map2>)
      }
      scf.yield
    }
    return
  }
}

// CHECK-LABEL: func @matmul
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//   CHECK-DAG:   %[[C4:.+]] = constant 4 : index
//   CHECK-DAG:   %[[C8:.+]] = constant 8 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[UB0:.+]] = dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[UB1:.+]] = dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[UB2:.+]] = dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[GDIMX:.+]] = "gpu.grid_dim"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-DAG:   %[[GDIMY:.+]] = "gpu.grid_dim"() {dimension = "y"}
//       CHECK:   %[[BOFFSETY:.+]] = muli %[[BIDY]], %[[C8]]
//       CHECK:   %[[BSTEPY:.+]] = muli %[[GDIMY]], %[[C8]]
//       CHECK:   %[[BOFFSETX:.+]] = muli %[[BIDX]], %[[C8]]
//       CHECK:   %[[BSTEPX:.+]] = muli %[[GDIMX]], %[[C8]]
//       CHECK:   scf.for %[[BIV0:.+]] = %[[BOFFSETY]] to %[[UB0]] step %[[BSTEPY]]
//       CHECK:     scf.for %[[BIV1:.+]] = %[[BOFFSETX]] to %[[UB1]] step %[[BSTEPX]]
//       CHECK:       scf.for %[[BIV2:.+]] = %[[C0]] to %[[UB2]] step %[[C4]]
//   CHECK-DAG:         %[[VIEWUB0:.+]] = affine.min #{{.*}}(%[[BIV0]])[%[[UB0]]]
//   CHECK-DAG:         %[[VIEWUB1:.+]] = affine.min #{{.*}}(%[[BIV1]])[%[[UB1]]]
//   CHECK-DAG:         %[[VIEWUB2:.+]] = affine.min #{{.*}}(%[[BIV2]])[%[[UB2]]]
//   CHECK-DAG:         %[[TIDX:.+]] = "gpu.thread_id"() {dimension = "x"}
//   CHECK-DAG:         %[[TIDY:.+]] = "gpu.thread_id"() {dimension = "y"}
//       CHECK:         %[[INBOUNDY:.+]] = cmpi "slt", %[[TIDY]], %[[VIEWUB0]]
//       CHECK:         %[[INBOUNDX:.+]] = cmpi "slt", %[[TIDX]], %[[VIEWUB1]]
//       CHECK:         %[[COND:.+]] = and %[[INBOUNDY]], %[[INBOUNDX]]
//       CHECK:         scf.if %[[COND]]
//       CHECK:           scf.for %{{.*}} = %[[C0]] to %[[VIEWUB2]] step %[[C1]]
