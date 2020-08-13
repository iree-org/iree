// RUN: iree-opt -split-input-file -iree-codegen-linalg-tile-and-fuse=use-workgroup-memory -canonicalize -cse %s | IreeFileCheck %s

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @matmul_tile(%arg0 : memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    linalg.matmul %arg0, %arg1, %arg2 :
      (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>)
    return
  }
}
// CHECK-LABEL: func @matmul_tile
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//       CHECK: scf.parallel (%{{.*}}, %{{.*}})
//       CHECK:   scf.for %{{.*}}
//       CHECK:     %[[ARG0SV:.+]] = subview %[[ARG0]]
//       CHECK:     %[[ARG1SV:.+]] = subview %[[ARG1]]
//       CHECK:     %[[ARG2SV:.+]] = subview %[[ARG2]]
//       CHECK:     %[[ALLOC1:.+]] = alloc() : memref<8x4xf32, 3>
//       CHECK:     %[[SUBVIEW1:.+]] = subview %[[ALLOC1]]
//       CHECK:     %[[ALLOC2:.+]] = alloc() : memref<4x8xf32, 3>
//       CHECK:     %[[SUBVIEW2:.+]] = subview %[[ALLOC2]]
//       CHECK:     linalg.copy(%[[ARG0SV]], %[[SUBVIEW1]])
//  CHECK-SAME:       "copy_to_workgroup_memory"
//       CHECK:     linalg.copy(%[[ARG1SV]], %[[SUBVIEW2]])
//  CHECK-SAME:       "copy_to_workgroup_memory"
//       CHECK:     linalg.matmul
//  CHECK-SAME:       "workgroup_memory_numprocs_ge_numiters"
//  CHECK-SAME:       %[[SUBVIEW1]], %[[SUBVIEW2]], %[[ARG2SV]]
//   CHECK-DAG:     dealloc %[[ALLOC1]] : memref<8x4xf32, 3>
//   CHECK-DAG:     dealloc %[[ALLOC2]] : memref<4x8xf32, 3>

// -----


module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @conv_no_padding_tile(%arg0: memref<3x4x3x2xf32>, %arg1: memref<?x?x?x3xf32>, %arg2: memref<?x?x?x2xf32>) {
    linalg.conv(%arg0, %arg1, %arg2) {dilations = [1, 1], strides = [1, 1]} : memref<3x4x3x2xf32>, memref<?x?x?x3xf32>, memref<?x?x?x2xf32>
    return
  }
}
// CHECK-LABEL: func @conv_no_padding_tile
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: memref<3x4x3x2xf32>
//  CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?x?x3xf32>
//  CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?x?x2xf32>
//       CHECK: scf.parallel (%{{.*}}, %{{.*}}, %{{.*}})
//       CHECK:    %[[ARG1SV:.+]] = subview %[[ARG1]]
//       CHECK:    %[[ARG2SV:.+]] = subview %[[ARG2]]
//       CHECK:    %[[ALLOC1:.+]] = alloc() : memref<1x7x36x3xf32, 3>
//       CHECK:    %[[SUBVIEW1:.+]] = subview %[[ALLOC1]]
//       CHECK:    linalg.copy(%[[ARG1SV]], %[[SUBVIEW1]])
//  CHECK-SAME:       "copy_to_workgroup_memory"
//       CHECK:    linalg.conv(%[[ARG0]], %[[SUBVIEW1]], %[[ARG2SV]])
//  CHECK-SAME:       "workgroup_memory"
//       CHECK:    dealloc %[[ALLOC1]] : memref<1x7x36x3xf32, 3>
