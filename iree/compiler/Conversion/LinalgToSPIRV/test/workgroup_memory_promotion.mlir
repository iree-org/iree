// RUN: iree-opt -split-input-file -iree-codegen-linalg-tile-and-fuse=use-workgroup-memory %s | IreeFileCheck %s

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @matmul_tile() {
    %arg0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<96x96xf32>
    %arg1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<96x96xf32>
    %arg2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<96x96xf32>
    linalg.matmul %arg0, %arg1, %arg2 :
      (memref<96x96xf32>, memref<96x96xf32>, memref<96x96xf32>)
    return
  }

  hal.interface @legacy_io attributes {push_constants = 5 : i32, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=2, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write"
  }
}
//  CHECK-DAG: %[[C4:.+]] = constant 4 : index
//  CHECK-DAG: %[[C8:.+]] = constant 8 : index
//  CHECK-DAG: %[[C96:.+]] = constant 96 : index
//  CHECK-DAG: %[[C0:.+]] = constant 0 : index
//      CHECK: %[[ARG0:.+]] = iree.placeholder
// CHECK-SAME:               binding = @legacy_io::@arg0
//      CHECK: %[[ARG1:.+]] = iree.placeholder
// CHECK-SAME:               binding = @legacy_io::@arg1
//      CHECK: %[[RET0:.+]] = iree.placeholder
// CHECK-SAME:               binding = @legacy_io::@ret0
//      CHECK: scf.parallel (%{{.*}}, %{{.*}})
//      CHECK:   scf.for %{{.*}} = %[[C0]] to %{{.*}} step %[[C4]]
//      CHECK:     %[[ARG0SV:.+]] = subview %[[ARG0]]
//      CHECK:     %[[ARG1SV:.+]] = subview %[[ARG1]]
//      CHECK:     %[[RET0SV:.+]] = subview %[[RET0]]
//      CHECK:     %[[ALLOC1:.+]] = alloc(%[[C8]], %[[C4]]) : memref<?x?xf32, 3>
//      CHECK:     %[[SUBVIEW1:.+]] = subview %[[ALLOC1]]
//      CHECK:     %[[ALLOC2:.+]] = alloc(%[[C4]], %[[C8]]) : memref<?x?xf32, 3>
//      CHECK:     %[[SUBVIEW2:.+]] = subview %[[ALLOC2]]
//      CHECK:     linalg.copy(%[[ARG0SV]], %[[SUBVIEW1]])
// CHECK-SAME:       "workitem"
//      CHECK:     spv.ControlBarrier "Workgroup", "Workgroup", "AcquireRelease"
//      CHECK:     linalg.copy(%[[ARG1SV]], %[[SUBVIEW2]])
// CHECK-SAME:       "workitem"
//      CHECK:     spv.ControlBarrier "Workgroup", "Workgroup", "AcquireRelease"
//      CHECK:     linalg.matmul {{.*}}"workitem"{{.*}} %[[SUBVIEW1]], %[[SUBVIEW2]], %[[RET0SV]]
//      CHECK:     spv.ControlBarrier "Workgroup", "Workgroup", "AcquireRelease"
//  CHECK-DAG:     dealloc %[[ALLOC1]] : memref<?x?xf32, 3>
//  CHECK-DAG:     dealloc %[[ALLOC2]] : memref<?x?xf32, 3>
