// RUN: iree-opt --pass-pipeline='hal.executable(hal.executable.variant(builtin.module(func.func(iree-gpu-distribute-shared-memory-copy))))' --cse %s | FileCheck %s

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 32 + s0 floordiv 4)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 4) * 16)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 32 + s0 floordiv 4 + 32)>
// CHECK-DAG: #[[$MAP3:.*]] = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 128)>
// CHECK-DAG: #[[$MAP4:.*]] = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 128 + 128)>
// CHECK-DAG: #[[$MAP5:.*]] = affine_map<()[s0, s1, s2] -> (s0 * 4 + s1 * 128 + s2 * 512)>

#map0 = affine_map<()[s0, s1, s2] -> (s0 * 4 + s1 * 128 + s2 * 512)>

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @shared_mem_cpy  {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @shared_mem_cpy layout(#executable_layout) {
      workgroup_size = [32: index, 4: index, 1:index]
    }
    builtin.module {
      memref.global "private" @__shared_memory___1 : memref<3x512xf32, 3>
      memref.global "private" @__shared_memory___0 : memref<256x4xf32, 3>
      memref.global "private" @__shared_memory__ : memref<64x16xf32, 3>
    // CHECK-LABEL: @shared_mem_cpy(
      func.func @shared_mem_cpy(
        %m0 : memref<64x16xf32>, %m1 : memref<256x4xf32>, %m2 : memref<3x512xf32>) {
        %c0 = arith.constant 0 : index

        %0 = "affine.apply"(%c0) {map = affine_map<(d0) -> (d0)>} : (index) -> (index)
        %sm0 = memref.get_global @__shared_memory__ : memref<64x16xf32, 3>
        %sm1 = memref.get_global @__shared_memory___0 : memref<256x4xf32, 3>
        %sm2 = memref.get_global @__shared_memory___1 : memref<3x512xf32, 3>
        gpu.barrier
    // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[TX:.*]] = gpu.thread_id x
    // CHECK-DAG: %[[TY:.*]] = gpu.thread_id y
    // CHECK-DAG: %[[TZ:.*]] = gpu.thread_id z

    // CHECK-DAG: %[[Y0:.*]] = affine.apply #[[$MAP0]]()[%[[TX]], %[[TY]], %[[TZ]]]
    // CHECK-DAG: %[[X0:.*]] = affine.apply #[[$MAP1]]()[%[[TX]]]
    //     CHECK: %[[R0:.*]] = vector.transfer_read %{{.*}}[%[[Y0]], %[[X0]]], %{{.*}} {in_bounds = [true, true]} : memref<64x16xf32>, vector<1x4xf32>
    // CHECK-DAG: %[[Y1:.*]] = affine.apply #[[$MAP2]]()[%[[TX]], %[[TY]], %[[TZ]]]
    //     CHECK: %[[R1:.*]] = vector.transfer_read %{{.*}}[%[[Y1]], %[[X0]]], %{{.*}} {in_bounds = [true, true]} : memref<64x16xf32>, vector<1x4xf32>
    //     CHECK: vector.transfer_write %[[R0]], %{{.*}}[%[[Y0]], %[[X0]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<64x16xf32, 3>
    //     CHECK: vector.transfer_write %[[R1]], %{{.*}}[%[[Y1]], %[[X0]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<64x16xf32, 3>

        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
          ins(%m0 : memref<64x16xf32>)
          outs(%sm0 : memref<64x16xf32, 3>)
          attrs= {__internal_linalg_transform__ = "copy_to_workgroup_memory"} {
          ^bb0(%arg4: f32, %s: f32):  // no predecessors
            linalg.yield %arg4 : f32
        }

    //     CHECK: %[[Y1:.*]] = affine.apply #[[$MAP3]]()[%[[TX]], %[[TY]], %[[TZ]]]
    //     CHECK: %[[R2:.*]] = vector.transfer_read %{{.*}}[%[[Y1]], %[[C0]]], %{{.*}} {in_bounds = [true, true]} : memref<256x4xf32>, vector<1x4xf32>
    //     CHECK: %[[Y2:.*]] = affine.apply #[[$MAP4]]()[%[[TX]], %[[TY]], %[[TZ]]]
    //     CHECK: %[[R3:.*]] = vector.transfer_read %{{.*}}[%[[Y2]], %[[C0]]], %{{.*}} {in_bounds = [true, true]} : memref<256x4xf32>, vector<1x4xf32>
    //     CHECK: vector.transfer_write %[[R2]], %{{.*}}[%[[Y1]], %[[C0]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<256x4xf32, 3>
    //     CHECK: vector.transfer_write %[[R3]], %{{.*}}[%[[Y2]], %[[C0]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<256x4xf32, 3>

        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
          ins(%m1 : memref<256x4xf32>)
          outs(%sm1 : memref<256x4xf32, 3>)
          attrs= {__internal_linalg_transform__ = "copy_to_workgroup_memory"} {
          ^bb0(%arg4: f32, %s: f32):  // no predecessors
            linalg.yield %arg4 : f32
        }

    //     CHECK: %[[X1:.*]] = affine.apply #[[$MAP5]]()[%[[TX]], %[[TY]], %[[TZ]]]
    //     CHECK: %[[R4:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[X1]]], %{{.*}} {in_bounds = [true, true]} : memref<3x512xf32>, vector<1x4xf32>
    //     CHECK: %[[R5:.*]] = vector.transfer_read %{{.*}}[%[[C1]], %[[X1]]], %{{.*}} {in_bounds = [true, true]} : memref<3x512xf32>, vector<1x4xf32>
    //     CHECK: %[[R6:.*]] = vector.transfer_read %{{.*}}[%[[C2]], %[[X1]]], %{{.*}} {in_bounds = [true, true]} : memref<3x512xf32>, vector<1x4xf32>
    //     CHECK: vector.transfer_write %[[R4]], %{{.*}}[%c0, %15] {in_bounds = [true, true]} : vector<1x4xf32>, memref<3x512xf32, 3>
    //     CHECK: vector.transfer_write %[[R5]], %{{.*}}[%c1, %15] {in_bounds = [true, true]} : vector<1x4xf32>, memref<3x512xf32, 3>
    //     CHECK: vector.transfer_write %[[R6]], %{{.*}}[%c2, %15] {in_bounds = [true, true]} : vector<1x4xf32>, memref<3x512xf32, 3>

        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
          ins(%m2 : memref<3x512xf32>)
          outs(%sm2 : memref<3x512xf32, 3>)
          attrs= {__internal_linalg_transform__ = "copy_to_workgroup_memory"} {
          ^bb0(%arg4: f32, %s: f32):  // no predecessors
            linalg.yield %arg4 : f32
        }
        gpu.barrier
        return
      }
    }
  }
}
