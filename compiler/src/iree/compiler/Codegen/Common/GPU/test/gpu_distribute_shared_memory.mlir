// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-gpu-distribute-shared-memory-copy, fold-memref-alias-ops, canonicalize, cse)))))' %s | FileCheck %s

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 32 + s0 floordiv 4)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 4) * 16)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 32 + s0 floordiv 4 + 32)>
// CHECK-DAG: #[[$MAP3:.*]] = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 128)>
// CHECK-DAG: #[[$MAP4:.*]] = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 128 + 128)>
// CHECK-DAG: #[[$MAP5:.*]] = affine_map<()[s0, s1, s2] -> (s0 * 4 + s1 * 128 + s2 * 512)>

#map0 = affine_map<()[s0, s1, s2] -> (s0 * 4 + s1 * 128 + s2 * 512)>

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @shared_mem_cpy  {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @shared_mem_cpy layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 4: index, 1:index]
    } {
    ^bb0(%arg0: !hal.device, %arg1 : index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
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
    //     CHECK: vector.transfer_write %[[R0]], %{{.*}}[%[[Y0]], %[[X0]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<64x16xf32, 3>
    // CHECK-DAG: %[[Y1:.*]] = affine.apply #[[$MAP2]]()[%[[TX]], %[[TY]], %[[TZ]]]
    //     CHECK: %[[R1:.*]] = vector.transfer_read %{{.*}}[%[[Y1]], %[[X0]]], %{{.*}} {in_bounds = [true, true]} : memref<64x16xf32>, vector<1x4xf32>
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
    //     CHECK: vector.transfer_write %[[R2]], %{{.*}}[%[[Y1]], %[[C0]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<256x4xf32, 3>
    //     CHECK: %[[Y2:.*]] = affine.apply #[[$MAP4]]()[%[[TX]], %[[TY]], %[[TZ]]]
    //     CHECK: %[[R3:.*]] = vector.transfer_read %{{.*}}[%[[Y2]], %[[C0]]], %{{.*}} {in_bounds = [true, true]} : memref<256x4xf32>, vector<1x4xf32>
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
    //     CHECK: vector.transfer_write %[[R4]], %{{.*}}[%[[C0]], %[[X1]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<3x512xf32, 3>
    //     CHECK: %[[R5:.*]] = vector.transfer_read %{{.*}}[%[[C1]], %[[X1]]], %{{.*}} {in_bounds = [true, true]} : memref<3x512xf32>, vector<1x4xf32>
    //     CHECK: vector.transfer_write %[[R5]], %{{.*}}[%[[C1]], %[[X1]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<3x512xf32, 3>
    //     CHECK: %[[R6:.*]] = vector.transfer_read %{{.*}}[%[[C2]], %[[X1]]], %{{.*}} {in_bounds = [true, true]} : memref<3x512xf32>, vector<1x4xf32>
    //     CHECK: vector.transfer_write %[[R6]], %{{.*}}[%[[C2]], %[[X1]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<3x512xf32, 3>

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

// -----

// CHECK-DAG: #[[$OFFSET_MAP:.+]] = affine_map<()[s0] -> (s0 * 4)>

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>]>
]>

hal.executable private @unaligned_shared_memory_copy  {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @unaligned_shared_memory_copy layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 8: index, 1:index]
    } {
    ^bb0(%arg0: !hal.device, %arg1 : index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {

      // CHECK-LABEL: func.func @unaligned_shared_memory_copy
      //  CHECK-SAME: (%[[GLOBAL_MEM:.+]]: memref<56x32xf32, {{.+}}>, %[[SHARED_MEM:.+]]: memref<56x32xf32, 3>)
      func.func @unaligned_shared_memory_copy(
        %global : memref<56x32xf32, strided<[128, 1], offset: ?>>, %shared : memref<56x32xf32, 3>) {

        //  CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
        //  CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
        //  CHECK-DAG:   %[[C56:.+]] = arith.constant 56 : index
        //  CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index

        //  CHECK-DAG:   %[[TID_X:.+]] = gpu.thread_id  x
        //  CHECK-DAG:   %[[TID_Y:.+]] = gpu.thread_id  y

        //      CHECK:   scf.for %[[IV_Y:.+]] = %[[TID_Y]] to %[[C56]] step %[[C8]] {
        //      CHECK:     %[[OFFSET_X:.+]] = affine.apply #[[$OFFSET_MAP]]()[%[[TID_X]]]
        //      CHECK:     scf.for %[[IV_X:.+]] = %[[OFFSET_X]] to %[[C32]] step %[[C128]] {
        //      CHECK:       %[[GLOBAL_SUBVIEW:.+]] = memref.subview %[[GLOBAL_MEM]][%[[IV_Y]], %[[IV_X]]] [1, 4] [1, 1]
        // CHECK-SAME:         : memref<56x32xf32, {{.+}}> to memref<1x4xf32, {{.+}}>
        //      CHECK:       %[[SHARED_SUBVIEW:.+]] = memref.subview %[[SHARED_MEM]][%[[IV_Y]], %[[IV_X]]] [1, 4] [1, 1]
        // CHECK-SAME:         : memref<56x32xf32, 3> to memref<1x4xf32, strided<[32, 1], offset: ?>, 3>
        //      CHECK:       linalg.generic
        // CHECK-SAME:         ins(%[[GLOBAL_SUBVIEW]]
        // CHECK-SAME:         outs(%[[SHARED_SUBVIEW]]
        linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        }
          ins(%global : memref<56x32xf32, strided<[128, 1], offset: ?>>)
          outs(%shared : memref<56x32xf32, 3>)
          attrs =  {__internal_linalg_transform__ = "copy_to_workgroup_memory"} {
        ^bb0(%arg0: f32, %arg1: f32):
          linalg.yield %arg0 : f32
        }
        return
      }
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>]>
]>

hal.executable private @zero_dim_shared_memory_copy  {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @zero_dim_shared_memory_copy layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 8: index, 1:index]
    } {
    ^bb0(%arg0: !hal.device, %arg1 : index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // CHECK-LABEL: func.func @zero_dim_shared_memory_copy
      //  CHECK-SAME: (%[[GLOBAL_MEM:.+]]: memref<f32>, %[[SHARED_MEM:.+]]: memref<f32>)
      func.func @zero_dim_shared_memory_copy(%global : memref<f32>, %shared : memref<f32>) {
        //      CHECK:       linalg.generic
        // CHECK-SAME:         ins(%[[GLOBAL_MEM]]
        // CHECK-SAME:         outs(%[[SHARED_MEM]]
        linalg.generic {
          indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>],
          iterator_types = []
        }
          ins(%global : memref<f32>)
          outs(%shared : memref<f32>)
          attrs =  {__internal_linalg_transform__ = "copy_to_workgroup_memory"} {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        }
        return
      }
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>]>
]>

hal.executable private @zero_dim_shared_memory_copy  {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @zero_dim_shared_memory_copy layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 8: index, 1:index]
    } {
    ^bb0(%arg0: !hal.device, %arg1 : index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @zero_dim_shared_memory_copy(%A: memref<1x32x128xi4>, %B: memref<1x128xf32>, %C: memref<1x128xi4>,
                                             %SM: memref<1x32x128xf32, #gpu.address_space<workgroup>>) {
        linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
            affine_map<(d0, d1, d2) -> (d0, d2)>,
            affine_map<(d0, d1, d2) -> (d0, d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>
          ],
          iterator_types = ["parallel", "parallel", "parallel"]
        }
        ins(%A, %B, %C : memref<1x32x128xi4>, memref<1x128xf32>, memref<1x128xi4>)
        outs(%SM : memref<1x32x128xf32, #gpu.address_space<workgroup>>)
        attrs = {__internal_linalg_transform__ = "copy_to_workgroup_memory"} {
        ^bb0(%in: i4, %in_14: f32, %in_15: i4, %out: f32):
          %19 = arith.extui %in : i4 to i32
          %20 = arith.extui %in_15 : i4 to i32
          %21 = arith.subi %19, %20 : i32
          %22 = arith.sitofp %21 : i32 to f32
          %23 = arith.mulf %22, %in_14 : f32
          linalg.yield %23 : f32
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @zero_dim_shared_memory_copy
//  CHECK-SAME: (%[[A:.+]]: memref<1x32x128xi4>, %{{.+}}: memref<1x128xf32>, %[[C:.+]]: memref<1x128xi4>, %[[SM:.+]]: memref<1x32x128xf32, {{.*}}>)

// CHECK:   %[[A0:.+]] = vector.transfer_read %[[A]]
// CHECK:   %[[C0:.+]] = vector.transfer_read %[[C]]
// CHECK:   %[[A0E:.+]] = arith.extui %[[A0]] : vector<1x1x8xi4> to vector<1x1x8xi32>
// CHECK:   %[[C0E:.+]] = arith.extui %[[C0]] : vector<1x1x8xi4> to vector<1x1x8xi32>
// CHECK:   %[[SUB0:.+]] = arith.subi %[[A0E]], %[[C0E]] : vector<1x1x8xi32>
// CHECK:   %[[EXT0:.+]] = arith.sitofp %[[SUB0]] : vector<1x1x8xi32> to vector<1x1x8xf32>
// CHECK:   %[[MUL0:.+]] = arith.mulf %[[EXT0]], %{{.+}} : vector<1x1x8xf32>
// CHECK:   vector.transfer_write %[[MUL0]], %[[SM]]

// CHECK:   %[[A1:.+]] = vector.transfer_read %[[A]]
// CHECK:   %[[C1:.+]] = vector.transfer_read %[[C]]
// CHECK:   %[[A1E:.+]] = arith.extui %[[A1]] : vector<1x1x8xi4> to vector<1x1x8xi32>
// CHECK:   %[[C1E:.+]] = arith.extui %[[C1]] : vector<1x1x8xi4> to vector<1x1x8xi32>
// CHECK:   %[[SUB1:.+]] = arith.subi %[[A1E]], %[[C1E]] : vector<1x1x8xi32>
// CHECK:   %[[EXT1:.+]] = arith.sitofp %[[SUB1]] : vector<1x1x8xi32> to vector<1x1x8xf32>
// CHECK:   %[[MUL1:.+]] = arith.mulf %[[EXT1]], %{{.+}} : vector<1x1x8xf32>
// CHECK:   vector.transfer_write %[[MUL1]], %[[SM]]

