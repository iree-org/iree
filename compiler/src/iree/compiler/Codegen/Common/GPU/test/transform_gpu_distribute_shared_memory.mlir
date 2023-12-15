// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule | FileCheck %s

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 32 + s0 floordiv 4)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 4) * 16)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 32 + s0 floordiv 4 + 32)>
// CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1) -> (d0, d1)>

#map1 = affine_map<(d0, d1) -> (d0, d1)>

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>
  ]>
]>
hal.executable private @shared_mem_cpy  {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export @shared_mem_cpy layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 4: index, 1:index]
    } {
    ^bb0(%arg0: !hal.device, %arg1 : index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      memref.global "private" @__shared_memory__ : memref<64x16xf32, #gpu.address_space<workgroup>>
    // CHECK-LABEL: @shared_mem_cpy(
      func.func @shared_mem_cpy(%m0 : memref<64x16xf32>) {
        %c0 = arith.constant 0 : index

        %0 = "affine.apply"(%c0) {map = affine_map<(d0) -> (d0)>} : (index) -> (index)
        %sm0 = memref.get_global @__shared_memory__ : memref<64x16xf32, #gpu.address_space<workgroup>>
        gpu.barrier
    // CHECK-DAG: %[[TX:.*]] = gpu.thread_id x
    // CHECK-DAG: %[[TY:.*]] = gpu.thread_id y
    // CHECK-DAG: %[[TZ:.*]] = gpu.thread_id z

    // CHECK-DAG: %[[Y0:.*]] = affine.apply #[[$MAP0]]()[%[[TX]], %[[TY]], %[[TZ]]]
    // CHECK-DAG: %[[X0:.*]] = affine.apply #[[$MAP1]]()[%[[TX]]]
    //     CHECK: %[[R0:.*]] = vector.transfer_read %{{.*}}[%[[Y0]], %[[X0]]], %{{.*}} {in_bounds = [true, true]} : memref<64x16xf32>, vector<1x4xf32>
    //     CHECK: vector.transfer_write %[[R0]], %{{.*}}[%[[Y0]], %[[X0]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<64x16xf32, #gpu.address_space<workgroup>>
    // CHECK-DAG: %[[Y1:.*]] = affine.apply #[[$MAP2]]()[%[[TX]], %[[TY]], %[[TZ]]]
    //     CHECK: %[[R1:.*]] = vector.transfer_read %{{.*}}[%[[Y1]], %[[X0]]], %{{.*}} {in_bounds = [true, true]} : memref<64x16xf32>, vector<1x4xf32>
    //     CHECK: vector.transfer_write %[[R1]], %{{.*}}[%[[Y1]], %[[X0]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<64x16xf32, #gpu.address_space<workgroup>>

        linalg.generic {indexing_maps = [#map1, #map1],
        iterator_types = ["parallel", "parallel"]}
          ins(%m0 : memref<64x16xf32>)
          outs(%sm0 : memref<64x16xf32, #gpu.address_space<workgroup>>) {
          ^bb0(%arg3: f32, %s: f32):
            linalg.yield %arg3 : f32
        }
    // CHECK: gpu.barrier

    // CHECK: linalg.generic
        linalg.generic {indexing_maps = [#map1, #map1],
        iterator_types = ["parallel", "parallel"]}
          ins(%sm0 : memref<64x16xf32, #gpu.address_space<workgroup>>)
          outs(%sm0 : memref<64x16xf32, #gpu.address_space<workgroup>>) {
          ^bb0(%arg4: f32, %s: f32):
            %add = arith.addf %arg4, %arg4 : f32
            linalg.yield %add : f32
        }

        return
      }
    }
  }

  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.gpu_distribute_shared_memory_copy %func : (!transform.any_op) -> ()
    transform.apply_patterns to %func { 
      transform.apply_patterns.memref.fold_memref_alias_ops
      transform.apply_patterns.canonicalization
      transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.any_op
    transform.iree.apply_cse %func : !transform.any_op
  }
}
