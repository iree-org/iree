// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --cse --split-input-file | FileCheck %s

// Check that we can specify `num_threads` when lowering
// `workgroup_count_from_slice` using
// `transform.iree.populate_workgroup_count_region_using_num_threads_slice`


#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>

// Check that num_threads (32) is reflected in the map.
// CHECK: #[[$NUM_THREADS_MAP:.*]] = affine_map<(d0) -> (d0 * 32)>

hal.executable private @matmul_static_dispatch_0 {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {

    hal.executable.export public @matmul_static_dispatch_0_matmul_1024x4096x12345 ordinal(0) layout(#pipeline_layout) {
    // Check that num_threads is reflected in the workgroup size.
    // CHECK-LABEL: hal.executable.export public @matmul_static_dispatch_0_matmul_1024x4096x12345
    // CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: hal.return %[[C32]], %[[C1]], %[[C1]] : index, index, index
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }

    builtin.module {
      func.func @matmul_static_dispatch_0_matmul_1024x4096x12345() {
        // Check that the tiling matches num_threads.
        // CHECK-LABEL: func.func @matmul_static_dispatch_0_matmul_1024x4096x12345
        // CHECK: = scf.forall (%[[IV:.*]]) in (32) shared_outs(%{{.*}}) -> (tensor<1024x4096xf32>) {
        // CHECK: %[[OFFSET:.*]] = affine.apply #[[$NUM_THREADS_MAP]](%[[IV]])
        // CHECK: %extracted_slice = tensor.extract_slice %{{.*}}[%[[OFFSET]], 0] [32, 12345] [1, 1] : tensor<1024x12345xf32> to tensor<32x12345xf32>
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x12345xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<12345x4096xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1024x4096xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 12345], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x12345xf32>> -> tensor<1024x12345xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [12345, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<12345x4096xf32>> -> tensor<12345x4096xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1024, 4096], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<1024x4096xf32>> -> tensor<1024x4096xf32>
        %6 = linalg.matmul ins(%3, %4 : tensor<1024x12345xf32>, tensor<12345x4096xf32>) outs(%5 : tensor<1024x4096xf32>) -> tensor<1024x4096xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [1024, 4096], strides = [1, 1] : tensor<1024x4096xf32> -> !flow.dispatch.tensor<readwrite:tensor<1024x4096xf32>>
        return
      }
    }
  }
}

transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  %original_matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op

  %matmul, %forall =
    transform.structured.tile_using_forall %original_matmul num_threads [32]
      ( mapping = [#gpu.block<x>] )
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Late canonicalizations to cleanup and pass the checks.
  // Needs to occur on the whole variant to perform cse on the workgroup_count region
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall
    : (!transform.any_op) -> ()
}

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>

hal.executable private @matmul_static_dispatch_0 {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
  
    hal.executable.export public @elementwise_out_of_order_block_id ordinal(0) layout(#pipeline_layout) {
    // Check that num_threads is consistent with the specified mapping
    // CHECK-LABEL: hal.executable.export public @elementwise_out_of_order_block_id

    // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
    // CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
    // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
    // CHECK: hal.return %[[C3]], %[[C5]], %[[C8]] : index, index, index
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }   
  
    builtin.module {
      func.func @elementwise_out_of_order_block_id() {
        // CHECK-LABEL: func.func @elementwise_out_of_order_block_id
        // CHECK: = scf.forall (%[[IV:.*]]) in (3, 5, 8) shared_outs(%{{.*}}) -> (tensor<3x5x8xf32>) {
        // CHECK: } {mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>]}
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x5x8xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<3x5x8xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [3, 5, 8], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x5x8xf32>> -> tensor<3x5x8xf32>
        %empty = tensor.empty() : tensor<3x5x8xf32>
        %3 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                           affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
          iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%2 : tensor<3x5x8xf32>) outs(%empty : tensor<3x5x8xf32>) {   
          ^bb0(%in: f32, %in_0: f32):
            %4 = math.sqrt %in : f32 
            linalg.yield %4 : f32 
          } -> tensor<3x5x8xf32>
        flow.dispatch.tensor.store %3, %1, offsets = [0, 0, 0], sizes = [3, 5, 8], strides = [1, 1, 1] : tensor<3x5x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<3x5x8xf32>>
        return
      }   
    }   
  }
}

transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  %1 = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %tiled_op, %forall_op = transform.structured.tile_using_forall %1   num_threads [] tile_sizes [1, 1, 1](mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>]): (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_op : (!transform.any_op) -> ()
}

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>

hal.executable private @matmul_static_dispatch_0 {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
  
    hal.executable.export public @vecadd2d_dispatch_0_generic_9x512_f32 ordinal(0) layout(#pipeline_layout) {
    // Check that num_threads is consistent with the specified mapping
    // CHECK-LABEL: hal.executable.export public @vecadd2d_dispatch_0_generic_9x512_f32

    // CHECK-DAG: %[[C171:.*]] = arith.constant 171 : index
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
    // CHECK: hal.return %[[C171]], %[[C1]], %[[C2]] : index, index, index
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }   
  
    builtin.module {
      func.func @vecadd2d_dispatch_0_generic_9x512_f32() {
        %c18432 = arith.constant 18432 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c18432) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<9x512xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512x9xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<512x9xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [9, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<9x512xf32>> -> tensor<9x512xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 9], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x9xf32>> -> tensor<512x9xf32>
        %5 = tensor.empty() : tensor<512x9xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%3, %4 : tensor<9x512xf32>, tensor<512x9xf32>) outs(%5 : tensor<512x9xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %7 = arith.addf %in, %in_0 : f32
          linalg.yield %7 : f32
        } -> tensor<512x9xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [512, 9], strides = [1, 1] : tensor<512x9xf32> -> !flow.dispatch.tensor<writeonly:tensor<512x9xf32>>
        return
      }
    }
  }
}

transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  %1 = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %tiled_op, %forall_op = transform.structured.tile_using_forall %1   num_threads [] tile_sizes [5, 3](mapping = [#gpu.block<z>, #gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_op : (!transform.any_op) -> ()
}
