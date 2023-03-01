// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

// Check that we can specify `num_threads` when lowering
// `workgroup_count_from_dag_root` using
// `transform.iree.tile_to_forall_and_workgroup_count_region`


#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
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
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
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

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %original_matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation

  %forall, %matmul =
    transform.iree.tile_to_forall_and_workgroup_count_region %original_matmul
      num_threads [32]
      ( mapping = [#gpu.block<x>] )
}
