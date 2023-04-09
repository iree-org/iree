// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file --verify-diagnostics | FileCheck %s

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>

hal.executable private @matmul_static_dispatch_0 {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {

    hal.executable.export public @matmul_static_dispatch_0_matmul_1024x4096x12345 ordinal(0) layout(#pipeline_layout) {
    // Check that num_threads is reflected in the workgroup size.
    // CHECK-LABEL: hal.executable.export public @matmul_static_dispatch_0_matmul_1024x4096x12345
    // CHECK-DAG: %[[c32:.*]] = arith.constant 32 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c1_0:.*]] = arith.constant 1 : index
    // CHECK: hal.return %[[c32]], %[[c1]], %[[c1_0]] : index, index, index
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }

    builtin.module {
      func.func @matmul_static_dispatch_0_matmul_1024x4096x12345() {
        // Check that the tiling matches num_threads.
        // CHECK-LABEL: func.func @matmul_static_dispatch_0_matmul_1024x4096x12345
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
^bb1(%variant_op: !pdl.operation):
  %original_matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation

  transform.iree.compute_workgroup_count_region %original_matmul
      num_threads [32]
      ( mapping = [#gpu.block<x>] )
}

// -----
// Test with different num_threads.
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>

hal.executable private @matmul_static_dispatch_1 {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {

    hal.executable.export public @matmul_static_dispatch_1_matmul_1024x4096x12345 ordinal(0) layout(#pipeline_layout) {
    // Check that num_threads is reflected in the workgroup size.
    // CHECK-LABEL: hal.executable.export public @matmul_static_dispatch_1_matmul_1024x4096x12345
    // CHECK-DAG: %[[c32:.*]] = arith.constant 32 : index
    // CHECK-DAG: %[[c16:.*]] = arith.constant 16 : index
    // CHECK-DAG: %[[c16_0:.*]] = arith.constant 16 : index
    // CHECK: hal.return %[[c32]], %[[c16]], %[[c16_0]] : index, index, index
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }

    builtin.module {
      func.func @matmul_static_dispatch_1_matmul_1024x4096x12345() {
        // Check that the tiling matches num_threads.
        // CHECK-LABEL: func.func @matmul_static_dispatch_1_matmul_1024x4096x12345
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
^bb1(%variant_op: !pdl.operation):
  %original_matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation

  transform.iree.compute_workgroup_count_region %original_matmul
      num_threads [32, 16, 16]
      ( mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>] )
}

// -----
// Test with different tile_sizes.
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
// CHECK: #map = affine_map<()[s0] -> (s0 ceildiv 32)>
// CHECK: #map1 = affine_map<()[s0] -> (s0 ceildiv 16)>

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>

hal.executable private @matmul_static_dispatch_1 {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {

    hal.executable.export public @matmul_static_dispatch_1_matmul_1024x4096x12345 ordinal(0) layout(#pipeline_layout) {
    // Check that num_threads is reflected in the workgroup size.
    // CHECK-LABEL: hal.executable.export public @matmul_static_dispatch_1_matmul_1024x4096x12345
    // CHECK-DAG: %0 = affine.apply #map()[%arg1]
    // CHECK-DAG: %1 = affine.apply #map1()[%arg2]
    // CHECK-DAG: %2 = affine.apply #map1()[%arg3]
    // CHECK: hal.return %0, %1, %2 : index, index, index
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }

    builtin.module {
      func.func @matmul_static_dispatch_1_matmul_1024x4096x12345() {
        // Check that the tiling matches num_threads.
        // CHECK-LABEL: func.func @matmul_static_dispatch_1_matmul_1024x4096x12345
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
^bb1(%variant_op: !pdl.operation):
  %original_matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation

  transform.iree.compute_workgroup_count_region %original_matmul
      tile_sizes [32, 16, 16] 
      ( mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>] )
}

// -----
// Test either num_threads or tile_sizes should be present.

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %original_matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation

  // expected-error @below {{'transform.iree.compute_workgroup_count_region' op either num_threads or tile_sizes must be specified}}
  transform.iree.compute_workgroup_count_region %original_matmul
}


