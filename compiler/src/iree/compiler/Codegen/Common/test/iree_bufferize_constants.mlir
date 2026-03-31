// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-codegen-iree-bufferize-constants)" %s | FileCheck %s

// Test that constants are bufferized with gpu.address_space<global> for ROCM targets
// when accessed with dynamic indexing (e.g., split reduction scenario).

#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

// CHECK: memref.global "private" constant @[[GLOBAL:[a-zA-Z0-9_]+]] : memref<4xi32, #gpu.address_space<global>> = dense<[1, 2, 3, 4]>
// CHECK: func.func @constant_with_dynamic_indexing_rocm
// CHECK: %[[CST:.+]] = memref.get_global @[[GLOBAL]] : memref<4xi32, #gpu.address_space<global>>
module attributes {hal.executable.target = #executable_target_rocm} {
  func.func @constant_with_dynamic_indexing_rocm() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index

    // A constant that will be accessed with dynamic indexing.
    %cst = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

    %result = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<i32, #hal.descriptor_type<storage_buffer>>

    // Dynamic indexing pattern (similar to split reduction).
    scf.for %i = %c0 to %c4 step %c1 {
      %extracted = tensor.extract %cst[%i] : tensor<4xi32>
      memref.store %extracted, %result[] : memref<i32, #hal.descriptor_type<storage_buffer>>
    }
    return
  }
}

// -----

// Test with a larger constant (the actual crash scenario).

#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

// CHECK: memref.global "private" constant @[[GLOBAL:[a-zA-Z0-9_]+]] : memref<100x100xi32, #gpu.address_space<global>>
// CHECK: func.func @large_constant_rocm
// CHECK: %[[CST:.+]] = memref.get_global @[[GLOBAL]] : memref<100x100xi32, #gpu.address_space<global>>
module attributes {hal.executable.target = #executable_target_rocm} {
  func.func @large_constant_rocm() {
    %c0 = arith.constant 0 : index

    // Large constant that triggered the original crash.
    %cst = arith.constant dense<1> : tensor<100x100xi32>

    %result = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<i32, #hal.descriptor_type<storage_buffer>>

    %extracted = tensor.extract %cst[%c0, %c0] : tensor<100x100xi32>
    memref.store %extracted, %result[] : memref<i32, #hal.descriptor_type<storage_buffer>>
    return
  }
}

// -----

// Test that non-ROCM targets use default memory space (no gpu.address_space).

#executable_target_llvm = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

// CHECK: memref.global "private" constant @[[GLOBAL:[a-zA-Z0-9_]+]] : memref<4xi32> = dense<[1, 2, 3, 4]>
// CHECK-NOT: #gpu.address_space
// CHECK: func.func @constant_default_cpu
// CHECK: %[[CST:.+]] = memref.get_global @[[GLOBAL]] : memref<4xi32>
module attributes {hal.executable.target = #executable_target_llvm} {
  func.func @constant_default_cpu() {
    %c0 = arith.constant 0 : index

    %cst = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

    %result = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<i32, #hal.descriptor_type<storage_buffer>>

    %extracted = tensor.extract %cst[%c0] : tensor<4xi32>
    memref.store %extracted, %result[] : memref<i32, #hal.descriptor_type<storage_buffer>>
    return
  }
}

// -----

// Test splat constant (optimizes differently but should still work).

#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

// CHECK: memref.global "private" constant @[[GLOBAL:[a-zA-Z0-9_]+]] : memref<10xi32, #gpu.address_space<global>>
// CHECK: func.func @splat_constant_rocm
// CHECK: %[[CST:.+]] = memref.get_global @[[GLOBAL]] : memref<10xi32, #gpu.address_space<global>>
module attributes {hal.executable.target = #executable_target_rocm} {
  func.func @splat_constant_rocm() {
    %c0 = arith.constant 0 : index

    // Splat constant.
    %cst = arith.constant dense<42> : tensor<10xi32>

    %result = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<i32, #hal.descriptor_type<storage_buffer>>

    %extracted = tensor.extract %cst[%c0] : tensor<10xi32>
    memref.store %extracted, %result[] : memref<i32, #hal.descriptor_type<storage_buffer>>
    return
  }
}

// -----

// Test with CUDA target (should also use default, no gpu.address_space).

#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

// CHECK: memref.global "private" constant @[[GLOBAL:[a-zA-Z0-9_]+]] : memref<4xi32> = dense<[1, 2, 3, 4]>
// CHECK-NOT: #gpu.address_space
// CHECK: func.func @constant_cuda
// CHECK: %[[CST:.+]] = memref.get_global @[[GLOBAL]] : memref<4xi32>
module attributes {hal.executable.target = #executable_target_cuda} {
  func.func @constant_cuda() {
    %c0 = arith.constant 0 : index

    %cst = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

    %result = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<i32, #hal.descriptor_type<storage_buffer>>

    %extracted = tensor.extract %cst[%c0] : tensor<4xi32>
    memref.store %extracted, %result[] : memref<i32, #hal.descriptor_type<storage_buffer>>
    return
  }
}
