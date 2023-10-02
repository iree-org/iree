// RUN: iree-opt %s --split-input-file \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration})))"\
// RUN:   --iree-spirv-enable-transform-dialect-jit=true | FileCheck %s

hal.executable @matmul {
hal.executable.variant public @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.6,
    [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
    [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, NVIDIA:DiscreteGPU,
    #spirv.resource_limits<
      cooperative_matrix_properties_khr = [
        #spirv.coop_matrix_props_khr<
          a_type = f32, b_type = f32, c_type = f32, k_size = 8,
          m_size = 16, n_size = 16, result_type = f32, acc_sat = false, scope = <Subgroup>>
      ],
      max_compute_shared_memory_size = 49152,
      max_compute_workgroup_invocations = 1024,
      max_compute_workgroup_size = [2147483647, 65535, 65535],
      subgroup_size = 32>
     >}> {
  hal.executable.export public @matmul ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @matmul() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2052x2556xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2556x2052xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2052x2052xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2052, 2556], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2052x2556xf32>> -> tensor<2052x2556xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2556, 2052], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2556x2052xf32>> -> tensor<2556x2052xf32>
      %5 = tensor.empty() : tensor<2052x2052xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2052x2052xf32>) -> tensor<2052x2052xf32>
      %7 = linalg.matmul ins(%3, %4 : tensor<2052x2556xf32>, tensor<2556x2052xf32>) outs(%6 : tensor<2052x2052xf32>) -> tensor<2052x2052xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2052, 2052], strides = [1, 1] : tensor<2052x2052xf32> -> !flow.dispatch.tensor<writeonly:tensor<2052x2052xf32>>
      return
    }
  }
}
}

// CHECK-LABEL: func @matmul

// CHECK: transform.named_sequence

/// The specific vector sizes are tested in the LLVMGPU tests and thus omitted
/// here. This is just to check that masked vectorization is used.
// CHECK-COUNT-3: transform.structured.vectorize

// Verify use of WMMA.
// CHECK: apply_patterns to %{{.*}} {
// CHECK:   transform.apply_patterns.iree.unroll_vectors_gpu_wmma_sync
// CHECK: } : !transform.any_op
// CHECK: transform.iree.vector.vector_to_mma_conversion %{{.*}} {use_wmma}

// Verify asynchronous copy is not used.
// CHECK-NOT: transform.iree.create_async_groups
