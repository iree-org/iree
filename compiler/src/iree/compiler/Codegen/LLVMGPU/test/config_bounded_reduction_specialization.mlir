// RUN: iree-opt "%s" --iree-gpu-test-target=sm_80 --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-specialize-exports,builtin.module(iree-codegen-llvmgpu-configuration-pipeline))))' --mlir-print-local-scope | FileCheck "%s"

// Export conditions and the configuration estimate must use the same size.
// Both specialized kernels retain a dynamic reduction and a generic fallback.
#target = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @bounded_reduction {
  hal.executable.variant public @cuda target(#target) {
    // CHECK: hal.executable.export public @sum
    // CHECK: condition(
    // CHECK: arith.constant 128 : index
    // CHECK: fallback(@sum_0)
    hal.executable.export public @sum ordinal(0) layout(#layout)
        count(%device: !hal.device, %n: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    // CHECK: hal.executable.export public @sum_0
    // CHECK: condition(
    // CHECK: arith.constant 1024 : index
    // CHECK: fallback(@sum_0_1)
    // CHECK: hal.executable.export public @sum_0_1
    builtin.module {
      // CHECK-LABEL: func.func @sum()
      // CHECK-SAME: workgroup_size = [32, 1, 1]
      // CHECK: util.assume.int %{{.*}}<umax = 128>
      // CHECK: ins(%{{.*}} : tensor<16x?xf32>)
      // CHECK-LABEL: func.func @sum_0()
      // CHECK-SAME: workgroup_size = [256, 1, 1]
      // CHECK: util.assume.int %{{.*}}<umin = 129, umax = 1024>
      // CHECK: ins(%{{.*}} : tensor<16x?xf32>)
      // CHECK-LABEL: func.func @sum_0_1()
      // CHECK-SAME: workgroup_size = [1024, 1, 1]
      // CHECK: ins(%{{.*}} : tensor<16x?xf32>)
      func.func @sum() {
        %zero = arith.constant 0.0 : f32
        %arg = hal.interface.constant.load layout(#layout) ordinal(0) : i32
        %size = arith.index_castui %arg : i32 to index
        %n = iree_tensor_ext.dispatch.workload.ordinal %size, 0 : index
        %source = hal.interface.binding.subspan layout(#layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n}
        %dest = hal.interface.binding.subspan layout(#layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
        %input = iree_tensor_ext.dispatch.tensor.load %source, offsets = [0, 0], sizes = [16, %n], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n} -> tensor<16x?xf32>
        %empty = tensor.empty() : tensor<16xf32>
        %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
        %sum = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
            iterator_types = ["parallel", "reduction"],
            iree_codegen.specialization_ranges = #util<int.assumption.multi_array[
              [<>, <umax = 128>], [<>, <umin = 129, umax = 1024>]]>}
            ins(%input : tensor<16x?xf32>) outs(%init : tensor<16xf32>) {
          ^bb0(%x: f32, %acc: f32):
            %v = arith.addf %x, %acc : f32
            linalg.yield %v : f32
          } -> tensor<16xf32>
        iree_tensor_ext.dispatch.tensor.store %sum, %dest, offsets = [0], sizes = [16], strides = [1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
        return
      }
    }
  }
}
