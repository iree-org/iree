// Tests the iree-benchmark-executable tool against the portable VMVX target.
// Other backends can be tested by using the appropriate compiler flags and
// matching device and executable format flags.
//
// Examples:
//   --iree-hal-target-device=local \
//   --iree-hal-local-target-device-backends=vmvx
//     --device=local-sync or --device=local-task
//     --executable_format=vmvx-bytecode-fb
//   --iree-hal-target-device=local \
//   --iree-hal-local-target-device-backends=llvm-cpu
//     --device=local-sync or --device=local-task
//     --executable_format=embedded-elf-x86_64
//     --executable_format=system-dll-x86_64
//   --iree-hal-target-device=vulkan
//     --device=vulkan
//     --executable_format=vulkan-spirv-fb

// RUN: iree-compile \
// RUN:     --compile-mode=hal-executable \
// RUN:     --iree-hal-target-device=local \
// RUN:     --iree-hal-local-target-device-backends=vmvx \
// RUN:     %s | \
// RUN: iree-benchmark-executable \
// RUN:     --device=local-sync \
// RUN:     --executable_format=vmvx-bytecode-fb \
// RUN:     --executable_file=- \
// RUN:     --entry_point=0 \
// RUN:     --binding=512xf32 \
// RUN:     --binding=512xf32 \
// RUN:     --binding=512xf32 \
// RUN:     --workgroup_count=1,1,1 \
// RUN:     --workgroup_count=512,1,1 | \
// RUN: FileCheck %s

// CHECK: BM_dispatch_1x1x1
// CHECK: BM_dispatch_512x1x1

// lhs * rhs => dst / s0b0 * s0b1 => s0b2
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable.source public @executable {
  hal.executable.export public @elementwise_mul ordinal(0) layout(#pipeline_layout) count(%device: !hal.device) -> (index, index, index) {
    // Unused - the workgroup count is provided to the tool.
    %c1 = arith.constant 1 : index
    hal.return %c1, %c1, %c1 : index, index, index
  } attributes {
    workgroup_size = [1 : index, 1 : index, 1 : index]
  }
  builtin.module {
    func.func @elementwise_mul() {
      %lhs = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(32) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>>
      %rhs = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(32) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>>
      %dst = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(32) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>
      // TODO(#16554): GPU/SPIR-V lowering doesn't handle workgroup size queries.
      // %workgroup_size_x = hal.interface.workgroup.size[0] : index
      %workgroup_size_x = arith.constant 1 : index
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_count_x = hal.interface.workgroup.count[0] : index
      %base_i = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
      %step_i = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
      %end_i = arith.constant 4 : index
      scf.for %i = %base_i to %end_i step %step_i {
        %remaining = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4)>(%i)[%workgroup_size_x]
        %lhs_tile = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [%i], sizes = [%remaining], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<?xf32>
        %rhs_tile = iree_tensor_ext.dispatch.tensor.load %rhs, offsets = [%i], sizes = [%remaining], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<?xf32>
        %dst_init = tensor.empty(%remaining) : tensor<?xf32>
        %dst_tile = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%lhs_tile, %rhs_tile : tensor<?xf32>, tensor<?xf32>)
          outs(%dst_init : tensor<?xf32>) {
          ^bb0(%lhs_value: f32, %rhs_value: f32, %init_value: f32):
            %dst_value = arith.mulf %lhs_value, %rhs_value : f32
            linalg.yield %dst_value : f32
          } -> tensor<?xf32>
        iree_tensor_ext.dispatch.tensor.store %dst_tile, %dst, offsets = [%i], sizes = [%remaining], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>
      }
      return
    }
  }
}
