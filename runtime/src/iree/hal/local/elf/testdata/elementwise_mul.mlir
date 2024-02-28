// An elementwise multiply of two 4xf32 values:
//   %dst = arith.mulf %lhs, %rhs : tensor<4xf32>
// This program could be that simple however this example demonstrates how to
// perform workgroup-level tiling.
//
// Can be run with:
// iree/hal/local/executable_library_benchmark \
//    --executable_format=embedded-elf \
//    --executable_file=iree/hal/local/elf/testdata/elementwise_mul_x86_64.so \
//    --entry_point=0 \
//    --workgroup_count_x=1 \
//    --workgroup_count_y=1 \
//    --workgroup_count_z=1 \
//    --workgroup_size_x=1 \
//    --workgroup_size_y=1 \
//    --workgroup_size_z=1 \
//    --binding=4xf32=1,2,3,4 \
//    --binding=4xf32=100,200,300,400 \
//    --binding=4xf32=0,0,0,0

// lhs * rhs => dst / s0b0 * s0b1 => s0b2
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

// A single executable source definition is allowed per translation in this mode
// as linking and multi-executable embedding support requires our host-side IR.
hal.executable.source public @ex {
  // Exported functions are declared with the layout they use and may optionally
  // contain other information - though when hand-authoring that's usually
  // omitted.
  //
  // The ordinal is used to specify the entry point on command line tools and
  // must be unique across all entry points within the same executable.
  hal.executable.export public @elementwise_mul ordinal(0) layout(#pipeline_layout)

  // The inner module defining the executable. This may have any number of
  // private functions and only those with declared entry points will be
  // exported.
  builtin.module {
    func.func @elementwise_mul() {
      %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) : !flow.dispatch.tensor<readonly:tensor<4xf32>>
      %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) : !flow.dispatch.tensor<readonly:tensor<4xf32>>
      %dst = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) : !flow.dispatch.tensor<writeonly:tensor<4xf32>>
      %workgroup_size_x = hal.interface.workgroup.size[0] : index
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_count_x = hal.interface.workgroup.count[0] : index
      %base_i = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
      %step_i = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
      %end_i = arith.constant 4 : index
      scf.for %i = %base_i to %end_i step %step_i {
        %remaining = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4)>(%i)[%workgroup_size_x]
        %lhs_tile = flow.dispatch.tensor.load %lhs, offsets = [%i], sizes = [%remaining], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<?xf32>
        %rhs_tile = flow.dispatch.tensor.load %rhs, offsets = [%i], sizes = [%remaining], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<?xf32>
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
        flow.dispatch.tensor.store %dst_tile, %dst, offsets = [%i], sizes = [%remaining], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:tensor<4xf32>>
      }
      return
    }
  }
}
