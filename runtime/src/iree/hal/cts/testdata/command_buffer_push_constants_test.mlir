// This program reads a value from an input buffer at the offset specified by
// a push constant then stores that value into an output buffer.

#pipeline_layout = #hal.pipeline.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

hal.executable.source public @executable {
  hal.executable.export public @extract_value layout(#pipeline_layout) {
  ^bb0(%arg0: !hal.device):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @extract_value() {
      // I/O buffers.
      %c0 = arith.constant 0 : index
      %in = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:tensor<4xf32>>
      %out = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:tensor<f32>>

      // Read the input push constant.
      %push_constant_i32 = hal.interface.constant.load[0] : i32

      // Load from the input buffer at the index in the push constant data.
      %push_constant_index = arith.index_castui %push_constant_i32 : i32 to index
      %loaded_value = flow.dispatch.tensor.load %in, offsets = [%push_constant_index], sizes = [1], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<f32>

      // Store into the output buffer.
      flow.dispatch.tensor.store %loaded_value, %out, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>

      return
    }
  }
}
