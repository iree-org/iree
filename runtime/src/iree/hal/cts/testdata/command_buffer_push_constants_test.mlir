// This program writes push constant values into an output buffer.

#pipeline_layout = #hal.pipeline.layout<push_constants = 4, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>
  ]>
]>

hal.executable.source public @executable {
  hal.executable.export public @write_push_constants layout(#pipeline_layout) attributes {workgroup_size = [1 : index, 1 : index, 1 : index]} {
  ^bb0(%arg0: !hal.device):
    %c1 = arith.constant 1 : index
    hal.return %c1, %c1, %c1 : index, index, index
  }
  builtin.module {
    func.func @write_push_constants() {
      %input_0 = hal.interface.constant.load[0] : i32
      %input_1 = hal.interface.constant.load[1] : i32
      %input_2 = hal.interface.constant.load[2] : i32
      %input_3 = hal.interface.constant.load[3] : i32

      %out = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4xi32>

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      memref.store %input_0, %out[%c0] : memref<4xi32>
      memref.store %input_1, %out[%c1] : memref<4xi32>
      memref.store %input_2, %out[%c2] : memref<4xi32>
      memref.store %input_3, %out[%c3] : memref<4xi32>

      return
    }
  }
}
