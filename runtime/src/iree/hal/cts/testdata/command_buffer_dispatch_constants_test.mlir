// This program writes push constant values into an output buffer.

#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable.source public @executable {
  hal.executable.export public @write_constants ordinal(0) layout(#pipeline_layout) attributes {workgroup_size = [1 : index, 1 : index, 1 : index]} {
  ^bb0(%arg0: !hal.device):
    %c1 = arith.constant 1 : index
    hal.return %c1, %c1, %c1 : index, index, index
  }
  builtin.module {
    func.func @write_constants() {
      %input_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
      %input_1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
      %input_2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
      %input_3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32

      %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<4xi32>

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
