// Benchmark executable with two tiny exports using the same HAL ABI layout.
//
// The kernels touch both bindings so dynamic command-buffer paths exercise real
// kernarg user-data pointer fixups, but each launch performs only one load and
// one store so runtime is dominated by dispatch machinery.

#layout_2 = #hal.pipeline.layout<constants = 0, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable.source public @pm4_command_buffer_benchmark {
  hal.executable.export public @model_a ordinal(0) layout(#layout_2) count(%arg0: !hal.device) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    hal.return %c1, %c1, %c1 : index, index, index
  } attributes {workgroup_size = [1 : index, 1 : index, 1 : index]}
  hal.executable.export public @model_b ordinal(1) layout(#layout_2) count(%arg0: !hal.device) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    hal.return %c1, %c1, %c1 : index, index, index
  } attributes {workgroup_size = [1 : index, 1 : index, 1 : index]}
  builtin.module {
    func.func @model_a() {
      %in = hal.interface.binding.subspan layout(#layout_2) binding(0) : memref<4xi32>
      %out = hal.interface.binding.subspan layout(#layout_2) binding(1) : memref<4xi32>
      %c0 = arith.constant 0 : index
      %value = memref.load %in[%c0] : memref<4xi32>
      memref.store %value, %out[%c0] : memref<4xi32>
      return
    }
    func.func @model_b() {
      %in = hal.interface.binding.subspan layout(#layout_2) binding(0) : memref<4xi32>
      %out = hal.interface.binding.subspan layout(#layout_2) binding(1) : memref<4xi32>
      %c1 = arith.constant 1 : index
      %value = memref.load %in[%c1] : memref<4xi32>
      memref.store %value, %out[%c1] : memref<4xi32>
      return
    }
  }
}
