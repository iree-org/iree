// Writes an indirect dispatch parameter buffer containing workgroup_count =
// [4, 1, 1]. Used by CTS tests that need a dispatch to produce the parameters
// consumed by a later dynamic indirect dispatch.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable.source public @executable {
  hal.executable.export public @write_dispatch_params ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    hal.return %c1, %c1, %c1 : index, index, index
  } attributes {workgroup_size = [1 : index, 1 : index, 1 : index]}
  builtin.module {
    func.func @write_dispatch_params() {
      %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<3xi32>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %count_x = arith.constant 4 : i32
      %count_y = arith.constant 1 : i32
      %count_z = arith.constant 1 : i32
      memref.store %count_x, %out[%c0] : memref<3xi32>
      memref.store %count_y, %out[%c1] : memref<3xi32>
      memref.store %count_z, %out[%c2] : memref<3xi32>
      return
    }
  }
}
