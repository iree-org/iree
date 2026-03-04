// Reads scale and offset from push constants, reads input buffer, writes
// output[i] = input[i] * scale + offset.

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable.source public @executable {
  hal.executable.export public @scale_and_offset ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    hal.return %c1, %c1, %c1 : index, index, index
  } attributes {workgroup_size = [1 : index, 1 : index, 1 : index]}
  builtin.module {
    func.func @scale_and_offset() {
      %scale = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
      %offset = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32

      %in = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<4xi32>
      %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<4xi32>

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index

      %v0 = memref.load %in[%c0] : memref<4xi32>
      %v1 = memref.load %in[%c1] : memref<4xi32>
      %v2 = memref.load %in[%c2] : memref<4xi32>
      %v3 = memref.load %in[%c3] : memref<4xi32>

      %s0 = arith.muli %v0, %scale : i32
      %r0 = arith.addi %s0, %offset : i32
      %s1 = arith.muli %v1, %scale : i32
      %r1 = arith.addi %s1, %offset : i32
      %s2 = arith.muli %v2, %scale : i32
      %r2 = arith.addi %s2, %offset : i32
      %s3 = arith.muli %v3, %scale : i32
      %r3 = arith.addi %s3, %offset : i32

      memref.store %r0, %out[%c0] : memref<4xi32>
      memref.store %r1, %out[%c1] : memref<4xi32>
      memref.store %r2, %out[%c2] : memref<4xi32>
      memref.store %r3, %out[%c3] : memref<4xi32>

      return
    }
  }
}
