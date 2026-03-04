// Executable with two entry points for testing multi-entrypoint dispatch.
//   Entry 0 (negate): output[i] = -input[i]
//   Entry 1 (double_it): output[i] = input[i] * 2

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable.source public @executable {
  hal.executable.export public @negate ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    hal.return %c1, %c1, %c1 : index, index, index
  } attributes {workgroup_size = [1 : index, 1 : index, 1 : index]}

  hal.executable.export public @double_it ordinal(1) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    hal.return %c1, %c1, %c1 : index, index, index
  } attributes {workgroup_size = [1 : index, 1 : index, 1 : index]}

  builtin.module {
    func.func @negate() {
      %in = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<4xi32>
      %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<4xi32>

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %zero = arith.constant 0 : i32

      %v0 = memref.load %in[%c0] : memref<4xi32>
      %v1 = memref.load %in[%c1] : memref<4xi32>
      %v2 = memref.load %in[%c2] : memref<4xi32>
      %v3 = memref.load %in[%c3] : memref<4xi32>

      %r0 = arith.subi %zero, %v0 : i32
      %r1 = arith.subi %zero, %v1 : i32
      %r2 = arith.subi %zero, %v2 : i32
      %r3 = arith.subi %zero, %v3 : i32

      memref.store %r0, %out[%c0] : memref<4xi32>
      memref.store %r1, %out[%c1] : memref<4xi32>
      memref.store %r2, %out[%c2] : memref<4xi32>
      memref.store %r3, %out[%c3] : memref<4xi32>

      return
    }

    func.func @double_it() {
      %in = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<4xi32>
      %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<4xi32>

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %two = arith.constant 2 : i32

      %v0 = memref.load %in[%c0] : memref<4xi32>
      %v1 = memref.load %in[%c1] : memref<4xi32>
      %v2 = memref.load %in[%c2] : memref<4xi32>
      %v3 = memref.load %in[%c3] : memref<4xi32>

      %r0 = arith.muli %v0, %two : i32
      %r1 = arith.muli %v1, %two : i32
      %r2 = arith.muli %v2, %two : i32
      %r3 = arith.muli %v3, %two : i32

      memref.store %r0, %out[%c0] : memref<4xi32>
      memref.store %r1, %out[%c1] : memref<4xi32>
      memref.store %r2, %out[%c2] : memref<4xi32>
      memref.store %r3, %out[%c3] : memref<4xi32>

      return
    }
  }
}
