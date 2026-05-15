// Source module with hal.executable.source (extern kernel dispatch).
// Used by iree-link-executable-source.mlir.

module @module_d {
  // Extern kernel executable (e.g., a pre-compiled GPU kernel object).
  hal.executable.source private @extern_kernel attributes {
    objects = #hal.executable.objects<{}>
  } {
    hal.executable.export public @entry ordinal(0)
        layout(#hal.pipeline.layout<constants = 2, bindings = [
          #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer>
        ]>) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    } attributes {workgroup_size = [64 : index, 1 : index, 1 : index]}
  }

  // Function that dispatches to the extern kernel.
  util.func public @transform(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %c4 = arith.constant 4 : i32
    %c1 = arith.constant 1 : index
    %0 = flow.dispatch @extern_kernel::@entry(%c4, %arg0)
        : (i32, tensor<4xf32>) -> tensor<4xf32>
    util.return %0 : tensor<4xf32>
  }
}
