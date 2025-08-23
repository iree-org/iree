// NOTE: this test *should* have a lot of different exports with different
// layouts, but our Vulkan compiler target currently doesn't handle splitting
// MLIR modules into multiple SPIR-V modules so we can't. For now we have one
// mostly representative export that can be used to test that metadata is being
// encoded/decoded appropriately but unfortunately can't test exhaustively.
// As we get more complex reflection APIs we will need to fix the SPIR-V
// pipeline.

hal.executable.source public @executable {
  hal.executable.export public @export0 ordinal(0) layout(#hal.pipeline.layout<constants = 2, bindings = [
    #hal.pipeline.binding<storage_buffer>,
    #hal.pipeline.binding<storage_buffer>
  ]>) count(%arg0: !hal.device, %dim: index) -> (index, index, index) {
    hal.return %dim, %dim, %dim : index, index, index
  }
  builtin.module {
    func.func @export0() {
      return
    }
  }
}
