// Replacement executable for substitute_executables.mlir.
hal.executable private @executable0 {
  hal.executable.variant public @variant target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export public @dispatch0 ordinal(0) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>
    ]>) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
      %c123 = arith.constant 123 : index
      hal.return %c123, %c123, %c123 : index, index, index
    }
    builtin.module {
      func.func @dispatch0() {
        // Here only to give us something to CHECK on.
        arith.constant 456 : index
        return
      }
    }
  }
}
