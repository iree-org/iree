// Replacement executable for substitute_executables.mlir.
hal.executable private @executable0 {
  hal.executable.variant public @variant, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
    hal.executable.export public @dispatch0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
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
