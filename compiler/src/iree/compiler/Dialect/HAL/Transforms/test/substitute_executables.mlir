// RUN: iree-opt --split-input-file %s \
// RUN:   --iree-hal-executable-object-search-path=%S \
// RUN:   --pass-pipeline='builtin.module(iree-hal-substitute-executables{substitutions=executable0=substitute_executables_replacement.mlir,executable1=substitute_executables_replacement.obj})' | \
// RUN: FileCheck %s

// This entire executable should be replaced including the export.
// CHECK: hal.executable private @executable0
hal.executable private @executable0 {
  hal.executable.variant public @variant, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export public @dispatch0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      // CHECK: arith.constant 123
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      // CHECK: func.func @dispatch0
      func.func @dispatch0() {
        // CHECK-NEXT: arith.constant 456
        return
      }
    }
  }
}

// This executable declaration should remain but the inner module should be
// dropped and the object file attached. Note that we just check that the object
// data is loaded and attached but don't bother checking the size as it may
// differ across platforms.
// CHECK: hal.executable private @executable1
hal.executable private @executable1 {
  // CHECK: hal.executable.variant public @variant
  // CHECK-SAME: objects = [#hal.executable.object<{data = dense<[72, 69, 76, 76, 79, 33,
  hal.executable.variant public @variant, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export public @dispatch1 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      // CHECK: arith.constant 100 : index
      %c100 = arith.constant 100 : index
      hal.return %c100, %c100, %c100 : index, index, index
    }
    // CHECK-NOT: builtin.module
    builtin.module {
      func.func @dispatch1() {
        // CHECK-NOT: arith.constant 999
        arith.constant 999 : index
        return
      }
    }
  }
}
