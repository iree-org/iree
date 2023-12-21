// RUN: iree-opt --split-input-file %s \
// RUN:   --pass-pipeline="builtin.module(hal.executable(iree-hal-preprocess-executables-with-pipeline{pipeline=\"builtin.module(iree-codegen-test-executable-preprocessing)\"}))" | \
// RUN: FileCheck %s

// RUN: iree-opt --split-input-file %s \
// RUN:   --pass-pipeline="builtin.module(hal.executable(iree-hal-preprocess-executables-with-tool{command=\"iree-opt --iree-codegen-test-executable-preprocessing\"}))" | \
// RUN: FileCheck %s

// Uses a test pass to simulate an external user pipeline or tool that
// preprocesses executables. Each executable is passed to the tool separately
// and the test pass replaces a constant with a value specified on the target
// config to simulate some kind of target-specific specialization. Only variants
// relevant to the pass should be modified so we throw one in that the pass must
// skip.
//
// A real usage of the preprocessing mechanism would likely change the workgroup
// count function, add additional objects to link, or change the contents of
// the dispatches in meaningful ways.

// CHECK: hal.executable private @executable_a
hal.executable private @executable_a {
  // CHECK: hal.executable.variant public @variant_a
  hal.executable.variant public @variant_a target(#hal.executable.target<"cuda", "cuda-nvptx-fb", {replace_i64 = 123 : i64}>) {
    hal.executable.export public @dispatch_a ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      // CHECK: func.func @dispatch_a
      func.func @dispatch_a() {
        // CHECK-NEXT: arith.constant 123
        %cst = arith.constant 8080 : i64
        return
      }
    }
  }
  // CHECK: hal.executable.variant public @variant_unmodified
  hal.executable.variant public @variant_unmodified target(#hal.executable.target<"cuda", "cuda-nvptx-fb", {}>) {
    hal.executable.export public @dispatch_unmodified ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      // CHECK: func.func @dispatch_unmodified
      func.func @dispatch_unmodified() {
        // CHECK-NEXT: arith.constant 8181
        %cst = arith.constant 8181 : i64
        return
      }
    }
  }
}

// CHECK: hal.executable private @executable_b
hal.executable private @executable_b {
  // CHECK: hal.executable.variant public @variant_b
  hal.executable.variant public @variant_b target(#hal.executable.target<"cuda", "cuda-nvptx-fb", {replace_i64 = 456 : i64}>) {
    hal.executable.export public @dispatch_b ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      // CHECK: func.func @dispatch_b
      func.func @dispatch_b() {
        // CHECK-NEXT: arith.constant 456
        %cst = arith.constant 8282 : i64
        return
      }
    }
  }
}
