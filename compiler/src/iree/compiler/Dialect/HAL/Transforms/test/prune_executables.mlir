// RUN: iree-opt --split-input-file --iree-hal-prune-executables %s | FileCheck %s

// Tests that an executable with no references is dropped.
// The MLIR SymbolDCE pass will do this for us but it makes more sense to do it
// as part of this pass for consistency (after running no executables/variants/
// exports that are unused exist).

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

// Should be removed as there are no uses.
// CHECK-NOT: hal.executable private @unused_exe
hal.executable private @unused_exe {
  hal.executable.variant public @unused_variant target(<"backend", "format">) {
    hal.executable.export public @unused_export layout(#pipeline_layout)
  }
}

// Should not be removed as it's public.
// CHECK: hal.executable public @unused_public_exe
hal.executable public @unused_public_exe {}

// Should not be removed as there's a use.
// CHECK: hal.executable private @used_exe
hal.executable private @used_exe {}

util.func private @user(%cond: i1) {
  scf.if %cond {
    // Ensure we do full walks to find nested ops.
    util.optimization_barrier {some.ref = @used_exe} %cond : i1
    scf.yield
  }
  util.return
}

// -----

// Tests that a variant with no references is dropped.

hal.executable private @exe {
  // CHECK-NOT: hal.executable.variant public @unused_variant
  hal.executable.variant public @unused_variant target(<"backend", "format">) {}
  // CHECK: hal.executable.variant public @used_variant
  hal.executable.variant public @used_variant target(<"backend", "format">) {}
}
util.func private @user() attributes {
  // Ensure we walk into nested attrs.
  some.ref = {
    key = [@exe::@used_variant]
  }
}

// -----

// Tests that an export with no references is dropped.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @exe {
  hal.executable.variant public @variant target(<"backend", "format">) {
    // CHECK-NOT: hal.executable.export public @unused_export
    hal.executable.export public @unused_export layout(#pipeline_layout)
    // CHECK: hal.executable.export public @used_export
    hal.executable.export public @used_export layout(#pipeline_layout)
  }
}
util.func private @user() attributes {
  some.ref = @exe::@variant::@used_export
}

// -----

// Tests that an export that is used as a fallback is not dropped but one only
// used as the fallback for a dropped export is.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @exe {
  hal.executable.variant public @variant target(<"backend", "format">) {
    // CHECK-NOT: hal.executable.export public @unused_export
    hal.executable.export public @unused_export layout(#pipeline_layout) condition(%device: !hal.device) -> i1 {
      %false = arith.constant 0 : i1
      hal.return %false : i1
    } fallback(@unused_fallback_export)
    // CHECK-NOT: hal.executable.export public @unused_fallback_export
    hal.executable.export public @unused_fallback_export layout(#pipeline_layout)
    // CHECK: hal.executable.export public @used_export
    hal.executable.export public @used_export layout(#pipeline_layout) condition(%device: !hal.device) -> i1 {
      %false = arith.constant 0 : i1
      hal.return %false : i1
    } fallback(@used_fallback_export)
    // CHECK: hal.executable.export public @used_fallback_export
    hal.executable.export public @used_fallback_export layout(#pipeline_layout)
  }
}
util.func private @user() attributes {
  some.ref = @exe::@variant::@used_export
}
