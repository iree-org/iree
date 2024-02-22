// RUN: iree-opt --split-input-file --iree-hal-prune-executables %s | FileCheck %s

// Tests that an executable with no references is dropped.
// The MLIR SymbolDCE pass will do this for us but it makes more sense to do it
// as part of this pass for consistency (after running no executables/variants/
// exports that are unused exist).

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>
  ]>
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

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>
  ]>
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
