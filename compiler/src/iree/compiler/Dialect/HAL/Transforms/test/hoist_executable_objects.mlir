// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-hoist-executable-objects)))" %s | FileCheck %s

// Tests that attributes on top-level ops and nested ops are all detected,
// deduplicated, and moved to the variant.

// CHECK: hal.executable public @executable
hal.executable public @executable {
  // CHECK: hal.executable.variant public @backend
  // CHECK-SAME: objects([
  // CHECK-SAME:   #hal.executable.object<{path = "existing_variant.obj"}>,
  // CHECK-SAME:   #hal.executable.object<{path = "extern_fn_common.obj"}>,
  // CHECK-SAME:   #hal.executable.object<{path = "extern_fn_a.obj"}>,
  // CHECK-SAME:   #hal.executable.object<{path = "extern_fn_b.obj"}>,
  // CHECK-SAME:   #hal.executable.object<{path = "nested_common.obj"}>,
  // CHECK-SAME:   #hal.executable.object<{path = "nested_a.obj"}>,
  // CHECK-SAME:   #hal.executable.object<{path = "nested_b.obj"}>
  hal.executable.variant public @backend target(#hal.executable.target<"backend", "format">) objects([
    #hal.executable.object<{path = "existing_variant.obj"}>
  ]) {
    hal.executable.export public @entry0 ordinal(0) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>
    ]>)
    builtin.module {
      // CHECK: func.func private @extern_fn_a
      // CHECK-NOT: hal.executable.objects
      func.func private @extern_fn_a() attributes {
        hal.executable.objects = [
          #hal.executable.object<{path = "extern_fn_common.obj"}>,
          #hal.executable.object<{path = "extern_fn_a.obj"}>
        ]
      }
      // CHECK: func.func private @extern_fn_b
      // CHECK-NOT: hal.executable.objects
      func.func private @extern_fn_b() attributes {
        hal.executable.objects = [
          #hal.executable.object<{path = "extern_fn_common.obj"}>,
          #hal.executable.object<{path = "extern_fn_b.obj"}>
        ]
      }
      func.func @entry0() {
        // CHECK: call @extern_fn_a
        // CHECK-NOT: hal.executable.objects
        call @extern_fn_a() {
          hal.executable.objects = [
            #hal.executable.object<{path = "nested_common.obj"}>,
            #hal.executable.object<{path = "nested_a.obj"}>
          ]
        } : () -> ()
        call @extern_fn_b() {
          // CHECK: call @extern_fn_b
          // CHECK-NOT: hal.executable.objects
          hal.executable.objects = [
            #hal.executable.object<{path = "nested_common.obj"}>,
            #hal.executable.object<{path = "nested_b.obj"}>
          ]
        } : () -> ()
        return
      }
    }
  }
}
