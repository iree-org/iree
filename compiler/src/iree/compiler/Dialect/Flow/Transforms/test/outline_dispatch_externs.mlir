// RUN: iree-opt --allow-unregistered-dialect --split-input-file --iree-flow-outline-dispatch-externs --mlir-print-local-scope %s | FileCheck %s

//      CHECK: hal.executable private @extern_dispatch_0
// CHECK-NEXT:   hal.executable.variant public @a target(<"llvm-cpu", "a">)
// CHECK-SAME:       objects([#hal.executable.object<{path = "a.o"}>])
// CHECK-NEXT:     hal.executable.export public @main ordinal(100)
// CHECK-SAME:         layout(#hal.pipeline.layout<push_constants = 1, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>)
// CHECK-NEXT:     ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
// CHECK-NEXT:       %ok, %value = hal.device.query<%arg0 : !hal.device> key("some" :: "value") : i1, i32
// CHECK-NEXT:       %0 = arith.index_cast %value : i32 to index
// CHECK-NEXT:       hal.return %arg1, %arg2, %0 : index, index, index
//      CHECK:   hal.executable.variant public @b target(<"llvm-cpu", "b">)
// CHECK-SAME:       objects([#hal.executable.object<{path = "b.o"}>])
// CHECK-NEXT:     hal.executable.condition(%arg0: !hal.device) -> i1 {
// CHECK-NEXT:       %ok, %value = hal.device.query<%arg0 : !hal.device> key("some" :: "feature") : i1, i32
// CHECK-NEXT:       hal.return %ok : i1
//      CHECK:     hal.executable.export public @main ordinal(200)
// CHECK-SAME:         layout(#hal.pipeline.layout<push_constants = 1, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>)
// CHECK-NEXT:     ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):

// Demonstrates the full functionality of an extern dispatch op.
// Note that some fields are optional.

// CHECK-LABEL: func.func @dispatchExtern
func.func @dispatchExtern(%arg0: tensor<4xi32>, %arg1: tensor<8xi32>, %arg2: i32) -> tensor<8xi32> {
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  // Dispatch workgroups to the externally defined function "main" in the
  // referenced object files.
  // CHECK: %[[RESULT:.+]] = flow.dispatch {@extern_dispatch_0::@a::@main, @extern_dispatch_0::@b::@main}[%c100, %c50](%arg0, %arg1, %arg2) {
  // CHECK-SAME: hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]
  // CHECK-SAME: } : (tensor<4xi32>, tensor<8xi32>, i32) -> %arg1
  %result = hal.dispatch.extern "main"[%x, %y](%arg0, %arg1, %arg2) : (tensor<4xi32>, tensor<8xi32>, i32) -> %arg1
    // Translates the workload (%x and %y captured above) into an XYZ workgroup
    // count, optionally using device information.
    count(%device: !hal.device, %x_capture: index, %y_capture: index) -> (index, index, index) {
      // Shows how device queries can be used when computing the workgroup count.
      // The device is the one used at runtime.
      %ok, %z_i32 = hal.device.query<%device : !hal.device> key("some" :: "value") : i1, i32
      %z = arith.index_cast %z_i32 : i32 to index
      hal.return %x_capture, %y_capture, %z : index, index, index
    }
    // Must match the external definition.
    layout(#hal.pipeline.layout<push_constants = 1, sets = [
      <0, bindings = [
          <0, storage_buffer, ReadOnly>,
          <1, storage_buffer>
      ]>
    ]>)
    // Optional, automatically inferred if omitted.
    bindings([
      #hal.interface.binding<0, 0>,
      #hal.interface.binding<0, 1>
    ])
    // Can have object references for multiple targets or configurations.
    objects({
      #hal.executable.target<"llvm-cpu", "a"> ordinal(100) = [#hal.executable.object<{path = "a.o"}>],
      #hal.executable.target<"llvm-cpu", "b"> if(%device: !hal.device) -> i1 {
        %ok, %z_i32 = hal.device.query<%device : !hal.device> key("some" :: "feature") : i1, i32
        hal.return %ok : i1
      } ordinal(200) = [#hal.executable.object<{path = "b.o"}>]
    })
  // CHECK: return %[[RESULT]]
  return %result : tensor<8xi32>
}
