// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @tensorImportStatic
func.func @tensorImportStatic(%arg0: !hal.buffer_view) -> tensor<5xi32> {
  // CHECK: hal.tensor.import %arg0 "hello" : !hal.buffer_view -> tensor<5xi32>
  %0 = hal.tensor.import %arg0 "hello" : !hal.buffer_view -> tensor<5xi32>
  return %0 : tensor<5xi32>
}

// -----

// CHECK-LABEL: @tensorImportDynamic
func.func @tensorImportDynamic(%arg0: !hal.buffer_view, %arg1: index) -> tensor<?x3xi32> {
  // CHECK: hal.tensor.import %arg0 : !hal.buffer_view -> tensor<?x3xf32> as tensor<?x3xi32>{%arg1}
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<?x3xf32> as tensor<?x3xi32>{%arg1}
  return %0 : tensor<?x3xi32>
}

// -----

// CHECK-LABEL: @tensorImportAsync
func.func @tensorImportAsync(%arg0: !hal.buffer_view, %arg1: !hal.fence) -> tensor<5xi32> {
  // CHECK: hal.tensor.import wait(%arg1) => %arg0 : !hal.buffer_view -> tensor<5xi32>
  %0 = hal.tensor.import wait(%arg1) => %arg0 : !hal.buffer_view -> tensor<5xi32>
  return %0 : tensor<5xi32>
}

// -----

// CHECK-LABEL: @tensorExportDynamic
func.func @tensorExportDynamic(%arg0: tensor<?x3xi32>, %arg1: index) -> !hal.buffer_view {
  // CHECK: hal.tensor.export %arg0 "goodbye" : tensor<?x3xf32> as tensor<?x3xi32>{%arg1} -> !hal.buffer_view
  %0 = hal.tensor.export %arg0 "goodbye" : tensor<?x3xf32> as tensor<?x3xi32>{%arg1} -> !hal.buffer_view
  return %0 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @tensorExportInPlace
func.func @tensorExportInPlace(%arg0: tensor<?x3xi32>, %arg1: index, %arg2: !hal.buffer) -> !hal.buffer_view {
  // CHECK: hal.tensor.export %arg0 into(%arg2 : !hal.buffer) : tensor<?x3xf32> as tensor<?x3xi32>{%arg1} -> !hal.buffer_view
  %0 = hal.tensor.export %arg0 into(%arg2 : !hal.buffer) : tensor<?x3xf32> as tensor<?x3xi32>{%arg1} -> !hal.buffer_view
  return %0 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @tensorBarrier
func.func @tensorBarrier(%arg0: tensor<3xf32>, %arg1: tensor<4xf32>, %arg2: !hal.fence) -> (tensor<3xf32>, tensor<4xf32>) {
  // CHECK: :2 = hal.tensor.barrier join(%arg0, %arg1 : tensor<3xf32>, tensor<4xf32>) => %arg2 : !hal.fence
  %0:2 = hal.tensor.barrier join(%arg0, %arg1 : tensor<3xf32>, tensor<4xf32>) => %arg2 : !hal.fence
  return %0#0, %0#1 : tensor<3xf32>, tensor<4xf32>
}

// -----

// Demonstrates the full functionality of an extern dispatch op.
// Note that some fields are optional.

// CHECK-LABEL: func.func @dispatchExtern
func.func @dispatchExtern(%arg0: tensor<4xi32>, %arg1: tensor<8xi32>, %arg2: i32) -> tensor<8xi32> {
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  // Dispatch workgroups to the externally defined function "main" in the
  // referenced object files.
  %0 = hal.dispatch.extern "main"[%x, %y](%arg0, %arg1, %arg2) : (tensor<4xi32>, tensor<8xi32>, i32) -> %arg1
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
    objects(#hal.executable.objects<{
      #hal.executable.target<"llvm-cpu", "a"> = [#hal.executable.object<{path = "a.o"}>],
      #hal.executable.target<"llvm-cpu", "b"> = [#hal.executable.object<{path = "b.o"}>]
    }>)
  return %0 : tensor<8xi32>
}
