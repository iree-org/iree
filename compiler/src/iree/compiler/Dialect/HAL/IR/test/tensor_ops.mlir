// RUN: iree-opt --split-input-file --mlir-print-local-scope %s | iree-opt --split-input-file --mlir-print-local-scope | FileCheck %s

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
  // CHECK-DAG: %[[WORKLOAD_X:.+]] = arith.constant 100
  %workload_x = arith.constant 100 : index
  // CHECK-DAG: %[[WORKLOAD_Y:.+]] = arith.constant 50
  %workload_y = arith.constant 50 : index

  // Dispatch workgroups to the externally defined function "main" in the
  // referenced object files with the ordinal specified per object group.
  // CHECK: %[[RESULT:.+]] = hal.dispatch.extern "main"[%[[WORKLOAD_X]], %[[WORKLOAD_Y]]](%arg0, %arg1, %arg2) : (tensor<4xi32>, tensor<8xi32>, i32) -> %arg1
  %0 = hal.dispatch.extern "main"[%workload_x, %workload_y](%arg0, %arg1, %arg2) : (tensor<4xi32>, tensor<8xi32>, i32) -> %arg1
    // Translates the workload (%x and %y captured above) into an XYZ workgroup
    // count, optionally using device information.
    // CHECK: count(%[[DEVICE:.+]]: !hal.device, %[[X_CAPTURE:.+]]: index, %[[Y_CAPTURE:.+]]: index) -> (index, index, index) {
    count(%device: !hal.device, %x_capture: index, %y_capture: index) -> (index, index, index) {
      // Shows how device queries can be used when computing the workgroup count.
      // The device is the one used at runtime.
      // CHECK: = hal.device.query<%[[DEVICE]] : !hal.device>
      %ok, %z_i32 = hal.device.query<%device : !hal.device> key("some" :: "value") : i1, i32
      %z = arith.index_cast %z_i32 : i32 to index
      hal.return %x_capture, %y_capture, %z : index, index, index
    }
    // Must match the external definition.
    // CHECK: layout(<push_constants = 1, sets =
    layout(#hal.pipeline.layout<push_constants = 1, sets = [
      <0, bindings = [
          <0, storage_buffer, ReadOnly>,
          <1, storage_buffer>
      ]>
    ]>)
    // Optional, automatically inferred if omitted.
    // CHECK: bindings([#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>])
    bindings([
      #hal.interface.binding<0, 0>,
      #hal.interface.binding<0, 1>
    ])
    // Can have object references for multiple targets or configurations.
    // CHECK: objects({
    objects({
      // CHECK: #hal.executable.target<"llvm-cpu", "a"> ordinal(100) = [#hal.executable.object<{path = "a.o"}>]
      #hal.executable.target<"llvm-cpu", "a"> ordinal(100) = [#hal.executable.object<{path = "a.o"}>],
      // CHECK: #hal.executable.target<"llvm-cpu", "b"> if(%[[B_DEVICE:.+]]: !hal.device) -> i1 {
      #hal.executable.target<"llvm-cpu", "b"> if(%device: !hal.device) -> i1 {
        // CHECK: = hal.device.query<%[[B_DEVICE]] : !hal.device>
        %ok, %z_i32 = hal.device.query<%device : !hal.device> key("some" :: "feature_b") : i1, i32
        hal.return %ok : i1
      // CHECK: } ordinal(200) = [#hal.executable.object<{path = "b.o"}>]
      } ordinal(200) = [#hal.executable.object<{path = "b.o"}>],
      // CHECK: #hal.executable.target<"llvm-cpu", "c"> if(%[[C_DEVICE:.+]]: !hal.device) -> i1 {
      #hal.executable.target<"llvm-cpu", "c"> if(%device: !hal.device) -> i1 {
        // CHECK: = hal.device.query<%[[C_DEVICE]] : !hal.device>
        %ok, %z_i32 = hal.device.query<%device : !hal.device> key("some" :: "feature_c") : i1, i32
        hal.return %ok : i1
      // CHECK: } ordinal(300) = [#hal.executable.object<{path = "c.o"}>]
      } ordinal(300) = [#hal.executable.object<{path = "c.o"}>]
    })
  // CHECK: return %[[RESULT]]
  return %0 : tensor<8xi32>
}
