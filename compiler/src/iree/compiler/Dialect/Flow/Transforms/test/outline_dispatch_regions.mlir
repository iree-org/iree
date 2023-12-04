// RUN: iree-opt --allow-unregistered-dialect --split-input-file --iree-flow-outline-dispatch-regions --mlir-print-local-scope %s | FileCheck %s

//      CHECK: flow.executable private @staticShapeDispatch_dispatch_0
// CHECK-NEXT:   flow.executable.export public @staticShapeDispatch_dispatch_0
//      CHECK: func.func @staticShapeDispatch_dispatch_0(
// CHECK-SAME:     %[[ARG:.+]]: !flow.dispatch.tensor<readonly:tensor<8x4xf32>>,
// CHECK-SAME:     %[[RET:.+]]: !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>) {
//  CHECK-DAG:   %[[ARG_VALUE:.+]] = flow.dispatch.tensor.load %[[ARG]], {{.*}} : !flow.dispatch.tensor<readonly:tensor<8x4xf32>> -> tensor<8x4xf32>
// CHECK-NEXT:   %[[RET_VALUE:.+]] = "test.sink"(%[[ARG_VALUE]]) : (tensor<8x4xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   flow.dispatch.tensor.store %[[RET_VALUE]], %[[RET]], {{.*}} : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @staticShapeDispatch(
// CHECK-SAME: %[[ARG0:.+]]: tensor<8x4xf32>)
func.func @staticShapeDispatch(%arg0 : tensor<8x4xf32>) -> tensor<4x8xf32> {
  // CHECK-DAG: %[[X:.+]] = arith.constant 100
  %x = arith.constant 100 : index
  // CHECK-DAG: %[[Y:.+]] = arith.constant 50
  %y = arith.constant 50 : index
  // CHECK: %[[RET:.+]] = flow.dispatch @staticShapeDispatch_dispatch_0::@staticShapeDispatch_dispatch_0[
  // CHECK-SAME: %[[X]], %[[Y]]
  // CHECK-SAME: ](%[[ARG0]]) : (tensor<8x4xf32>) -> tensor<4x8xf32>
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<8x4xf32>) -> tensor<4x8xf32> = (
    %arg: !flow.dispatch.tensor<readonly:tensor<8x4xf32>>, %ret: !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
  ) {
    %arg_value = flow.dispatch.tensor.load %arg, offsets=[0, 0], sizes=[8, 4], strides=[1, 1] : !flow.dispatch.tensor<readonly:tensor<8x4xf32>> -> tensor<8x4xf32>
    %ret_value = "test.sink"(%arg_value) : (tensor<8x4xf32>) -> (tensor<4x8xf32>)
    flow.dispatch.tensor.store %ret_value, %ret,  offsets=[0, 0], sizes=[4, 8], strides=[1, 1] : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
    flow.return
  }
  // CHECK-NEXT: return %[[RET]]
  return %0 : tensor<4x8xf32>
}

// -----

//      CHECK: flow.executable private @dispatchFnMuli_dispatch_0
// CHECK-NEXT:   flow.executable.export public @dispatchFnMuli_dispatch_0
//      CHECK: func.func @dispatchFnMuli_dispatch_0(

//      CHECK: flow.executable private @dispatchFnMuli_dispatch_1
// CHECK-NEXT:   flow.executable.export public @dispatchFnMuli_dispatch_1
//      CHECK: func.func @dispatchFnMuli_dispatch_1(

// CHECK-LABEL: func.func @dispatchFnMuli(
// CHECK-SAME: %[[ARG0:.+]]: tensor<8x4xf32>)
func.func @dispatchFnMuli(%arg0 : tensor<8x4xf32>) -> tensor<8x4xf32> {
  // CHECK-DAG: %[[X:.+]] = arith.constant 100
  %x = arith.constant 100 : index
  // CHECK-DAG: %[[Y:.+]] = arith.constant 50
  %y = arith.constant 50 : index
  // CHECK: %[[RET0:.+]] = flow.dispatch @dispatchFnMuli_dispatch_0::@dispatchFnMuli_dispatch_0[
  // CHECK-SAME: %[[X]], %[[Y]]
  // CHECK-SAME: ](%[[ARG0]]) : (tensor<8x4xf32>) -> tensor<4x8xf32>
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<8x4xf32>) -> (tensor<4x8xf32>) = (
    %arg: !flow.dispatch.tensor<readonly:tensor<8x4xf32>>, %ret: !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
  ) {
    %arg_value = flow.dispatch.tensor.load %arg, offsets=[0, 0], sizes=[8, 4], strides=[1, 1] : !flow.dispatch.tensor<readonly:tensor<8x4xf32>> -> tensor<8x4xf32>
    %ret_value = "test.sink1"(%arg_value) : (tensor<8x4xf32>) -> (tensor<4x8xf32>)
    flow.dispatch.tensor.store %ret_value, %ret, offsets=[0, 0], sizes=[4, 8], strides=[1, 1] : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
    flow.return
  }
  // CHECK: %[[RET1:.+]] = flow.dispatch @dispatchFnMuli_dispatch_1::@dispatchFnMuli_dispatch_1[
  // CHECK-SAME: %[[Y]], %[[X]]
  // CHECK-SAME: ](%[[RET0]]) : (tensor<4x8xf32>) -> tensor<8x4xf32>
  %1 = flow.dispatch.workgroups[%y, %x](%0) : (tensor<4x8xf32>) -> (tensor<8x4xf32>) = (
    %arg: !flow.dispatch.tensor<readonly:tensor<4x8xf32>>, %ret: !flow.dispatch.tensor<writeonly:tensor<8x4xf32>>
  ) {
    %arg_value = flow.dispatch.tensor.load %arg, offsets=[0, 0], sizes=[4, 8], strides=[1, 1] : !flow.dispatch.tensor<readonly:tensor<4x8xf32>> -> tensor<8x4xf32>
    %ret_value = "test.sink2"(%arg_value) : (tensor<8x4xf32>) -> (tensor<8x4xf32>)
    flow.dispatch.tensor.store %ret_value, %ret, offsets=[0, 0], sizes=[8, 4], strides=[1, 1] : tensor<8x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x4xf32>>
    flow.return
  }
  // CHECK-NEXT: return %[[RET1]]
  return %1 : tensor<8x4xf32>
}

// -----

// CHECK: flow.executable private @dispatchFn1_dispatch_0

// CHECK-LABEL: func.func @dispatchFn1
func.func @dispatchFn1(%arg0 : tensor<8x4xf32>) -> tensor<4x8xf32> {
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  // CHECK: flow.dispatch @dispatchFn1_dispatch_0::@dispatchFn1_dispatch_0
  // CHECK-SAME: stream.affinity = #hal.affinity.queue<[0]>
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<8x4xf32>) -> (tensor<4x8xf32>) attributes {
     stream.affinity = #hal.affinity.queue<[0]>
  } = (
    %arg: !flow.dispatch.tensor<readonly:tensor<8x4xf32>>, %ret: !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
  ) {
    flow.return
  }
  return %0 : tensor<4x8xf32>
}

// CHECK: flow.executable private @dispatchFn2_dispatch_0

// CHECK-LABEL: func.func @dispatchFn2
func.func @dispatchFn2(%arg0 : tensor<8x4xf32>) -> tensor<4x8xf32> {
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  // CHECK: flow.dispatch @dispatchFn2_dispatch_0::@dispatchFn2_dispatch_0
  // CHECK-SAME: stream.affinity = #hal.affinity.queue<[1]>
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<8x4xf32>) -> (tensor<4x8xf32>) attributes {
    stream.affinity = #hal.affinity.queue<[1]>
  } = (
    %arg: !flow.dispatch.tensor<readonly:tensor<8x4xf32>>, %ret: !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
  ) {
    flow.return
  }
  return %0 : tensor<4x8xf32>
}

// -----

//      CHECK: flow.executable private @dynamicShapeDispatch_dispatch_0
// CHECK-NEXT:   flow.executable.export public @dynamicShapeDispatch_dispatch_0
//      CHECK: func.func @dynamicShapeDispatch_dispatch_0(
// CHECK-SAME:     %[[ARG_TENSOR:.+]]: !flow.dispatch.tensor<readonly:tensor<7x?x24x?xf32>>,
// CHECK-SAME:     %[[DIM1_CAPTURE:.+]]: index, %[[DIM3_CAPTURE:.+]]: index,
// CHECK-SAME:     %[[RET_TENSOR:.+]]: !flow.dispatch.tensor<writeonly:tensor<?x?x1024xf32>>) {

//      CHECK: %[[ARG_TILE:.+]] = flow.dispatch.tensor.load %[[ARG_TENSOR]], {{.+}} : !flow.dispatch.tensor<readonly:tensor<7x?x24x?xf32>>{%[[DIM1_CAPTURE]], %[[DIM3_CAPTURE]]}
// CHECK-NEXT: %[[RET_TILE:.+]] = "test.tile_math"(%[[ARG_TILE]])
// CHECK-NEXT: flow.dispatch.tensor.store %[[RET_TILE]], %[[RET_TENSOR]], {{.+}} -> !flow.dispatch.tensor<writeonly:tensor<?x?x1024xf32>>{%[[DIM3_CAPTURE]], %[[DIM1_CAPTURE]]}

// CHECK:   return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @dynamicShapeDispatch(
// CHECK-SAME: %[[ARG0:.+]]: tensor<7x?x24x?xf32>
func.func @dynamicShapeDispatch(%arg0 : tensor<7x?x24x?xf32>) -> tensor<?x?x1024xf32> {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  // CHECK-DAG: %[[DIM1:.+]] = tensor.dim %[[ARG0]], %c1
  %dim1 = tensor.dim %arg0, %c1 : tensor<7x?x24x?xf32>
  // CHECK-DAG: %[[DIM3:.+]] = tensor.dim %[[ARG0]], %c3
  %dim3 = tensor.dim %arg0, %c3 : tensor<7x?x24x?xf32>
  // CHECK-DAG: %[[X:.+]] = arith.constant 1024
  %x = arith.constant 1024 : index
  // CHECK-DAG: %[[Y:.+]] = arith.constant 512
  %y = arith.constant 512 : index
  // CHECK-NEXT: %[[RET0:.+]] = flow.dispatch @dynamicShapeDispatch_dispatch_0::@dynamicShapeDispatch_dispatch_0[
  // CHECK-SAME:   %[[X]], %[[Y]]
  // CHECK-SAME: ](%arg0, %[[DIM1]], %[[DIM3]])
  // CHECK-SAME: : (tensor<7x?x24x?xf32>{%[[DIM1]], %[[DIM3]]}, index, index) -> tensor<?x?x1024xf32>{%[[DIM3]], %[[DIM1]]}
  %ret0 = flow.dispatch.workgroups[%x, %y](%arg0, %dim1, %dim3) : (tensor<7x?x24x?xf32>{%dim1, %dim3}, index, index) -> tensor<?x?x1024xf32>{%dim3, %dim1} = (
    %arg: !flow.dispatch.tensor<readonly:tensor<7x?x24x?xf32>>,
    %dim1_capture: index, %dim3_capture: index,
    %ret: !flow.dispatch.tensor<writeonly:tensor<?x?x1024xf32>>
  ) {
    %arg_tile = flow.dispatch.tensor.load %arg, offsets=[0, 0, 0, 0], sizes=[7, %dim1_capture, 24, %dim3_capture], strides=[1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<7x?x24x?xf32>>{%dim1_capture, %dim3_capture} -> tensor<7x?x24x?xf32>
    %ret_tile = "test.tile_math"(%arg_tile) : (tensor<7x?x24x?xf32>) -> (tensor<?x?x1024xf32>)
    flow.dispatch.tensor.store %ret_tile, %ret, offsets=[0, 0, 0], sizes=[%dim3_capture, %dim1_capture, 1024], strides=[1, 1, 1] : tensor<?x?x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x1024xf32>>{%dim3_capture, %dim1_capture}
    flow.return
  }
  // CHECK-NEXT: return %[[RET0]]
  return %ret0 : tensor<?x?x1024xf32>
}

// -----

// CHECK-LABEL: func.func @dispatchWithCountRegion
func.func @dispatchWithCountRegion(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<4xi32>) -> %arg0 =
      (%arg0_capture: !flow.dispatch.tensor<readwrite:tensor<4xi32>>) {
    flow.return
  } count(%x_capture: index, %y_capture: index) -> (index, index, index) {
    %z = arith.constant 1 : index
    flow.return %x_capture, %y_capture, %z : index, index, index
  }
  return %0 : tensor<4xi32>
}

// -----

//      CHECK: flow.executable private @dispatchExtern_dispatch_0
// CHECK-NEXT:   flow.executable.export public @main
// CHECK-SAME:     workgroups(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
// CHECK-NEXT:       %ok, %value = hal.device.query<%arg0 : !hal.device> key("some" :: "value") : i1, i32
// CHECK-NEXT:       %0 = arith.index_cast %value : i32 to index
// CHECK-NEXT:       hal.return %arg1, %arg2, %0 : index, index, index
// CHECK-NEXT:     } attributes {
// CHECK-SAME:       hal.interface.layout = #hal.pipeline.layout<push_constants = 1, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>
// CHECK-SAME:     }

// Demonstrates the full functionality of an extern dispatch op.
// Note that some fields are optional.

// CHECK-LABEL: func.func @dispatchExtern
func.func @dispatchExtern(%arg0: tensor<4xi32>, %arg1: tensor<8xi32>, %arg2: i32) -> tensor<8xi32> {
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  // Dispatch workgroups to the externally defined function "main" in the
  // referenced object files.
  // CHECK: %[[RESULT:.+]] = flow.dispatch @dispatchExtern_dispatch_0::@main[%c100, %c50](%arg0, %arg1, %arg2) {
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
    objects(#hal.executable.objects<{
      #hal.executable.target<"llvm-cpu", "a"> = [#hal.executable.object<{path = "a.o"}>],
      #hal.executable.target<"llvm-cpu", "b"> = [#hal.executable.object<{path = "b.o"}>]
    }>)
  // CHECK: return %[[RESULT]]
  return %result : tensor<8xi32>
}
