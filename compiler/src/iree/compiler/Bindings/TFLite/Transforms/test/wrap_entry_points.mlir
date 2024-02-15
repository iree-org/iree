// RUN: iree-opt --iree-tflite-wrap-entry-points --canonicalize -cse --split-input-file %s | FileCheck %s

// NOTE: CSE is run because otherwise there's just way too much IR and we don't
//       care about 100 random 0-N constants.
// NOTE: we do a lot of CHECK-NEXTing here because we want to ensure we are
//       emitting things in the same order as they are in the function
//       signatures to make the IR easier to read.

// CHECK-DAG: util.global private mutable @_tflite_dynamicEntry_input0_shape_dim0 : index
// CHECK-DAG: util.global private mutable @_tflite_dynamicEntry_input1_shape_dim0 : index
// CHECK-DAG: util.global private mutable @_tflite_dynamicEntry_output0_shape_dim0 : index
// CHECK-DAG: util.global private mutable @_tflite_dynamicEntry_output1_shape_dim0 : index
// CHECK-DAG: util.global private mutable @_tflite_dynamicEntry_shapes_dirty = true



// CHECK-LABEL: util.func private @_tflite_dynamicEntry_calculate_shapes() {

// Only recalculate shapes if the shapes are dirty.
//       CHECK:   %[[IS_DIRTY:.+]] = util.global.load @_tflite_dynamicEntry_shapes_dirty : i1
//  CHECK-NEXT:   cf.cond_br %[[IS_DIRTY]], ^bb1, ^bb2

//       CHECK: ^bb1:
//  CHECK-NEXT:   %[[NULL:.+]] = util.null : !hal.buffer

// Tie input0 shapes.
//  CHECK-NEXT:   %[[IN0_DIM0:.+]] = util.global.load @_tflite_dynamicEntry_input0_shape_dim0 : index
//  CHECK-NEXT:   %[[IN0:.+]] = hal.tensor.import %[[NULL]] : !hal.buffer -> tensor<?x8x8x3xf32>{%[[IN0_DIM0]]}

// Tie input1 shapes.
//  CHECK-NEXT:   %[[IN1_DIM0:.+]] = util.global.load @_tflite_dynamicEntry_input1_shape_dim0 : index
//  CHECK-NEXT:   %[[IN1:.+]] = hal.tensor.import %[[NULL]] : !hal.buffer -> tensor<?x8x8x3xf32>{%[[IN1_DIM0]]}

// The actual model code used to (eventually) compute shapes.
//  CHECK-NEXT:   %[[OUT0:.+]] = arith.addf %[[IN0]], %[[IN1]]
//  CHECK-NEXT:   %[[OUT1:.+]] = arith.addf %[[OUT0]], %[[IN0]]

// Store back the new dynamic dimensions of out0/out1.
//       CHECK:   %[[OUT0_DIM0:.+]] = tensor.dim %[[OUT0]], %c0
//  CHECK-NEXT:   util.global.store %[[OUT0_DIM0]], @_tflite_dynamicEntry_output0_shape_dim0 : index
//       CHECK:   %[[OUT1_DIM0:.+]] = tensor.dim %[[OUT1]], %c0
//  CHECK-NEXT:   util.global.store %[[OUT1_DIM0]], @_tflite_dynamicEntry_output1_shape_dim0 : index

// Clear dirty bit now that the shapes have been recalculated.
//       CHECK:   util.global.store %false, @_tflite_dynamicEntry_shapes_dirty : i1
//  CHECK-NEXT:   util.return

// Exit for when the shapes are not dirty and no work is needed.
//  CHECK-NEXT: ^bb2:
//  CHECK-NEXT:   util.return
//  CHECK-NEXT: }



// CHECK-LABEL: util.func public @_tflite_dynamicEntry_query_input_shape
//  CHECK-SAME:   (%[[INDEX:.+]]: index, %[[LIST:.+]]: !util.list<index>)

// Query input0 shape:
//       CHECK:   %[[IS_0:.+]] = arith.cmpi eq, %[[INDEX]], %c0 : index
//  CHECK-NEXT:   cf.cond_br %[[IS_0]], ^bb1, ^bb2
//  CHECK-NEXT: ^bb1:
//  CHECK-NEXT:   util.list.resize %[[LIST]], %c4 : !util.list<index>
//  CHECK-NEXT:   %[[IN0_DIM0:.+]] = util.global.load @_tflite_dynamicEntry_input0_shape_dim0 : index
//  CHECK-NEXT:   util.list.set %[[LIST]][%c0], %[[IN0_DIM0]] : !util.list<index>
//  CHECK-NEXT:   util.list.set %[[LIST]][%c1], %c8 : !util.list<index>
//  CHECK-NEXT:   util.list.set %[[LIST]][%c2], %c8 : !util.list<index>
//  CHECK-NEXT:   util.list.set %[[LIST]][%c3], %c3 : !util.list<index>
//  CHECK-NEXT:   cf.br ^bb4

// Query input1 shape:
//       CHECK: ^bb2:
//  CHECK-NEXT:   %[[IS_1:.+]] = arith.cmpi eq, %[[INDEX]], %c1 : index
//  CHECK-NEXT:   cf.cond_br %[[IS_1]], ^bb3, ^bb4
//  CHECK-NEXT: ^bb3:
//  CHECK-NEXT:   util.list.resize %[[LIST]], %c4 : !util.list<index>
//  CHECK-NEXT:   %[[IN1_DIM0:.+]] = util.global.load @_tflite_dynamicEntry_input1_shape_dim0 : index
//  CHECK-NEXT:   util.list.set %[[LIST]][%c0], %[[IN1_DIM0]] : !util.list<index>
//  CHECK-NEXT:   util.list.set %[[LIST]][%c1], %c8 : !util.list<index>
//  CHECK-NEXT:   util.list.set %[[LIST]][%c2], %c8 : !util.list<index>
//  CHECK-NEXT:   util.list.set %[[LIST]][%c3], %c3 : !util.list<index>
//  CHECK-NEXT:   cf.br ^bb4

// Invalid input index:
//       CHECK: ^bb4:
//  CHECK-NEXT:   util.return
//  CHECK-NEXT: }



// CHECK-LABEL: util.func public @_tflite_dynamicEntry_resize_input_shape
//  CHECK-SAME:   (%[[INDEX:.+]]: index, %[[LIST:.+]]: !util.list<index>)

//       CHECK:   %[[IS_0:.+]] = arith.cmpi eq, %[[INDEX]], %c0 : index
//  CHECK-NEXT:   cf.cond_br %[[IS_0]], ^bb1, ^bb2
//  CHECK-NEXT: ^bb1:
//  CHECK-NEXT:   %[[IN0_DIM0:.+]] = util.list.get %[[LIST]][%c0] : !util.list<index>
//  CHECK-NEXT:   util.global.store %[[IN0_DIM0]], @_tflite_dynamicEntry_input0_shape_dim0 : index
//  CHECK-NEXT:   cf.br ^bb4

//       CHECK: ^bb2:
//  CHECK-NEXT:   %[[IS_1:.+]] = arith.cmpi eq, %[[INDEX]], %c1 : index
//  CHECK-NEXT:   cf.cond_br %[[IS_1]], ^bb3, ^bb4
//  CHECK-NEXT: ^bb3:
//  CHECK-NEXT:   %[[IN1_DIM0:.+]] = util.list.get %[[LIST]][%c0] : !util.list<index>
//  CHECK-NEXT:   util.global.store %[[IN1_DIM0]], @_tflite_dynamicEntry_input1_shape_dim0 : index
//  CHECK-NEXT:   cf.br ^bb4

// Set the dirty flag so that shape calculation must run again.
//  CHECK-NEXT: ^bb4:
//  CHECK-NEXT:   util.global.store %true, @_tflite_dynamicEntry_shapes_dirty : i1
//  CHECK-NEXT:   util.return
//  CHECK-NEXT: }



// CHECK-LABEL: util.func public @_tflite_dynamicEntry_query_output_shape
//  CHECK-SAME:   (%[[INDEX:.+]]: index, %[[LIST:.+]]: !util.list<index>)

// Recalculate shapes, if needed.
//       CHECK:   call @_tflite_dynamicEntry_calculate_shapes() : () -> ()

// Query output0:
//       CHECK:   %[[IS_0:.+]] = arith.cmpi eq, %[[INDEX]], %c0 : index
//  CHECK-NEXT:   cf.cond_br %[[IS_0]], ^bb1, ^bb2
//  CHECK-NEXT: ^bb1:
//  CHECK-NEXT:   util.list.resize %[[LIST]], %c4 : !util.list<index>
//  CHECK-NEXT:   %[[OUT0_DIM0:.+]] = util.global.load @_tflite_dynamicEntry_output0_shape_dim0 : index
//  CHECK-NEXT:   util.list.set %[[LIST]][%c0], %[[OUT0_DIM0]] : !util.list<index>
//  CHECK-NEXT:   util.list.set %[[LIST]][%c1], %c8 : !util.list<index>
//  CHECK-NEXT:   util.list.set %[[LIST]][%c2], %c8 : !util.list<index>
//  CHECK-NEXT:   util.list.set %[[LIST]][%c3], %c3 : !util.list<index>
//  CHECK-NEXT:   cf.br ^bb4

// Query output1:
//       CHECK: ^bb2:
//  CHECK-NEXT:   %[[IS_1:.+]] = arith.cmpi eq, %[[INDEX]], %c1 : index
//  CHECK-NEXT:   cf.cond_br %[[IS_1]], ^bb3, ^bb4
//  CHECK-NEXT: ^bb3:
//  CHECK-NEXT:   util.list.resize %[[LIST]], %c4 : !util.list<index>
//  CHECK-NEXT:   %[[OUT1_DIM0:.+]] = util.global.load @_tflite_dynamicEntry_output1_shape_dim0 : index
//  CHECK-NEXT:   util.list.set %[[LIST]][%c0], %[[OUT1_DIM0]] : !util.list<index>
//  CHECK-NEXT:   util.list.set %[[LIST]][%c1], %c8 : !util.list<index>
//  CHECK-NEXT:   util.list.set %[[LIST]][%c2], %c8 : !util.list<index>
//  CHECK-NEXT:   util.list.set %[[LIST]][%c3], %c3 : !util.list<index>
//  CHECK-NEXT:   cf.br ^bb4

//  CHECK-NEXT: ^bb4:
//  CHECK-NEXT:   util.return
//  CHECK-NEXT: }



// CHECK-LABEL: util.func public @_tflite_main(
//  CHECK-SAME:   %[[IN0_BUFFER:.+]]: !hal.buffer {iree.identifier = "input0"},
//  CHECK-SAME:   %[[IN1_BUFFER:.+]]: !hal.buffer {iree.identifier = "input1"})
//  CHECK-SAME: -> (
//  CHECK-SAME:   !hal.buffer {iree.identifier = "output0"},
//  CHECK-SAME:   !hal.buffer {iree.identifier = "output1"}
//  CHECK-SAME: ) attributes {
//  CHECK-SAME:   iree.abi.stub,
//  CHECK-SAME:   iree.reflection = {
//  CHECK-SAME:     tfl.io.names = "input0;input1;output0;output1"
//  CHECK-SAME:   }
//  CHECK-SAME: } {

// Cast input0 buffer to a shaped tensor.
//      CHECK:   %[[IN0_DIM0:.+]] = util.global.load @_tflite_dynamicEntry_input0_shape_dim0 : index
// CHECK-NEXT:   %[[IN0:.+]] = hal.tensor.import %[[IN0_BUFFER]] : !hal.buffer -> tensor<?x8x8x3xf32>{%[[IN0_DIM0]]}

// Cast input1 buffer to a shaped tensor.
//      CHECK:   %[[IN1_DIM0:.+]] = util.global.load @_tflite_dynamicEntry_input1_shape_dim0 : index
// CHECK-NEXT:   %[[IN1:.+]] = hal.tensor.import %[[IN1_BUFFER]] : !hal.buffer -> tensor<?x8x8x3xf32>{%[[IN1_DIM0]]}

// Call the original function with tensor arguments.
//      CHECK:   %[[OUT:.+]]:2 = util.call @dynamicEntry(%[[IN0]], %[[IN1]]) : (tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>) -> (tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>)

// Query output0 shape and get the HAL buffer to return.
//      CHECK:   %[[OUT0_DIM0:.+]] = tensor.dim %[[OUT]]#0, %c0 : tensor<?x8x8x3xf32>
// CHECK-NEXT:   %[[OUT0_BUFFER:.+]] = hal.tensor.export %[[OUT]]#0 : tensor<?x8x8x3xf32>{%[[OUT0_DIM0]]} -> !hal.buffer
// CHECK-NEXT:   util.global.store %[[OUT0_DIM0]], @_tflite_dynamicEntry_output0_shape_dim0 : index

// Query output1 shape and get the HAL buffer to return.
//      CHECK:   %[[OUT1_DIM0:.+]] = tensor.dim %[[OUT]]#1, %c0 : tensor<?x8x8x3xf32>
// CHECK-NEXT:   %[[OUT1_BUFFER:.+]] = hal.tensor.export %[[OUT]]#1 : tensor<?x8x8x3xf32>{%[[OUT1_DIM0]]} -> !hal.buffer
// CHECK-NEXT:   util.global.store %[[OUT1_DIM0]], @_tflite_dynamicEntry_output1_shape_dim0 : index

// Clear shape dirty bit as we've updated the shapes unconditionally.
// CHECK-NEXT:   util.global.store %false, @_tflite_dynamicEntry_shapes_dirty : i1

// CHECK-NEXT:   util.return %[[OUT0_BUFFER]], %[[OUT1_BUFFER]]
// CHECK-NEXT: }



// CHECK-LABEL: util.func private @dynamicEntry(
util.func public @dynamicEntry(
  %arg0: tensor<?x8x8x3xf32> {iree.identifier = "input0"},
  %arg1: tensor<?x8x8x3xf32> {iree.identifier = "input1"}
) -> (
  tensor<?x8x8x3xf32> {iree.identifier = "output0"},
  tensor<?x8x8x3xf32> {iree.identifier = "output1"}
) {
  // CHECK: = arith.addf
  %0 = arith.addf %arg0, %arg1 : tensor<?x8x8x3xf32>
  // CHECK: = arith.addf
  %1 = arith.addf %0, %arg0 : tensor<?x8x8x3xf32>
  // CHECK: util.return
  util.return %0, %1 : tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>
}

// -----

// CHECK-LABEL: util.func public @_tflite_main(
//  CHECK-SAME:   %[[IN0_BUFFER:.+]]: !hal.buffer,
//  CHECK-SAME:   %[[IN1_BUFFER:.+]]: !hal.buffer)
//  CHECK-SAME: -> (
//  CHECK-SAME:   !hal.buffer,
//  CHECK-SAME:   !hal.buffer
//  CHECK-SAME: ) attributes {
//  CHECK-SAME:   iree.abi.stub,
//  CHECK-SAME:   iree.reflection = {
//  CHECK-SAME:     tfl.io.names = "arg0;arg1;ret0;ret1"
//  CHECK-SAME:   }
//  CHECK-SAME: } {

util.func public @dynamicEntryWithoutIdentifiers(
  %arg0: tensor<?x8x8x3xf32>,
  %arg1: tensor<?x8x8x3xf32>
) -> (
  tensor<?x8x8x3xf32>,
  tensor<?x8x8x3xf32>
) {
  // CHECK: = arith.addf
  %0 = arith.addf %arg0, %arg1 : tensor<?x8x8x3xf32>
  // CHECK: = arith.addf
  %1 = arith.addf %0, %arg0 : tensor<?x8x8x3xf32>
  // CHECK: util.return
  util.return %0, %1 : tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>
}
