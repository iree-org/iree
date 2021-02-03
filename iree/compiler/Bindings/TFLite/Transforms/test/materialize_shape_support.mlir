// RUN: iree-opt -iree-tflite-materialize-shape-support -canonicalize -split-input-file %s | IreeFileCheck %s

// NOTE: canonicalization is run because otherwise there's just way too much IR.

// CHECK-DAG: flow.variable @_tflite_dynamicEntry_input0_shape mutable : !shapex.ranked_shape<[?,8,8,3]>
// CHECK-DAG: flow.variable @_tflite_dynamicEntry_input1_shape mutable : !shapex.ranked_shape<[?,8,8,3]>
// CHECK-DAG: flow.variable @_tflite_dynamicEntry_output0_shape mutable : !shapex.ranked_shape<[?,8,8,3]>
// CHECK-DAG: flow.variable @_tflite_dynamicEntry_output1_shape mutable : !shapex.ranked_shape<[?,8,8,3]>
// CHECK-DAG: flow.variable @_tflite_dynamicEntry_shapes_dirty mutable true

// CHECK-LABEL: func private @_tflite_dynamicEntry_calculate_shapes() {
//  CHECK-NEXT:   %false = constant false
//  CHECK-NEXT:   %[[IS_DIRTY:.+]] = flow.variable.load @_tflite_dynamicEntry_shapes_dirty : i1
//  CHECK-NEXT:   cond_br %[[IS_DIRTY]], ^bb1, ^bb2
//  CHECK-NEXT: ^bb1:
//  CHECK-NEXT:   %[[IN0_NULL:.+]] = iree.null : tensor<?x8x8x3xf32>
//  CHECK-NEXT:   %[[IN0_SHAPE:.+]] = flow.variable.load @_tflite_dynamicEntry_input0_shape : !shapex.ranked_shape<[?,8,8,3]>
//  CHECK-NEXT:   %[[IN0:.+]] = shapex.tie_shape %[[IN0_NULL]], %[[IN0_SHAPE]] : tensor<?x8x8x3xf32>, !shapex.ranked_shape<[?,8,8,3]>
//  CHECK-NEXT:   %[[IN1_NULL:.+]] = iree.null : tensor<?x8x8x3xf32>
//  CHECK-NEXT:   %[[IN1_SHAPE:.+]] = flow.variable.load @_tflite_dynamicEntry_input1_shape : !shapex.ranked_shape<[?,8,8,3]>
//  CHECK-NEXT:   %[[IN1:.+]] = shapex.tie_shape %[[IN1_NULL]], %[[IN1_SHAPE]] : tensor<?x8x8x3xf32>, !shapex.ranked_shape<[?,8,8,3]>
//  CHECK-NEXT:   %[[TMP:.+]]:2 = call @dynamicEntry(%[[IN0]], %[[IN1]])
//  CHECK-NEXT:   %[[OUT0_SHAPE:.+]] = shapex.get_ranked_shape %[[TMP]]#0 : tensor<?x8x8x3xf32> -> !shapex.ranked_shape<[?,8,8,3]>
//  CHECK-NEXT:   flow.variable.store %[[OUT0_SHAPE]], @_tflite_dynamicEntry_output0_shape : !shapex.ranked_shape<[?,8,8,3]>
//  CHECK-NEXT:   %[[OUT1_SHAPE:.+]] = shapex.get_ranked_shape %[[TMP]]#1 : tensor<?x8x8x3xf32> -> !shapex.ranked_shape<[?,8,8,3]>
//  CHECK-NEXT:   flow.variable.store %[[OUT1_SHAPE]], @_tflite_dynamicEntry_output1_shape : !shapex.ranked_shape<[?,8,8,3]>
//  CHECK-NEXT:   flow.variable.store %false, @_tflite_dynamicEntry_shapes_dirty : i1
//  CHECK-NEXT:   return
//  CHECK-NEXT: ^bb2:
//  CHECK-NEXT:   return
//  CHECK-NEXT: }

// CHECK-LABEL: func @_tflite_dynamicEntry_query_input_shape
//  CHECK-SAME:   (%[[INDEX:.+]]: index, %[[LIST:.+]]: !iree.list<index>)
//       CHECK:   %[[IS_0:.+]] = cmpi eq, %[[INDEX]], %c0 : index
//  CHECK-NEXT:   cond_br %[[IS_0]], ^bb1, ^bb2
//  CHECK-NEXT: ^bb1:
//  CHECK-NEXT:   %[[IN0_SHAPE:.+]] = flow.variable.load @_tflite_dynamicEntry_input0_shape : !shapex.ranked_shape<[?,8,8,3]>
//  CHECK-NEXT:   iree.list.resize %[[LIST]], %c4 : !iree.list<index>
//  CHECK-NEXT:   %[[IN0_D0:.+]] = shapex.ranked_dim %[[IN0_SHAPE]][0] : !shapex.ranked_shape<[?,8,8,3]> -> index
//  CHECK-NEXT:   iree.list.set %[[LIST]], %c0, %[[IN0_D0]] : !iree.list<index>
//  CHECK-NEXT:   iree.list.set %[[LIST]], %c1, %c8 : !iree.list<index>
//  CHECK-NEXT:   iree.list.set %[[LIST]], %c2, %c8 : !iree.list<index>
//  CHECK-NEXT:   iree.list.set %[[LIST]], %c3, %c3 : !iree.list<index>
//  CHECK-NEXT:   br ^bb4
//  CHECK-NEXT: ^bb2:
//  CHECK-NEXT:   %[[IS_1:.+]] = cmpi eq, %[[INDEX]], %c1 : index
//  CHECK-NEXT:   cond_br %[[IS_1]], ^bb3, ^bb4
//  CHECK-NEXT: ^bb3:
//  CHECK-NEXT:   %[[IN1_SHAPE:.+]] = flow.variable.load @_tflite_dynamicEntry_input1_shape : !shapex.ranked_shape<[?,8,8,3]>
//  CHECK-NEXT:   iree.list.resize %[[LIST]], %c4 : !iree.list<index>
//  CHECK-NEXT:   %[[IN1_D0:.+]] = shapex.ranked_dim %[[IN1_SHAPE]][0] : !shapex.ranked_shape<[?,8,8,3]> -> index
//  CHECK-NEXT:   iree.list.set %[[LIST]], %c0, %[[IN1_D0]] : !iree.list<index>
//  CHECK-NEXT:   iree.list.set %[[LIST]], %c1, %c8 : !iree.list<index>
//  CHECK-NEXT:   iree.list.set %[[LIST]], %c2, %c8 : !iree.list<index>
//  CHECK-NEXT:   iree.list.set %[[LIST]], %c3, %c3 : !iree.list<index>
//  CHECK-NEXT:   br ^bb4
//  CHECK-NEXT: ^bb4:
//  CHECK-NEXT:   return
//  CHECK-NEXT: }

// CHECK-LABEL: func @_tflite_dynamicEntry_resize_input_shape
//  CHECK-SAME:   (%[[INDEX:.+]]: index, %[[LIST:.+]]: !iree.list<index>)
//       CHECK:   %[[IS_0:.+]] = cmpi eq, %[[INDEX]], %c0 : index
//  CHECK-NEXT:   cond_br %[[IS_0]], ^bb1, ^bb2
//  CHECK-NEXT: ^bb1:
//  CHECK-NEXT:   %[[IN0_D0:.+]] = iree.list.get %[[LIST]], %c0 : !iree.list<index>
//  CHECK-NEXT:   %[[IN0_SHAPE:.+]] = shapex.make_ranked_shape %[[IN0_D0]] : (index) -> !shapex.ranked_shape<[?,8,8,3]>
//  CHECK-NEXT:   flow.variable.store %[[IN0_SHAPE]], @_tflite_dynamicEntry_input0_shape : !shapex.ranked_shape<[?,8,8,3]>
//  CHECK-NEXT:   br ^bb4
//  CHECK-NEXT: ^bb2:
//  CHECK-NEXT:   %[[IS_1:.+]] = cmpi eq, %[[INDEX]], %c1 : index
//  CHECK-NEXT:   cond_br %[[IS_1]], ^bb3, ^bb4
//  CHECK-NEXT: ^bb3:
//  CHECK-NEXT:   %[[IN1_D0:.+]] = iree.list.get %[[LIST]], %c0 : !iree.list<index>
//  CHECK-NEXT:   %[[IN1_SHAPE:.+]] = shapex.make_ranked_shape %[[IN1_D0]] : (index) -> !shapex.ranked_shape<[?,8,8,3]>
//  CHECK-NEXT:   flow.variable.store %[[IN1_SHAPE]], @_tflite_dynamicEntry_input1_shape : !shapex.ranked_shape<[?,8,8,3]>
//  CHECK-NEXT:   br ^bb4
//  CHECK-NEXT: ^bb4:
//  CHECK-NEXT:   flow.variable.store %true, @_tflite_dynamicEntry_shapes_dirty : i1
//  CHECK-NEXT:   return
//  CHECK-NEXT: }

// CHECK-LABEL: func @_tflite_dynamicEntry_query_output_shape
//  CHECK-SAME:   (%[[INDEX:.+]]: index, %[[LIST:.+]]: !iree.list<index>)
//       CHECK:   call @_tflite_dynamicEntry_calculate_shapes() : () -> ()
//  CHECK-NEXT:   %[[IS_0:.+]] = cmpi eq, %[[INDEX]], %c0 : index
//  CHECK-NEXT:   cond_br %[[IS_0]], ^bb1, ^bb2
//  CHECK-NEXT: ^bb1:
//  CHECK-NEXT:   %[[OUT0_SHAPE:.+]] = flow.variable.load @_tflite_dynamicEntry_output0_shape : !shapex.ranked_shape<[?,8,8,3]>
//  CHECK-NEXT:   iree.list.resize %[[LIST]], %c4 : !iree.list<index>
//  CHECK-NEXT:   %[[OUT0_D0:.+]] = shapex.ranked_dim %[[OUT0_SHAPE]][0] : !shapex.ranked_shape<[?,8,8,3]> -> index
//  CHECK-NEXT:   iree.list.set %[[LIST]], %c0, %[[OUT0_D0]] : !iree.list<index>
//  CHECK-NEXT:   iree.list.set %[[LIST]], %c1, %c8 : !iree.list<index>
//  CHECK-NEXT:   iree.list.set %[[LIST]], %c2, %c8 : !iree.list<index>
//  CHECK-NEXT:   iree.list.set %[[LIST]], %c3, %c3 : !iree.list<index>
//  CHECK-NEXT:   br ^bb4
//  CHECK-NEXT: ^bb2:
//  CHECK-NEXT:   %[[IS_1:.+]] = cmpi eq, %[[INDEX]], %c1 : index
//  CHECK-NEXT:   cond_br %[[IS_1]], ^bb3, ^bb4
//  CHECK-NEXT: ^bb3:
//  CHECK-NEXT:   %[[OUT1_SHAPE:.+]] = flow.variable.load @_tflite_dynamicEntry_output1_shape : !shapex.ranked_shape<[?,8,8,3]>
//  CHECK-NEXT:   iree.list.resize %[[LIST]], %c4 : !iree.list<index>
//  CHECK-NEXT:   %[[OUT1_D0:.+]] = shapex.ranked_dim %[[OUT1_SHAPE]][0] : !shapex.ranked_shape<[?,8,8,3]> -> index
//  CHECK-NEXT:   iree.list.set %[[LIST]], %c0, %[[OUT1_D0]] : !iree.list<index>
//  CHECK-NEXT:   iree.list.set %[[LIST]], %c1, %c8 : !iree.list<index>
//  CHECK-NEXT:   iree.list.set %[[LIST]], %c2, %c8 : !iree.list<index>
//  CHECK-NEXT:   iree.list.set %[[LIST]], %c3, %c3 : !iree.list<index>
//  CHECK-NEXT:   br ^bb4
//  CHECK-NEXT: ^bb4:
//  CHECK-NEXT:   return
//  CHECK-NEXT: }

// CHECK-LABEL: func @_tflite_dynamicEntry(
func @_tflite_dynamicEntry(%arg0: tensor<?x8x8x3xf32> {iree.identifier = "input0"}, %arg1: tensor<?x8x8x3xf32> {iree.identifier = "input1"}) -> (tensor<?x8x8x3xf32> {iree.identifier = "output0"}, tensor<?x8x8x3xf32> {iree.identifier = "output1"}) attributes {
  iree.abi.stub,
  iree.module.export,
  iree.reflection = {
    tfl.io.names = "input0;input1;output0;output1"
  }
} {
  %0:2 = call @dynamicEntry(%arg0, %arg1) : (tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>) -> (tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>)
  return %0#0, %0#1 : tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>
}

// CHECK-LABEL: func private @dynamicEntry(
func private @dynamicEntry(%arg0: tensor<?x8x8x3xf32> {iree.identifier = "input0"}, %arg1: tensor<?x8x8x3xf32> {iree.identifier = "input1"}) -> (tensor<?x8x8x3xf32> {iree.identifier = "output0"}, tensor<?x8x8x3xf32> {iree.identifier = "output1"}) {
  %0 = mhlo.add %arg0, %arg1 : tensor<?x8x8x3xf32>
  %1 = mhlo.add %0, %arg0 : tensor<?x8x8x3xf32>
  return %0, %1 : tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>
}
