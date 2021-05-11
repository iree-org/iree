// RUN: iree-opt -allow-unregistered-dialect -split-input-file -canonicalize -cse %s | iree-opt -allow-unregistered-dialect -split-input-file | IreeFileCheck %s

// CHECK-LABEL: func @expand_buffer_view_subview
func @expand_buffer_view_subview(
  // CHECK-SAME: %[[VIEW:.+]]: !hal.buffer_view,
  %view : !hal.buffer_view,
  // CHECK-SAME: %[[INDEX0:.+]]: index, %[[INDEX1:.+]]: index, %[[LENGTH0:.+]]: index, %[[LENGTH1:.+]]: index
  %index0 : index, %index1 : index, %length0 : index, %length1 : index
) -> !hal.buffer_view {
  //      CHECK: = hal.buffer_view.dim %[[VIEW]], 1 : index
  //      CHECK: %[[ELEMENT_TYPE:.+]] = hal.buffer_view.element_type %[[VIEW]] : i32
  // << A BUNCH OF MATH >>
  //      CHECK: %[[BUFFER:.+]] = hal.buffer_view.buffer %[[VIEW]] : !hal.buffer
  // CHECK-NEXT: %[[SUBSPAN:.+]] = hal.buffer.subspan<%[[BUFFER]] : !hal.buffer>[%{{.+}}, %{{.+}}] : !hal.buffer
  //      CHECK: %[[SUBVIEW:.+]] = hal.buffer_view.create
  // CHECK-SAME:     %[[SUBSPAN]],
  // CHECK-SAME:     element_type = %[[ELEMENT_TYPE]],
  // CHECK-SAME:     shape = [%[[LENGTH0]], %[[LENGTH1]]] : !hal.buffer -> !hal.buffer_view
  %subview = hal.buffer_view.subview %view,
                                     indices = [%index0, %index1],
                                     lengths = [%length0, %length1] : !hal.buffer_view
  // CHECK-NEXT: return %[[SUBVIEW]]
  return %subview : !hal.buffer_view
}

// -----

// CHECK-LABEL: func @skip_buffer_view_buffer
// CHECK-SAME: %[[BUFFER:.+]]: !hal.buffer
func @skip_buffer_view_buffer(%buffer : !hal.buffer) -> !hal.buffer {
  %c10 = constant 10 : index
  %c11 = constant 11 : index
  %c32 = constant 32 : i32
  %view = hal.buffer_view.create %buffer, element_type = %c32, shape = [%c10, %c11] : !hal.buffer -> !hal.buffer_view
  %view_buffer = hal.buffer_view.buffer %view : !hal.buffer
  // CHECK: return %[[BUFFER]]
  return %view_buffer : !hal.buffer
}

// -----

// CHECK-LABEL: func @buffer_view_compute_offset
// CHECK-SAME: %[[VIEW:.+]]: !hal.buffer_view
func @buffer_view_compute_offset(%arg0 : !hal.buffer_view) -> index {
  // CHECK: %[[INDICES:.+]]:2 = "test_hal.indices"() : () -> (index, index)
  %0:2 = "test_hal.indices"() : () -> (index, index)
  // CHECK: %[[D0:.+]] = hal.buffer_view.dim %[[VIEW]], 1 : index
  // CHECK: %[[TYPE:.+]] = hal.buffer_view.element_type %[[VIEW]] : i32
  // CHECK: %[[T0:.+]] = muli %[[INDICES]]#0, %[[D0]] : index
  // CHECK: %[[T1:.+]] = addi %[[T0]], %[[INDICES]]#1 : index
  // CHECK: %[[T2:.+]] = index_cast %[[TYPE]] : i32 to index
  // CHECK: %[[T3:.+]] = and %[[T2]], %c255 : index
  // CHECK: %[[T4:.+]] = addi %[[T3]], %c7 : index
  // CHECK: %[[T6:.+]] = divi_unsigned %[[T4]], %c8 : index
  // CHECK: %[[T7:.+]] = muli %[[T1]], %[[T6]] : index
  %off = hal.buffer_view.compute_offset %arg0, indices = [%0#0, %0#1] : index
  // CHECK: return %[[T7]]
  return %off : index
}

// -----

// CHECK-LABEL: func @buffer_view_compute_range
// CHECK-SAME: %[[VIEW:.+]]: !hal.buffer_view
func @buffer_view_compute_range(%arg0 : !hal.buffer_view) -> (index, index) {
  %0:2 = "test_hal.indices"() : () -> (index, index)
  %1:2 = "test_hal.lengths"() : () -> (index, index)
  // Testing things like this is brittle :/
  // Since the canonicalizers are taking these buffer view ops to allocator ops
  // the testing there should cover with the checks here just to make sure the
  // right values from the buffer view are passed in.
  //      CHECK: = hal.buffer_view.dim %[[VIEW]], 1 : index
  //      CHECK: = hal.buffer_view.element_type %[[VIEW]] : i32
  // << A BUNCH OF MATH >>
  %off, %len = hal.buffer_view.compute_range %arg0, indices = [%0#0, %0#1], lengths = [%1#0, %1#1] : index, index
  // CHECK: return
  return %off, %len : index, index
}

// -----

// CHECK-LABEL: func @expand_buffer_view_dims
// CHECK-SAME: %[[VIEW:.+]]: !hal.buffer_view
func @expand_buffer_view_dims(%arg0 : !hal.buffer_view) -> (index, index, index) {
  // CHECK-DAG: %[[D0:.+]] = hal.buffer_view.dim %[[VIEW]], 0 : index
  // CHECK-DAG: %[[D1:.+]] = hal.buffer_view.dim %[[VIEW]], 1 : index
  // CHECK-DAG: %[[D2:.+]] = hal.buffer_view.dim %[[VIEW]], 2 : index
  %0, %1, %2 = hal.buffer_view.dims %arg0 : index, index, index
  // CHECK-NEXT: return %[[D0]], %[[D1]], %[[D2]]
  return %0, %1, %2 : index, index, index
}
