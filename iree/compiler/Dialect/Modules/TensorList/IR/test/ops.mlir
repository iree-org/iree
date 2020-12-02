// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @Reserve
func @Reserve(%element_shape: !hal.buffer_view, %num_elements: !hal.buffer_view) -> !tensorlist.list {
  // CHECK: tensorlist.Reserve
  %0 = "tensorlist.Reserve"(%element_shape, %num_elements) {element_type = 13 : i32 } : (!hal.buffer_view, !hal.buffer_view) -> !tensorlist.list
  return %0 : !tensorlist.list
}

// -----

// CHECK-LABEL: @GetItem
func @GetItem(%list: !tensorlist.list, %index: !hal.buffer_view) -> !hal.buffer_view {
  // CHECK: tensorlist.GetItem
  %0 = "tensorlist.GetItem"(%list, %index) : (!tensorlist.list, !hal.buffer_view) -> !hal.buffer_view
  return %0 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @SetItem
func @SetItem(%list: !tensorlist.list, %index: !hal.buffer_view, %item: !hal.buffer_view) -> !tensorlist.list {
  // CHECK: tensorlist.SetItem
  %0 = "tensorlist.SetItem"(%list, %index, %item) : (!tensorlist.list, !hal.buffer_view, !hal.buffer_view) -> !tensorlist.list
  return %0 : !tensorlist.list
}

// -----

// CHECK-LABEL: @Stack
func @Stack(%allocator: !hal.allocator, %list: !tensorlist.list, %num_elements: !hal.buffer_view) -> !hal.buffer_view {
  // CHECK: tensorlist.Stack
  %0 = "tensorlist.Stack"(%allocator, %list, %num_elements) : (!hal.allocator, !tensorlist.list, !hal.buffer_view) -> !hal.buffer_view
  return %0 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @Concat
func @Concat(%allocator: !hal.allocator, %list: !tensorlist.list) -> !hal.buffer_view {
  // CHECK: tensorlist.Concat
  %0 = "tensorlist.Concat"(%allocator, %list) : (!hal.allocator, !tensorlist.list) -> !hal.buffer_view
  return %0 : !hal.buffer_view
}
