// RUN: tensorlist-opt -split-input-file %s | tensorlist-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @Reserve
func @Reserve(%element_shape: !iree.ref<!hal.buffer_view>, %num_elements: !iree.ref<!hal.buffer_view>) -> !iree.ref<!tensorlist.list> {
  // CHECK: tensorlist.Reserve
  %0 = "tensorlist.Reserve"(%element_shape, %num_elements) : (!iree.ref<!hal.buffer_view>, !iree.ref<!hal.buffer_view>) -> !iree.ref<!tensorlist.list>
  return %0 : !iree.ref<!tensorlist.list>
}

// -----

// CHECK-LABEL: @GetItem
func @GetItem(%list: !iree.ref<!tensorlist.list>, %index: !iree.ref<!hal.buffer_view>, %element_shape: !iree.ref<!hal.buffer_view>) -> !iree.ref<!hal.buffer_view> {
  // CHECK: tensorlist.GetItem
  %0 = "tensorlist.GetItem"(%list, %index, %element_shape) : (!iree.ref<!tensorlist.list>, !iree.ref<!hal.buffer_view>, !iree.ref<!hal.buffer_view>) -> !iree.ref<!hal.buffer_view>
  return %0 : !iree.ref<!hal.buffer_view>
}

// -----

// CHECK-LABEL: @SetItem
func @SetItem(%list: !iree.ref<!tensorlist.list>, %index: !iree.ref<!hal.buffer_view>, %item: !iree.ref<!hal.buffer_view>) -> !iree.ref<!tensorlist.list>{
  // CHECK: tensorlist.SetItem
  %0 = "tensorlist.SetItem"(%list, %index, %item) : (!iree.ref<!tensorlist.list>, !iree.ref<!hal.buffer_view>, !iree.ref<!hal.buffer_view>) -> !iree.ref<!tensorlist.list>
  return %0 : !iree.ref<!tensorlist.list>
}
