func @identity_through_set_item_get_item(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.module.export} {
  %dev = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator %dev : !hal.allocator
  %0 = hal.buffer_view.const %allocator, "HostLocal|DeviceVisible", "All" : !hal.buffer_view = dense<1> : tensor<i32>
  %1 = hal.buffer_view.const %allocator, "HostLocal|DeviceVisible", "All" : !hal.buffer_view = dense<[]> : tensor<0xi32>
  %2 = hal.buffer_view.const %allocator, "HostLocal|DeviceVisible", "All" : !hal.buffer_view = dense<0> : tensor<i32>
  %3 = "tensorlist.Reserve"(%1, %0) : (!hal.buffer_view, !hal.buffer_view) -> !tensorlist.list
  %4 = "tensorlist.SetItem"(%3, %2, %arg0) : (!tensorlist.list, !hal.buffer_view, !hal.buffer_view) -> !tensorlist.list
  %5 = "tensorlist.GetItem"(%4, %2, %1) : (!tensorlist.list, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  return %5 : !hal.buffer_view
}

func @identity_through_stack(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.module.export} {
  %dev = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator %dev : !hal.allocator
  %element_shape = hal.buffer_view.const %allocator, "HostLocal|DeviceVisible", "All" : !hal.buffer_view = dense<[]> : tensor<0xi32>
  %num_elements = hal.buffer_view.const %allocator, "HostLocal|DeviceVisible", "All" : !hal.buffer_view = dense<2> : tensor<i32>
  %list = "tensorlist.FromTensor"(%arg0, %element_shape) : (!hal.buffer_view, !hal.buffer_view) -> !tensorlist.list
  %stacked = "tensorlist.Stack"(%list, %element_shape, %num_elements) : (!tensorlist.list, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  return %stacked : !hal.buffer_view
}
