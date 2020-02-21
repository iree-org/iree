func @identity_through_tensorlist(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.module.export} {
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
