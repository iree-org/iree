func @identity_through_tensorlist(%arg0: !iree.ref<!hal.buffer_view>) -> !iree.ref<!hal.buffer_view> attributes {iree.module.export} {
  %dev = hal.ex.shared_device : !iree.ref<!hal.device>
  %allocator = hal.device.allocator %dev : !iree.ref<!hal.allocator>
  %0 = hal.buffer_view.const %allocator, "HostLocal|DeviceVisible", "All" : !iree.ref<!hal.buffer_view> = dense<1> : tensor<i32>
  %1 = hal.buffer_view.const %allocator, "HostLocal|DeviceVisible", "All" : !iree.ref<!hal.buffer_view> = dense<[]> : tensor<0xi32>
  %2 = hal.buffer_view.const %allocator, "HostLocal|DeviceVisible", "All" : !iree.ref<!hal.buffer_view> = dense<0> : tensor<i32>
  %3 = "tensorlist.Reserve"(%1, %0) : (!iree.ref<!hal.buffer_view>, !iree.ref<!hal.buffer_view>) -> !iree.ref<!tensorlist.list>
  %4 = "tensorlist.SetItem"(%3, %2, %arg0) : (!iree.ref<!tensorlist.list>, !iree.ref<!hal.buffer_view>, !iree.ref<!hal.buffer_view>) -> !iree.ref<!tensorlist.list>
  %5 = "tensorlist.GetItem"(%4, %2, %1) : (!iree.ref<!tensorlist.list>, !iree.ref<!hal.buffer_view>, !iree.ref<!hal.buffer_view>) -> !iree.ref<!hal.buffer_view>
  return %5 : !iree.ref<!hal.buffer_view>
}
