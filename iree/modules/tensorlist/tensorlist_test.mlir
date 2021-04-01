func @identity_through_set_item_get_item(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.module.export, iree.abi.none} {
  %device = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
  %0 = hal.allocator.constant<%allocator : !hal.allocator>
         type("HostLocal|DeviceVisible") usage("All") : !hal.buffer_view =
         dense<1> : tensor<i32>
  %1 = hal.allocator.constant<%allocator : !hal.allocator>
         type("HostLocal|DeviceVisible") usage("All") : !hal.buffer_view =
         dense<[]> : tensor<0xi32>
  %2 = hal.allocator.constant<%allocator : !hal.allocator>
         type("HostLocal|DeviceVisible") usage("All") : !hal.buffer_view =
         dense<0> : tensor<i32>
  %3 = "tensorlist.Reserve"(%1, %0) { element_type = 50331680 : i32} : (!hal.buffer_view, !hal.buffer_view) -> !tensorlist.list
  %4 = "tensorlist.SetItem"(%3, %2, %arg0) : (!tensorlist.list, !hal.buffer_view, !hal.buffer_view) -> !tensorlist.list
  %5 = "tensorlist.GetItem"(%4, %2) : (!tensorlist.list, !hal.buffer_view) -> !hal.buffer_view
  return %5 : !hal.buffer_view
}

func @identity_through_set_item_get_item_2D(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.module.export, iree.abi.none} {
  %device = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
  %0 = hal.allocator.constant<%allocator : !hal.allocator>
         type("HostLocal|DeviceVisible") usage("All") : !hal.buffer_view =
         dense<1> : tensor<i32>
  %1 = hal.allocator.constant<%allocator : !hal.allocator>
         type("HostLocal|DeviceVisible") usage("All") : !hal.buffer_view =
         dense<[1, 1]> : tensor<2xi32>
  %2 = hal.allocator.constant<%allocator : !hal.allocator>
         type("HostLocal|DeviceVisible") usage("All") : !hal.buffer_view =
         dense<0> : tensor<i32>
  %3 = "tensorlist.Reserve"(%1, %0) { element_type = 50331680 : i32} : (!hal.buffer_view, !hal.buffer_view) -> !tensorlist.list
  %4 = "tensorlist.SetItem"(%3, %2, %arg0) : (!tensorlist.list, !hal.buffer_view, !hal.buffer_view) -> !tensorlist.list
  %stacked = "tensorlist.Stack"(%allocator, %4, %0) : (!hal.allocator, !tensorlist.list, !hal.buffer_view) -> !hal.buffer_view
  return %stacked : !hal.buffer_view
}

func @identity_through_concat(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.module.export, iree.abi.none} {
  %device = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
  %element_shape = hal.allocator.constant<%allocator : !hal.allocator>
         type("HostLocal|DeviceVisible") usage("All") : !hal.buffer_view =
         dense<[]> : tensor<0xi32>
  %list = "tensorlist.FromTensor"(%arg0) : (!hal.buffer_view) -> !tensorlist.list
  %concat = "tensorlist.Concat"(%allocator, %list) : (!hal.allocator, !tensorlist.list) -> !hal.buffer_view
  return %concat : !hal.buffer_view
}

func @concat_appends_empty(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.module.export, iree.abi.none} {
  %device = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
  %0 = hal.allocator.constant<%allocator : !hal.allocator>
         type("HostLocal|DeviceVisible") usage("All") : !hal.buffer_view =
         dense<2> : tensor<i32>
  %1 = hal.allocator.constant<%allocator : !hal.allocator>
         type("HostLocal|DeviceVisible") usage("All") : !hal.buffer_view =
         dense<[1]> : tensor<1xi32>
  %2 = hal.allocator.constant<%allocator : !hal.allocator>
         type("HostLocal|DeviceVisible") usage("All") : !hal.buffer_view =
         dense<0> : tensor<i32>
  %3 = "tensorlist.Reserve"(%1, %0) { element_type = 50331680 : i32} : (!hal.buffer_view, !hal.buffer_view) -> !tensorlist.list
  %4 = "tensorlist.SetItem"(%3, %2, %arg0) : (!tensorlist.list, !hal.buffer_view, !hal.buffer_view) -> !tensorlist.list
  %concat = "tensorlist.Concat"(%allocator, %4) : (!hal.allocator, !tensorlist.list) -> !hal.buffer_view
  return %concat : !hal.buffer_view
}

func @identity_through_stack(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.module.export, iree.abi.none} {
  %device = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
  %num_elements = hal.allocator.constant<%allocator : !hal.allocator>
         type("HostLocal|DeviceVisible") usage("All") : !hal.buffer_view =
         dense<2> : tensor<i32>
  %list = "tensorlist.FromTensor"(%arg0) : (!hal.buffer_view) -> !tensorlist.list
  %stacked = "tensorlist.Stack"(%allocator, %list, %num_elements) : (!hal.allocator, !tensorlist.list, !hal.buffer_view) -> !hal.buffer_view
  return %stacked : !hal.buffer_view
}

func @stack_appends_empty(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.module.export, iree.abi.none} {
  %device = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
  %0 = hal.allocator.constant<%allocator : !hal.allocator>
         type("HostLocal|DeviceVisible") usage("All") : !hal.buffer_view =
         dense<2> : tensor<i32>
  %1 = hal.allocator.constant<%allocator : !hal.allocator>
         type("HostLocal|DeviceVisible") usage("All") : !hal.buffer_view =
         dense<[]> : tensor<0xi32>
  %2 = hal.allocator.constant<%allocator : !hal.allocator>
         type("HostLocal|DeviceVisible") usage("All") : !hal.buffer_view =
         dense<0> : tensor<i32>
  %3 = "tensorlist.Reserve"(%1, %0) { element_type = 50331680 : i32} : (!hal.buffer_view, !hal.buffer_view) -> !tensorlist.list
  %4 = "tensorlist.SetItem"(%3, %2, %arg0) : (!tensorlist.list, !hal.buffer_view, !hal.buffer_view) -> !tensorlist.list
  %stacked = "tensorlist.Stack"(%allocator, %4, %0) : (!hal.allocator, !tensorlist.list, !hal.buffer_view) -> !hal.buffer_view
  return %stacked : !hal.buffer_view
}
