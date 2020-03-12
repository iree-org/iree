// RUN: check-translate --iree-hal-target-backends=vmla -iree-mlir-to-vm-bytecode-module %s | iree-check-module --driver=vmla
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (check-translate --iree-hal-target-backends=vulkan-spirv -iree-mlir-to-vm-bytecode-module %s | iree-check-module --driver=vulkan)

func @expect_true() attributes { iree.module.export } {
  %true = iree.unfoldable_constant 1 : i32
  check.expect_true(%true) : i32
  return
}

func @expect_false() attributes { iree.module.export } {
  %false = iree.unfoldable_constant 0 : i32
  check.expect_false(%false) : i32
  return
}

func @expect_all_true() attributes {iree.module.export} {
  %dev = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator %dev : !hal.allocator
  %all_true = hal.buffer_view.const %allocator, "HostLocal|DeviceVisible", "All" : !hal.buffer_view = dense<1> : tensor<2x2xi32>
  check.expect_all_true(%all_true) : !hal.buffer_view
  return
}

func @expect_all_true_tensor() attributes { iree.module.export } {
  %all_true = iree.unfoldable_constant dense<1> : tensor<2x2xi32>
  check.expect_all_true(%all_true) : tensor<2x2xi32>
  return
}

func @abs() attributes { iree.module.export } {
  %cm5 = iree.unfoldable_constant dense<-5> : tensor<i32>
  %result = "xla_hlo.abs"(%cm5) : (tensor<i32>) -> tensor<i32>
  %c5 = iree.unfoldable_constant dense<5> : tensor<i32>
  check.expect_eq(%result, %c5) : tensor<i32>
  return
}

