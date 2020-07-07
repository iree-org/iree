// RUN: iree-translate --iree-hal-target-backends=vmla -iree-mlir-to-vm-bytecode-module %s | iree-check-module --driver=vmla -
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-translate --iree-hal-target-backends=vulkan-spirv -iree-mlir-to-vm-bytecode-module %s | iree-check-module --driver=vulkan -)

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

func @expect_eq() attributes { iree.module.export } {
  %const0 = iree.unfoldable_constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  %const1 = iree.unfoldable_constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  check.expect_eq(%const0, %const1) : tensor<5xi32>
  return
}

func @expect_eq_const() attributes { iree.module.export } {
  %const0 = iree.unfoldable_constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  check.expect_eq_const(%const0, dense<[1, 2, 3, 4, 5]> : tensor<5xi32>) : tensor<5xi32>
  return
}

func @expect_almost_eq() attributes { iree.module.export } {
  %const0 = iree.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf32>
  %const1 = iree.unfoldable_constant dense<[0.999999, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf32>
  check.expect_almost_eq(%const0, %const1) : tensor<5xf32>
  return
}

func @expect_almost_eq_const() attributes { iree.module.export } {
  %const0 = iree.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf32>
  check.expect_almost_eq_const(%const0, dense<[0.999999, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf32>) : tensor<5xf32>
  return
}

func @add() attributes { iree.module.export } {
  %c5 = iree.unfoldable_constant dense<5> : tensor<i32>
  %result = "mhlo.add"(%c5, %c5) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %c10 = iree.unfoldable_constant dense<10> : tensor<i32>
  check.expect_eq(%result, %c10) : tensor<i32>
  return
}

func @floats() attributes { iree.module.export } {
  %cp1 = iree.unfoldable_constant dense<0.1> : tensor<f32>
  %c1 = iree.unfoldable_constant dense<1.0> : tensor<f32>
  %p2 = "mhlo.add"(%cp1, %cp1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %p3 = "mhlo.add"(%p2, %cp1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %p4 = "mhlo.add"(%p3, %cp1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %p5 = "mhlo.add"(%p4, %cp1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %p6 = "mhlo.add"(%p5, %cp1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %p7 = "mhlo.add"(%p6, %cp1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %p8 = "mhlo.add"(%p7, %cp1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %p9 = "mhlo.add"(%p8, %cp1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %approximately_1 = "mhlo.add"(%p9, %cp1) : (tensor<f32>, tensor<f32>) -> tensor<f32>

  check.expect_almost_eq(%approximately_1, %c1) : tensor<f32>
  return
}
