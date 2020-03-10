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

func @abs() attributes { iree.module.export } {
  %cm5 = iree.unfoldable_constant dense<-5> : tensor<i32>
  %result = "xla_hlo.abs"(%cm5) : (tensor<i32>) -> tensor<i32>
  %c5 = iree.unfoldable_constant dense<5> : tensor<i32>
  %eq = "xla_hlo.compare"(%result, %c5) {comparison_direction = "EQ"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %eq_el = extract_element %eq[] : tensor<i1>
  check.expect_true(%eq_el) : i1
  return
}

