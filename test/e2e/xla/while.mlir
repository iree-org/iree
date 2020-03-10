// A simple while loop example.

// RUN: iree-run-mlir %s -iree-hal-target-backends=vmla --export-all=false | IreeFileCheck %s --implicit-check-not="[" --implicit-check-not="]"
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir %s -iree-hal-target-backends=vulkan-spirv --export-all=false | IreeFileCheck %s --implicit-check-not="[" --implicit-check-not="]")

// CHECK-LABEL: EXEC @main
func @main() -> tensor<i32> attributes { iree.module.export }  {
  %start = iree.unfoldable_constant dense<1> : tensor<i32>
  %bound = iree.unfoldable_constant dense<3> : tensor<i32>
  %res = "xla_hlo.while"(%start) ( {
  ^bb0(%count: tensor<i32>):
    %1 = "xla_hlo.compare"(%count, %bound) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "xla_hlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%count: tensor<i32>):
    %1 = xla_hlo.add %count, %count : tensor<i32>
    "xla_hlo.return"(%1) : (tensor<i32>) -> ()
  }) : (tensor<i32>) -> tensor<i32>

  return %res : tensor<i32>
}

// CHECK: i32=4
