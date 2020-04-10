func @while() attributes { iree.module.export }  {
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
  check.expect_eq_const(%res, dense<4> : tensor<i32>) : tensor<i32>
  return
}
