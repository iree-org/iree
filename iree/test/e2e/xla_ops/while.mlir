func @while() attributes { iree.module.export }  {
  %start = iree.unfoldable_constant dense<1> : tensor<i32>
  %bound = iree.unfoldable_constant dense<3> : tensor<i32>
  %res = "mhlo.while"(%start) ( {
  ^bb0(%count: tensor<i32>):
    %1 = "mhlo.compare"(%count, %bound) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%count: tensor<i32>):
    %1 = mhlo.add %count, %count : tensor<i32>
    "mhlo.return"(%1) : (tensor<i32>) -> ()
  }) : (tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%res, dense<4> : tensor<i32>) : tensor<i32>
  return
}
