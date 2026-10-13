func.func @sort1D() {
  %input = util.unfoldable_constant dense<[3, 2, 1, 4]> : tensor<4xi32>

  %sort = "stablehlo.sort"(%input) ( {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %compare = "stablehlo.compare"(%arg1, %arg2) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%compare) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64, is_stable = false} : (tensor<4xi32>) -> tensor<4xi32>

  check.expect_eq_const(%sort, dense<[1, 2, 3, 4]> : tensor<4xi32>) : tensor<4xi32>
  return
}

func.func @sort2D() {
  %input = util.unfoldable_constant dense<[[1, 2, 3, 4],
                                           [4, 3, 2, 1]]> : tensor<2x4xi32>

  %sort = "stablehlo.sort"(%input) ( {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %compare = "stablehlo.compare"(%arg1, %arg2) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%compare) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = false} : (tensor<2x4xi32>) -> tensor<2x4xi32>

  check.expect_eq_const(%sort, dense<[[1, 2, 3, 4], [1, 2, 3, 4]]> : tensor<2x4xi32>) : tensor<2x4xi32>
  return
}

func.func @sort3D() {
  %input = util.unfoldable_constant dense<[[[1, 2, 3, 4],
                                            [4, 3, 2, 1]]]> : tensor<1x2x4xi32>

  %sort = "stablehlo.sort"(%input) ( {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %compare = "stablehlo.compare"(%arg1, %arg2) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%compare) : (tensor<i1>) -> ()
  }) {dimension = 2 : i64, is_stable = false} : (tensor<1x2x4xi32>) -> tensor<1x2x4xi32>

  check.expect_eq_const(%sort, dense<[[[1, 2, 3, 4], [1, 2, 3, 4]]]> : tensor<1x2x4xi32>) : tensor<1x2x4xi32>
  return
}

func.func @sort_to_decreasing_seq() {
  %input = util.unfoldable_constant dense<[3, 2, 1, 4]> : tensor<4xi32>

  %sort = "stablehlo.sort"(%input) ( {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %compare = "stablehlo.compare"(%arg1, %arg2) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%compare) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64, is_stable = false} : (tensor<4xi32>) -> tensor<4xi32>

  check.expect_eq_const(%sort, dense<[4, 3, 2, 1]> : tensor<4xi32>) : tensor<4xi32>
  return
}

func.func @sort_stable_equal_keys() {
  %keys = util.unfoldable_constant dense<[1, 0, 1, 0, 2]> : tensor<5xi32>
  %values = util.unfoldable_constant dense<[10, 20, 30, 40, 50]> : tensor<5xi32>

  %sort:2 = "stablehlo.sort"(%keys, %values) ( {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
    %compare = "stablehlo.compare"(%arg1, %arg2) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%compare) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64, is_stable = true} : (tensor<5xi32>, tensor<5xi32>) -> (tensor<5xi32>, tensor<5xi32>)

  check.expect_eq_const(%sort#0, dense<[0, 0, 1, 1, 2]> : tensor<5xi32>) : tensor<5xi32>
  check.expect_eq_const(%sort#1, dense<[20, 40, 10, 30, 50]> : tensor<5xi32>) : tensor<5xi32>
  return
}
