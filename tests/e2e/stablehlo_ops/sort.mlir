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

func.func @sort_nan_olt_stable() {
  %keys = util.unfoldable_constant dense<[2.0, 0x7FC00000, 1.0, 1.0, 0.0]> : tensor<5xf32>
  %values = util.unfoldable_constant dense<[0, 1, 2, 3, 4]> : tensor<5xi32>
  %sort:2 = "stablehlo.sort"(%keys, %values) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>, %lhs_value: tensor<i32>, %rhs_value: tensor<i32>):
    %cmp = stablehlo.compare LT, %lhs, %rhs, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<5xf32>, tensor<5xi32>) -> (tensor<5xf32>, tensor<5xi32>)
  // Compare bits so NaNs are checked without floating-point equality.
  %bits = arith.bitcast %sort#0 : tensor<5xf32> to tensor<5xi32>
  check.expect_eq_const(%bits, dense<[1073741824, 2143289344, 0, 1065353216, 1065353216]> : tensor<5xi32>) : tensor<5xi32>
  check.expect_eq_const(%sort#1, dense<[0, 1, 4, 2, 3]> : tensor<5xi32>) : tensor<5xi32>
  return
}

func.func @sort_nan_olt_unstable() {
  %keys = util.unfoldable_constant dense<[2.0, 0x7FC00000, 1.0, 1.0, 0.0]> : tensor<5xf32>
  %values = util.unfoldable_constant dense<[0, 1, 2, 3, 4]> : tensor<5xi32>
  %sort:2 = "stablehlo.sort"(%keys, %values) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>, %lhs_value: tensor<i32>, %rhs_value: tensor<i32>):
    %cmp = stablehlo.compare LT, %lhs, %rhs, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }) {dimension = 0 : i64, is_stable = false} : (tensor<5xf32>, tensor<5xi32>) -> (tensor<5xf32>, tensor<5xi32>)
  // Compare bits so NaNs are checked without floating-point equality.
  %bits = arith.bitcast %sort#0 : tensor<5xf32> to tensor<5xi32>
  check.expect_eq_const(%bits, dense<[0, 2143289344, 1065353216, 1065353216, 1073741824]> : tensor<5xi32>) : tensor<5xi32>
  check.expect_eq_const(%sort#1, dense<[4, 1, 3, 2, 0]> : tensor<5xi32>) : tensor<5xi32>
  return
}

func.func @sort_nan_ogt_stable() {
  %keys = util.unfoldable_constant dense<[0.0, 0x7FC00000, 1.0, 1.0, 2.0]> : tensor<5xf32>
  %values = util.unfoldable_constant dense<[0, 1, 2, 3, 4]> : tensor<5xi32>
  %sort:2 = "stablehlo.sort"(%keys, %values) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>, %lhs_value: tensor<i32>, %rhs_value: tensor<i32>):
    %cmp = stablehlo.compare GT, %lhs, %rhs, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<5xf32>, tensor<5xi32>) -> (tensor<5xf32>, tensor<5xi32>)
  // Compare bits so NaNs are checked without floating-point equality.
  %bits = arith.bitcast %sort#0 : tensor<5xf32> to tensor<5xi32>
  check.expect_eq_const(%bits, dense<[0, 2143289344, 1073741824, 1065353216, 1065353216]> : tensor<5xi32>) : tensor<5xi32>
  check.expect_eq_const(%sort#1, dense<[0, 1, 4, 2, 3]> : tensor<5xi32>) : tensor<5xi32>
  return
}

func.func @sort_nan_ogt_unstable() {
  %keys = util.unfoldable_constant dense<[0.0, 0x7FC00000, 1.0, 1.0, 2.0]> : tensor<5xf32>
  %values = util.unfoldable_constant dense<[0, 1, 2, 3, 4]> : tensor<5xi32>
  %sort:2 = "stablehlo.sort"(%keys, %values) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>, %lhs_value: tensor<i32>, %rhs_value: tensor<i32>):
    %cmp = stablehlo.compare GT, %lhs, %rhs, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }) {dimension = 0 : i64, is_stable = false} : (tensor<5xf32>, tensor<5xi32>) -> (tensor<5xf32>, tensor<5xi32>)
  // Compare bits so NaNs are checked without floating-point equality.
  %bits = arith.bitcast %sort#0 : tensor<5xf32> to tensor<5xi32>
  check.expect_eq_const(%bits, dense<[1073741824, 2143289344, 1065353216, 1065353216, 0]> : tensor<5xi32>) : tensor<5xi32>
  check.expect_eq_const(%sort#1, dense<[4, 1, 3, 2, 0]> : tensor<5xi32>) : tensor<5xi32>
  return
}

func.func @sort_nan_ult_stable() {
  %keys = util.unfoldable_constant dense<[2.0, 0x7FC00000, 1.0, 1.0, 0.0]> : tensor<5xf32>
  %values = util.unfoldable_constant dense<[0, 1, 2, 3, 4]> : tensor<5xi32>
  %sort:2 = "stablehlo.sort"(%keys, %values) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>, %lhs_value: tensor<i32>, %rhs_value: tensor<i32>):
    // IEEE f32 NaNs have an absolute bit pattern greater than infinity.
    // Use integer classification so the unordered predicate is explicit.
    %abs_mask = stablehlo.constant dense<2147483647> : tensor<i32>
    %inf_bits = stablehlo.constant dense<2139095040> : tensor<i32>
    %lhs_bits = "stablehlo.bitcast_convert"(%lhs) : (tensor<f32>) -> tensor<i32>
    %rhs_bits = "stablehlo.bitcast_convert"(%rhs) : (tensor<f32>) -> tensor<i32>
    %lhs_abs = stablehlo.and %lhs_bits, %abs_mask : tensor<i32>
    %rhs_abs = stablehlo.and %rhs_bits, %abs_mask : tensor<i32>
    %lhs_nan = stablehlo.compare GT, %lhs_abs, %inf_bits, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %rhs_nan = stablehlo.compare GT, %rhs_abs, %inf_bits, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %less = stablehlo.compare LT, %lhs, %rhs, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %unordered = stablehlo.or %lhs_nan, %rhs_nan : tensor<i1>
    %cmp = stablehlo.or %unordered, %less : tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<5xf32>, tensor<5xi32>) -> (tensor<5xf32>, tensor<5xi32>)
  // Compare bits so NaNs are checked without floating-point equality.
  %bits = arith.bitcast %sort#0 : tensor<5xf32> to tensor<5xi32>
  check.expect_eq_const(%bits, dense<[1073741824, 2143289344, 0, 1065353216, 1065353216]> : tensor<5xi32>) : tensor<5xi32>
  check.expect_eq_const(%sort#1, dense<[0, 1, 4, 2, 3]> : tensor<5xi32>) : tensor<5xi32>
  return
}

func.func @sort_nan_ult_unstable() {
  %keys = util.unfoldable_constant dense<[2.0, 0x7FC00000, 1.0, 1.0, 0.0]> : tensor<5xf32>
  %values = util.unfoldable_constant dense<[0, 1, 2, 3, 4]> : tensor<5xi32>
  %sort:2 = "stablehlo.sort"(%keys, %values) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>, %lhs_value: tensor<i32>, %rhs_value: tensor<i32>):
    // IEEE f32 NaNs have an absolute bit pattern greater than infinity.
    // Use integer classification so the unordered predicate is explicit.
    %abs_mask = stablehlo.constant dense<2147483647> : tensor<i32>
    %inf_bits = stablehlo.constant dense<2139095040> : tensor<i32>
    %lhs_bits = "stablehlo.bitcast_convert"(%lhs) : (tensor<f32>) -> tensor<i32>
    %rhs_bits = "stablehlo.bitcast_convert"(%rhs) : (tensor<f32>) -> tensor<i32>
    %lhs_abs = stablehlo.and %lhs_bits, %abs_mask : tensor<i32>
    %rhs_abs = stablehlo.and %rhs_bits, %abs_mask : tensor<i32>
    %lhs_nan = stablehlo.compare GT, %lhs_abs, %inf_bits, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %rhs_nan = stablehlo.compare GT, %rhs_abs, %inf_bits, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %less = stablehlo.compare LT, %lhs, %rhs, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %unordered = stablehlo.or %lhs_nan, %rhs_nan : tensor<i1>
    %cmp = stablehlo.or %unordered, %less : tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }) {dimension = 0 : i64, is_stable = false} : (tensor<5xf32>, tensor<5xi32>) -> (tensor<5xf32>, tensor<5xi32>)
  // Compare bits so NaNs are checked without floating-point equality.
  %bits = arith.bitcast %sort#0 : tensor<5xf32> to tensor<5xi32>
  check.expect_eq_const(%bits, dense<[1073741824, 2143289344, 0, 1065353216, 1065353216]> : tensor<5xi32>) : tensor<5xi32>
  check.expect_eq_const(%sort#1, dense<[0, 1, 4, 3, 2]> : tensor<5xi32>) : tensor<5xi32>
  return
}

// TOTALORDER puts identical positive quiet NaNs after finite keys.
func.func @sort_nan_totalorder_stable() {
  %keys = util.unfoldable_constant dense<[0x7FC00000, 1.0, 0x7FC00000, 0.0, 1.0]> : tensor<5xf32>
  %values = util.unfoldable_constant dense<[0, 1, 2, 3, 4]> : tensor<5xi32>
  %sort:2 = "stablehlo.sort"(%keys, %values) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>, %lhs_value: tensor<i32>, %rhs_value: tensor<i32>):
    %cmp = stablehlo.compare LT, %lhs, %rhs, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<5xf32>, tensor<5xi32>) -> (tensor<5xf32>, tensor<5xi32>)
  // Compare bits so NaNs are checked without floating-point equality.
  %bits = arith.bitcast %sort#0 : tensor<5xf32> to tensor<5xi32>
  check.expect_eq_const(%bits, dense<[0, 1065353216, 1065353216, 2143289344, 2143289344]> : tensor<5xi32>) : tensor<5xi32>
  check.expect_eq_const(%sort#1, dense<[3, 1, 4, 0, 2]> : tensor<5xi32>) : tensor<5xi32>
  return
}

// TOTALORDER puts identical positive quiet NaNs after finite keys.
func.func @sort_nan_totalorder_unstable() {
  %keys = util.unfoldable_constant dense<[0x7FC00000, 1.0, 0x7FC00000, 0.0, 1.0]> : tensor<5xf32>
  %values = util.unfoldable_constant dense<[0, 1, 2, 3, 4]> : tensor<5xi32>
  %sort:2 = "stablehlo.sort"(%keys, %values) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>, %lhs_value: tensor<i32>, %rhs_value: tensor<i32>):
    %cmp = stablehlo.compare LT, %lhs, %rhs, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }) {dimension = 0 : i64, is_stable = false} : (tensor<5xf32>, tensor<5xi32>) -> (tensor<5xf32>, tensor<5xi32>)
  // Compare bits so NaNs are checked without floating-point equality.
  %bits = arith.bitcast %sort#0 : tensor<5xf32> to tensor<5xi32>
  check.expect_eq_const(%bits, dense<[0, 1065353216, 1065353216, 2143289344, 2143289344]> : tensor<5xi32>) : tensor<5xi32>
  check.expect_eq_const(%sort#1, dense<[3, 4, 1, 2, 0]> : tensor<5xi32>) : tensor<5xi32>
  return
}
