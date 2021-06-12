func @compare_tensor() {
  %lhs = iree.unfoldable_constant dense<[1, 2, 7, 4]> : tensor<4xi32>
  %rhs = iree.unfoldable_constant dense<[5, 2, 3, 4]> : tensor<4xi32>
  %result = "mhlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %c0 = iree.unfoldable_constant dense<0> : tensor<4xi8>
  %c1 = iree.unfoldable_constant dense<1> : tensor<4xi8>
  %output = "mhlo.select"(%result, %c1, %c0) : (tensor<4xi1>, tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  check.expect_eq_const(%output, dense<[0, 1, 0, 1]> : tensor<4xi8>) : tensor<4xi8>
  return
}

func @compare_scalar() {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i32>
  %rhs = iree.unfoldable_constant dense<5> : tensor<i32>
  %result = "mhlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %c0 = iree.unfoldable_constant dense<0> : tensor<i8>
  %c1 = iree.unfoldable_constant dense<1> : tensor<i8>
  %output = "mhlo.select"(%result, %c1, %c0) : (tensor<i1>, tensor<i8>, tensor<i8>) -> tensor<i8>
  check.expect_eq_const(%output, dense<0> : tensor<i8>) : tensor<i8>
  return
}

func @compare_i8() {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i8>
  %rhs = iree.unfoldable_constant dense<5> : tensor<i8>
  %result = "mhlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<i8>, tensor<i8>) -> tensor<i1>
  %c0 = iree.unfoldable_constant dense<0> : tensor<i8>
  %c1 = iree.unfoldable_constant dense<1> : tensor<i8>
  %output = "mhlo.select"(%result, %c1, %c0) : (tensor<i1>, tensor<i8>, tensor<i8>) -> tensor<i8>
  check.expect_eq_const(%output, dense<0> : tensor<i8>) : tensor<i8>
  return
}

func @compare_i16() {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i16>
  %rhs = iree.unfoldable_constant dense<5> : tensor<i16>
  %result = "mhlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<i16>, tensor<i16>) -> tensor<i1>
  %c0 = iree.unfoldable_constant dense<0> : tensor<i8>
  %c1 = iree.unfoldable_constant dense<1> : tensor<i8>
  %output = "mhlo.select"(%result, %c1, %c0) : (tensor<i1>, tensor<i8>, tensor<i8>) -> tensor<i8>
  check.expect_eq_const(%output, dense<0> : tensor<i8>) : tensor<i8>
  return
}

func @compare_i32() {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i32>
  %rhs = iree.unfoldable_constant dense<5> : tensor<i32>
  %result = "mhlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %c0 = iree.unfoldable_constant dense<0> : tensor<i8>
  %c1 = iree.unfoldable_constant dense<1> : tensor<i8>
  %output = "mhlo.select"(%result, %c1, %c0) : (tensor<i1>, tensor<i8>, tensor<i8>) -> tensor<i8>
  check.expect_eq_const(%output, dense<0> : tensor<i8>) : tensor<i8>
  return
}

func @compare_i64() {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i64>
  %rhs = iree.unfoldable_constant dense<5> : tensor<i64>
  %result = "mhlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %c0 = iree.unfoldable_constant dense<0> : tensor<i8>
  %c1 = iree.unfoldable_constant dense<1> : tensor<i8>
  %output = "mhlo.select"(%result, %c1, %c0) : (tensor<i1>, tensor<i8>, tensor<i8>) -> tensor<i8>
  check.expect_eq_const(%output, dense<0> : tensor<i8>) : tensor<i8>
  return
}

func @compare_f32() {
  %lhs = iree.unfoldable_constant dense<1.0> : tensor<f32>
  %rhs = iree.unfoldable_constant dense<5.0> : tensor<f32>
  %result = "mhlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %c0 = iree.unfoldable_constant dense<0> : tensor<i8>
  %c1 = iree.unfoldable_constant dense<1> : tensor<i8>
  %output = "mhlo.select"(%result, %c1, %c0) : (tensor<i1>, tensor<i8>, tensor<i8>) -> tensor<i8>
  check.expect_eq_const(%output, dense<0> : tensor<i8>) : tensor<i8>
  return
}

func @compare_f64() {
  %lhs = iree.unfoldable_constant dense<1.0> : tensor<f64>
  %rhs = iree.unfoldable_constant dense<5.0> : tensor<f64>
  %result = "mhlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<f64>, tensor<f64>) -> tensor<i1>
  %c0 = iree.unfoldable_constant dense<0> : tensor<i8>
  %c1 = iree.unfoldable_constant dense<1> : tensor<i8>
  %output = "mhlo.select"(%result, %c1, %c0) : (tensor<i1>, tensor<i8>, tensor<i8>) -> tensor<i8>
  check.expect_eq_const(%output, dense<0> : tensor<i8>) : tensor<i8>
  return
}

func @compare_tensor_odd_length() {
  %lhs = iree.unfoldable_constant dense<[1, 2, 7]> : tensor<3xi32>
  %rhs = iree.unfoldable_constant dense<[5, 2, 3]> : tensor<3xi32>
  %result = "mhlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  %c0 = iree.unfoldable_constant dense<0> : tensor<3xi8>
  %c1 = iree.unfoldable_constant dense<1> : tensor<3xi8>
  %output = "mhlo.select"(%result, %c1, %c0) : (tensor<3xi1>, tensor<3xi8>, tensor<3xi8>) -> tensor<3xi8>
  check.expect_eq_const(%output, dense<[0, 1, 0]> : tensor<3xi8>) : tensor<3xi8>
  return
}

func @compare_eq() {
  %lhs = iree.unfoldable_constant dense<[1, 2, 7, 4]> : tensor<4xi32>
  %rhs = iree.unfoldable_constant dense<[5, 2, 3, 4]> : tensor<4xi32>
  %result = "mhlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %c0 = iree.unfoldable_constant dense<0> : tensor<4xi8>
  %c1 = iree.unfoldable_constant dense<1> : tensor<4xi8>
  %output = "mhlo.select"(%result, %c1, %c0) : (tensor<4xi1>, tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  check.expect_eq_const(%output, dense<[0, 1, 0, 1]> : tensor<4xi8>) : tensor<4xi8>
  return
}

func @compare_ne() {
  %lhs = iree.unfoldable_constant dense<[1, 2, 7, 4]> : tensor<4xi32>
  %rhs = iree.unfoldable_constant dense<[5, 2, 3, 4]> : tensor<4xi32>
  %result = "mhlo.compare"(%lhs, %rhs) {comparison_direction = "NE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %c0 = iree.unfoldable_constant dense<0> : tensor<4xi8>
  %c1 = iree.unfoldable_constant dense<1> : tensor<4xi8>
  %output = "mhlo.select"(%result, %c1, %c0) : (tensor<4xi1>, tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  check.expect_eq_const(%output, dense<[1, 0, 1, 0]> : tensor<4xi8>) : tensor<4xi8>
  return
}

func @compare_lt() {
  %lhs = iree.unfoldable_constant dense<[1, 2, 7, 4]> : tensor<4xi32>
  %rhs = iree.unfoldable_constant dense<[5, 2, 3, 4]> : tensor<4xi32>
  %result = "mhlo.compare"(%lhs, %rhs) {comparison_direction = "LT"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %c0 = iree.unfoldable_constant dense<0> : tensor<4xi8>
  %c1 = iree.unfoldable_constant dense<1> : tensor<4xi8>
  %output = "mhlo.select"(%result, %c1, %c0) : (tensor<4xi1>, tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  check.expect_eq_const(%output, dense<[1, 0, 0, 0]> : tensor<4xi8>) : tensor<4xi8>
  return
}

func @compare_le() {
  %lhs = iree.unfoldable_constant dense<[1, 2, 7, 4]> : tensor<4xi32>
  %rhs = iree.unfoldable_constant dense<[5, 2, 3, 4]> : tensor<4xi32>
  %result = "mhlo.compare"(%lhs, %rhs) {comparison_direction = "LE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %c0 = iree.unfoldable_constant dense<0> : tensor<4xi8>
  %c1 = iree.unfoldable_constant dense<1> : tensor<4xi8>
  %output = "mhlo.select"(%result, %c1, %c0) : (tensor<4xi1>, tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  check.expect_eq_const(%output, dense<[1, 1, 0, 1]> : tensor<4xi8>) : tensor<4xi8>
  return
}

func @compare_gt() {
  %lhs = iree.unfoldable_constant dense<[1, 2, 7, 4]> : tensor<4xi32>
  %rhs = iree.unfoldable_constant dense<[5, 2, 3, 4]> : tensor<4xi32>
  %result = "mhlo.compare"(%lhs, %rhs) {comparison_direction = "GT"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %c0 = iree.unfoldable_constant dense<0> : tensor<4xi8>
  %c1 = iree.unfoldable_constant dense<1> : tensor<4xi8>
  %output = "mhlo.select"(%result, %c1, %c0) : (tensor<4xi1>, tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  check.expect_eq_const(%output, dense<[0, 0, 1, 0]> : tensor<4xi8>) : tensor<4xi8>
  return
}

func @compare_ge() {
  %lhs = iree.unfoldable_constant dense<[1, 2, 7, 4]> : tensor<4xi32>
  %rhs = iree.unfoldable_constant dense<[5, 2, 3, 4]> : tensor<4xi32>
  %result = "mhlo.compare"(%lhs, %rhs) {comparison_direction = "GE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %c0 = iree.unfoldable_constant dense<0> : tensor<4xi8>
  %c1 = iree.unfoldable_constant dense<1> : tensor<4xi8>
  %output = "mhlo.select"(%result, %c1, %c0) : (tensor<4xi1>, tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  check.expect_eq_const(%output, dense<[0, 1, 1, 1]> : tensor<4xi8>) : tensor<4xi8>
  return
}
