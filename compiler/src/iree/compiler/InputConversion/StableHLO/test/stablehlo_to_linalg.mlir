// RUN: iree-opt %s --iree-stablehlo-to-linalg --split-input-file \
// RUN:   --canonicalize | FileCheck --enable-var-scope=false %s

// RUN: iree-opt %s --iree-stablehlo-to-linalg="enable-primitive-ops=true" \
// RUN:   --split-input-file --canonicalize | \
// RUN:   FileCheck %s --enable-var-scope=false --check-prefix=CHECK-PRIMITIVE

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @float_add
// CHECK-PRIMITIVE-LABEL: func @float_add
func.func @float_add(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: {someattr}
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: f32
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[RESULT:[a-zA-Z0-9_]*]] = arith.addf %[[ARG0]], %[[ARG1]]
  // CHECK: linalg.yield %[[RESULT]]

  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: arith.addf
  %0 = "stablehlo.add"(%lhs, %rhs) {someattr}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_add_dynamic_encoding
// CHECK-PRIMITIVE-LABEL: func @float_add_dynamic_encoding
func.func @float_add_dynamic_encoding(
  %lhs: tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>,
  %rhs: tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>)
    -> tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>> {
  // CHECK: linalg.generic
  // CHECK: arith.addf
  // CHECK: linalg.yield

  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: arith.addf
  %0 = "stablehlo.add"(%lhs, %rhs)
      : (tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>,
         tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>)
      -> tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>
  func.return %0 : tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>
}

// -----

// CHECK-LABEL: integer_add
// CHECK-PRIMITIVE-LABEL: integer_add
func.func @integer_add(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: addi
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: addi
  %0 = "stablehlo.add"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: complex_add
// CHECK-PRIMITIVE-LABEL: complex_add
func.func @complex_add(%lhs: tensor<2x2xcomplex<f32>>,
                  %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.add
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.add
  %0 = "stablehlo.add"(%lhs, %rhs) : (tensor<2x2xcomplex<f32>>,
      tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @complex_atan2
// CHECK-PRIMITIVE-LABEL: func @complex_atan2
func.func @complex_atan2(%lhs: tensor<2x2xcomplex<f32>>,
    %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "stablehlo.atan2"(%lhs, %rhs)
      : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)
      -> tensor<2x2xcomplex<f32>>
  // CHECK: linalg.generic
  // CHECK: complex.atan2
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.atan2
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}


// -----

// CHECK-LABEL: func @float_mul
// CHECK-PRIMITIVE-LABEL: func @float_mul
func.func @float_mul(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: mulf
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: mulf
  %0 = "stablehlo.multiply"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_mul
// CHECK-PRIMITIVE-LABEL: func @integer_mul
func.func @integer_mul(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: muli
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: muli
  %0 = "stablehlo.multiply"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @complex_mul
// CHECK-PRIMITIVE-LABEL: func @complex_mul
func.func @complex_mul(%lhs: tensor<2x2xcomplex<f32>>,
                  %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.mul
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.mul
  %0 = "stablehlo.multiply"(%lhs, %rhs)
          : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)
          -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_remainder
// CHECK-PRIMITIVE-LABEL: func @float_remainder
func.func @float_remainder(%lhs: tensor<2x2xf32>,
                      %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: remf
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: remf
  %0 = "stablehlo.remainder"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_remainder
// CHECK-PRIMITIVE-LABEL: func @integer_remainder
func.func @integer_remainder(%lhs: tensor<2x2xi32>,
                        %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: arith.remsi
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: arith.remsi
  %0 = "stablehlo.remainder"(%lhs, %rhs) : (tensor<2x2xi32>,
                                          tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @population_count_integer
// CHECK-PRIMITIVE-LABEL: func @population_count_integer
func.func @population_count_integer(%lhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: math.ctpop
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.ctpop
  %0 = "stablehlo.popcnt"(%lhs) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @complex_sqrt
// CHECK-PRIMITIVE-LABEL: func @complex_sqrt
func.func @complex_sqrt(%operand: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "stablehlo.sqrt"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  // CHECK: linalg.generic
  // CHECK: complex.sqrt
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.sqrt
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_rsqrt
// CHECK-PRIMITIVE-LABEL: func @float_rsqrt
func.func @float_rsqrt(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %tensor_result = "stablehlo.rsqrt"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: linalg.generic
  // CHECK: rsqrt
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: rsqrt
  func.return %tensor_result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_rsqrt
// CHECK-PRIMITIVE-LABEL: func @complex_rsqrt
func.func @complex_rsqrt(%operand: tensor<2x2xcomplex<f32>>)
    -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "stablehlo.rsqrt"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  // CHECK: linalg.generic
  // CHECK: complex.rsqrt
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.rsqrt
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_cbrt
// CHECK-PRIMITIVE-LABEL: func @float_cbrt
func.func @float_cbrt(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %tensor_result = "stablehlo.cbrt"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[IN:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[RESULT:.+]] = math.cbrt %[[IN]]
  // CHECK: linalg.yield %[[RESULT]]

  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.cbrt
  func.return %tensor_result : tensor<2x2xf32>
}

// -----


// CHECK-LABEL: func @float_sub
// CHECK-PRIMITIVE-LABEL: func @float_sub
func.func @float_sub(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: subf
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: subf
  %0 = "stablehlo.subtract"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_sub
// CHECK-PRIMITIVE-LABEL: func @integer_sub
func.func @integer_sub(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: subi
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: subi
  %0 = "stablehlo.subtract"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: complex_sub
// CHECK-PRIMITIVE-LABEL: complex_sub
func.func @complex_sub(%lhs: tensor<2x2xcomplex<f32>>,
                  %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.sub
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.sub
  %0 = "stablehlo.subtract"(%lhs, %rhs) : (tensor<2x2xcomplex<f32>>,
      tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_abs
// CHECK-PRIMITIVE-LABEL: func @float_abs
func.func @float_abs(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: {someattr}
  // CHECK: math.absf
  // CHECK-PRIMITIVE: linalg.map { math.absf }
  // CHECK-PRIMITIVE-SAME: ins(
  // CHECK-PRIMITIVE-SAME: outs(
  // CHECK-PRIMITIVE-SAME: {someattr}
  %0 = "stablehlo.abs"(%arg0) {someattr} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_exp
// CHECK-PRIMITIVE-LABEL: func @float_exp
func.func @float_exp(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: exp
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: exp
  %0 = "stablehlo.exponential"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_exp
func.func @complex_exp(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.exp
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.exp
  %0 = "stablehlo.exponential"(%arg0) : (tensor<2x2xcomplex<f32>>)
                                 -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_expm1
func.func @float_expm1(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: expm1
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: expm1
  %0 = "stablehlo.exponential_minus_one"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_expm1
func.func @complex_expm1(%arg0: tensor<2x2xcomplex<f32>>)
    -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.expm1
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.expm1
  %0 = "stablehlo.exponential_minus_one"(%arg0)
    : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_log
func.func @float_log(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.log
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.log
  %0 = "stablehlo.log"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_log
func.func @complex_log(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.log
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.log
  %0 = "stablehlo.log"(%arg0) : (tensor<2x2xcomplex<f32>>)
                         -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_log1p
// CHECK-PRIMITIVE-LABEL: func @float_log1p
func.func @float_log1p(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.log1p
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.log1p
  %0 = "stablehlo.log_plus_one"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_log1p
// CHECK-PRIMITIVE-LABEL: func @complex_log1p
func.func @complex_log1p(%arg0: tensor<2x2xcomplex<f32>>)
    -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.log1p
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.log1p
  %0 = "stablehlo.log_plus_one"(%arg0) : (tensor<2x2xcomplex<f32>>)
                                  -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_logistic
// CHECK-PRIMITIVE-LABEL: func @float_logistic
func.func @float_logistic(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[C1:.*]] = arith.constant 1.{{.*}}e+00
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG:.*]]: f32, %{{.*}}: f32):
  // CHECK: %[[NEG_ARG:.*]] = arith.negf %[[ARG]]
  // CHECK: %[[EXP_NEG_ARG:.*]] = math.exp %[[NEG_ARG]]
  // CHECK: %[[ONE_ADD_EXP_NEG_ARG:.*]] = arith.addf %[[EXP_NEG_ARG]], %[[C1]]
  // CHECK: %[[RESULT:.*]] = arith.divf %[[C1]], %[[ONE_ADD_EXP_NEG_ARG]]
  // CHECK: linalg.yield %[[RESULT]]
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: arith.negf
  // CHECK-PRIMITIVE: math.exp
  // CHECK-PRIMITIVE: arith.addf
  // CHECK-PRIMITIVE: arith.divf
  %0 = "stablehlo.logistic"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_logistic
func.func @complex_logistic(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG:.*]]: complex<f32>, %{{.*}}: complex<f32>):
  // CHECK: %[[NEG_ARG:.*]] = complex.neg %[[ARG]]
  // CHECK: %[[EXP_NEG_ARG:.*]] = complex.exp %[[NEG_ARG]]
  // CHECK: %[[CC1:.*]] = complex.create %[[C1]], %[[C0]] : complex<f32>
  // CHECK: %[[ONE_ADD_EXP_NEG_ARG:.*]] = complex.add %[[EXP_NEG_ARG]], %[[CC1]]
  // CHECK: %[[RESULT:.*]] = complex.div %[[CC1]], %[[ONE_ADD_EXP_NEG_ARG]]
  // CHECK: linalg.yield %[[RESULT]]
  %0 = "stablehlo.logistic"(%arg0) : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_ceil
// CHECK-PRIMITIVE-LABEL: func @float_ceil
func.func @float_ceil(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.ceil
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.ceil
  %0 = "stablehlo.ceil"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @floor
// CHECK-PRIMITIVE-LABEL: func @floor
func.func @floor(%input: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.floor
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.floor
  %0 = "stablehlo.floor"(%input) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_neg
// CHECK-PRIMITIVE-LABEL: func @float_neg
func.func @float_neg(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: negf
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: negf
  %0 = "stablehlo.negate"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_neg
// CHECK-PRIMITIVE-LABEL: func @complex_neg
func.func @complex_neg(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.neg
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.neg
  %0 = "stablehlo.negate"(%arg0) : (tensor<2x2xcomplex<f32>>)
                            -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @complex_sign
// CHECK-PRIMITIVE-LABEL: func @complex_sign
func.func @complex_sign(
    %arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.sign
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.sign
  %0 = "stablehlo.sign"(%arg0) : (tensor<2x2xcomplex<f32>>)
                          -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_tanh
// CHECK-PRIMITIVE-LABEL: func @float_tanh
func.func @float_tanh(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: tanh
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: tanh
  %0 = "stablehlo.tanh"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_tanh
// CHECK-PRIMITIVE-LABEL: func @complex_tanh
func.func @complex_tanh(%operand: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "stablehlo.tanh"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  // CHECK: linalg.generic
  // CHECK: complex.tanh
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.tanh
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @integer_and
// CHECK-PRIMITIVE-LABEL: func @integer_and
func.func @integer_and(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: and
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: and
  %0 = "stablehlo.and"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @integer_or
// CHECK-PRIMITIVE-LABEL: func @integer_or
func.func @integer_or(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: or
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: or
  %0 = "stablehlo.or"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @integer_xor
// CHECK-PRIMITIVE-LABEL: func @integer_xor
func.func @integer_xor(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: xor
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: xor
  %0 = "stablehlo.xor"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @count_leading_zeros
// CHECK-PRIMITIVE-LABEL: func @count_leading_zeros
func.func @count_leading_zeros(%lhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: math.ctlz
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.ctlz
  %0 = "stablehlo.count_leading_zeros"(%lhs) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: unsigned_convert
func.func @unsigned_convert(%in: tensor<2x2xui32>) -> tensor<2x2xui64> {
  // CHECK: linalg.generic
  // CHECK: arith.extui
  %0 = "stablehlo.convert"(%in) : (tensor<2x2xui32>) -> tensor<2x2xui64>
  func.return %0 : tensor<2x2xui64>
}

// -----

// CHECK-LABEL: func @float_cmp
// CHECK-PRIMITIVE-LABEL: func @float_cmp
func.func @float_cmp(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> (tensor<2x2xi1>) {
  %0 = "stablehlo.compare"(%lhs, %rhs) {comparison_direction = #stablehlo<comparison_direction EQ>}
          : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK: tensor.empty() : tensor<2x2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %{{.*}}: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpf oeq, %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.cmpf

// -----

// CHECK-LABEL: func @float_cmp_ne
// CHECK-PRIMITIVE-LABEL: func @float_cmp_ne
func.func @float_cmp_ne(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> (tensor<2x2xi1>) {
  %0 = "stablehlo.compare"(%lhs, %rhs) {comparison_direction = #stablehlo<comparison_direction NE>}
          : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK: tensor.empty() : tensor<2x2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %{{.*}}: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpf une, %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.cmpf

// -----

// CHECK-LABEL: func @float_cmp_totalorder
// CHECK-PRIMITIVE-LABEL: func @float_cmp_totalorder
func.func @float_cmp_totalorder(%lhs: tensor<2x2xbf16>,
                %rhs: tensor<2x2xbf16>) -> (tensor<2x2xi1>) {
  %0 = "stablehlo.compare"(%lhs, %rhs) {
    comparison_direction = #stablehlo<comparison_direction LT>,
    compare_type = #stablehlo<comparison_type TOTALORDER>
  } : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i16
// CHECK-DAG: %[[C32767:.*]] = arith.constant 32767 : i16
// CHECK: tensor.empty() : tensor<2x2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: bf16, %[[RHS_IN:.*]]: bf16, %{{.*}}: i1):
// CHECK-NEXT:   %[[LHS_INT:.*]] = arith.bitcast %[[LHS_IN]] : bf16 to i16
// CHECK-NEXT:   %[[LHS_CMP:.*]] = arith.cmpi slt, %[[LHS_INT]], %[[C0]] : i16
// CHECK-NEXT:   %[[LHS_SUB:.*]] = arith.subi %[[C32767]], %[[LHS_INT]] : i16
// CHECK-NEXT:   %[[LHS_SELECT:.*]] = arith.select %[[LHS_CMP]], %[[LHS_SUB]], %[[LHS_INT]] : i16
// CHECK-NEXT:   %[[RHS_INT:.*]] = arith.bitcast %[[RHS_IN]] : bf16 to i16
// CHECK-NEXT:   %[[RHS_CMP:.*]] = arith.cmpi slt, %[[RHS_INT]], %[[C0]] : i16
// CHECK-NEXT:   %[[RHS_SUB:.*]] = arith.subi %[[C32767]], %[[RHS_INT]] : i16
// CHECK-NEXT:   %[[RHS_SELECT:.*]] = arith.select %[[RHS_CMP]], %[[RHS_SUB]], %[[RHS_INT]] : i16
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpi slt, %[[LHS_SELECT]], %[[RHS_SELECT]] : i16
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// CHECK-PRIMITIVE-DAG: %[[C0:.*]] = arith.constant 0 : i16
// CHECK-PRIMITIVE-DAG: %[[C32767:.*]] = arith.constant 32767 : i16
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE-SAME: ins(
// CHECK-PRIMITIVE-SAME: outs(
// CHECK-PRIMITIVE-NEXT: (%[[LHS_IN:[a-zA-Z0-9]*]]: bf16, %[[RHS_IN:.*]]: bf16) {
// CHECK-PRIMITIVE-NEXT:   %[[LHS_INT:.*]] = arith.bitcast %[[LHS_IN]] : bf16 to i16
// CHECK-PRIMITIVE-NEXT:   %[[LHS_CMP:.*]] = arith.cmpi slt, %[[LHS_INT]], %[[C0]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[LHS_SUB:.*]] = arith.subi %[[C32767]], %[[LHS_INT]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[LHS_SELECT:.*]] = arith.select %[[LHS_CMP]], %[[LHS_SUB]], %[[LHS_INT]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[RHS_INT:.*]] = arith.bitcast %[[RHS_IN]] : bf16 to i16
// CHECK-PRIMITIVE-NEXT:   %[[RHS_CMP:.*]] = arith.cmpi slt, %[[RHS_INT]], %[[C0]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[RHS_SUB:.*]] = arith.subi %[[C32767]], %[[RHS_INT]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[RHS_SELECT:.*]] = arith.select %[[RHS_CMP]], %[[RHS_SUB]], %[[RHS_INT]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[RESULT:.*]] = arith.cmpi slt, %[[LHS_SELECT]], %[[RHS_SELECT]] : i16
// CHECK-PRIMITIVE-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @int_cmp
// CHECK-PRIMITIVE-LABEL: func @int_cmp
func.func @int_cmp(%lhs: tensor<2x2xi32>,
              %rhs: tensor<2x2xi32>) -> tensor<2x2xi1> {
  %0 = "stablehlo.compare"(%lhs, %rhs) {comparison_direction = #stablehlo<comparison_direction LT>}
          : (tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<2x2xi1>)
  func.return %0 : tensor<2x2xi1>
}
// CHECK: tensor.empty() : tensor<2x2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %{{.*}}: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpi slt, %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.cmpi

// -----

// CHECK-LABEL: func @complex_cmp_eq
// CHECK-PRIMITIVE-LABEL: func @complex_cmp_eq
func.func @complex_cmp_eq(%lhs: tensor<2xcomplex<f32>>,
                     %rhs: tensor<2xcomplex<f32>>) -> tensor<2xi1> {
  %0 = "stablehlo.compare"(%lhs, %rhs) {comparison_direction = #stablehlo<comparison_direction EQ>}
          : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xi1>)
  func.return %0 : tensor<2xi1>
}
// CHECK: tensor.empty() : tensor<2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: complex<f32>, %[[RHS_IN:.*]]: complex<f32>, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.eq %[[LHS_IN]], %[[RHS_IN]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: complex.eq

// -----

// CHECK-LABEL: func @complex_cmp_neq
// CHECK-PRIMITIVE-LABEL: func @complex_cmp_neq
func.func @complex_cmp_neq(%lhs: tensor<2xcomplex<f64>>,
                      %rhs: tensor<2xcomplex<f64>>) -> tensor<2xi1> {
  %0 = "stablehlo.compare"(%lhs, %rhs) {comparison_direction = #stablehlo<comparison_direction NE>}
          : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> (tensor<2xi1>)
  func.return %0 : tensor<2xi1>
}
// CHECK: tensor.empty() : tensor<2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: complex<f64>, %[[RHS_IN:.*]]: complex<f64>, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.neq %[[LHS_IN]], %[[RHS_IN]] : complex<f64>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: complex.neq

// -----

// CHECK-LABEL: func @float_cos
// CHECK-PRIMITIVE-LABEL: func @float_cos
func.func @float_cos(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.cos
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.cos
  %0 = "stablehlo.cosine"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_cos
// CHECK-PRIMITIVE-LABEL: func @complex_cos
func.func @complex_cos(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.cos
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.cos
  %0 = "stablehlo.cosine"(%arg0) : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_sin
// CHECK-PRIMITIVE-LABEL: func @float_sin
func.func @float_sin(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.sin
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.sin
  %0 = "stablehlo.sine"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_sin
// CHECK-PRIMITIVE-LABEL: func @complex_sin
func.func @complex_sin(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.sin
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.sin
  %0 = "stablehlo.sine"(%arg0) : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @is_finte
// CHECK-PRIMITIVE-LABEL: func @is_finte
func.func @is_finte(%input: tensor<2x2xf32>) -> tensor<2x2xi1> {
  %0 = "stablehlo.is_finite"(%input) : (tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK: %[[POS_INF:.+]] = arith.constant 0x7F800000 : f32
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32
// CHECK-NEXT:   %[[ABS_X:.+]] = math.absf %[[OPERAND_IN]] : f32
// CHECK-NEXT:   %[[RESULT:.+]] = arith.cmpf one, %[[ABS_X]], %[[POS_INF]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: math.absf
// CHECK-PRIMITIVE: arith.cmpf

// -----

// CHECK-LABEL: func @round_nearest_even
// CHECK-PRIMITIVE-LABEL: func @round_nearest_even
func.func @round_nearest_even(%val: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[IN:[a-zA-Z0-9_]*]]: f32
  // CHECK-SAME: %[[OUT:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[ROUND:.+]] = math.roundeven %[[IN]]
  // CHECK: linalg.yield %[[ROUND]]
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.roundeven
  %0 = "stablehlo.round_nearest_even"(%val) : (tensor<2x2xf32>) -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @round
// CHECK-PRIMITIVE-LABEL: func @round
func.func @round(%val: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[IN:[a-zA-Z0-9_]*]]: f32
  // CHECK-SAME: %[[OUT:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[ROUND:.+]] = math.round %[[IN]]
  // CHECK: linalg.yield %[[ROUND]]
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.round
  %0 = "stablehlo.round_nearest_afz"(%val) : (tensor<2x2xf32>) -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @select
func.func @select(%pred: tensor<2x2xi1>, %lhs: tensor<2x2xf32>,
             %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.select"(%pred, %lhs, %rhs)
         : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}
// CHECK: tensor.empty() : tensor<2x2xf32>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[PRED_IN:.*]]: i1, %[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.select %[[PRED_IN]], %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// CHECK-PRIMITIVE-LABEL: func @select
// CHECK-PRIMITIVE: tensor.empty() : tensor<2x2xf32>
// CHECK-PRIMITIVE: linalg.map { arith.select }
// CHECK-PRIMITIVE-SAME: ins(
// CHECK-PRIMITIVE-SAME: outs(

// -----

// CHECK-DAG:   #[[SCALAR_MAP:.*]] = affine_map<(d0, d1) -> ()>
// CHECK-DAG:   #[[ID_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @select_scalar_pred_dyn
// CHECK-SAME:  (%[[PRED:.*]]: tensor<i1>, %[[LHS:.*]]: tensor<2x?xf32>, %[[RHS:.*]]: tensor<2x?xf32>)
func.func @select_scalar_pred_dyn(%pred : tensor<i1>, %lhs: tensor<2x?xf32>, %rhs: tensor<2x?xf32>) -> tensor<2x?xf32> {
  %0 = "stablehlo.select"(%pred, %lhs, %rhs) {someattr} : (tensor<i1>, tensor<2x?xf32>, tensor<2x?xf32>) -> (tensor<2x?xf32>)
  func.return %0 : tensor<2x?xf32>
}
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1
// CHECK-DAG:  %[[DIM:.*]] = tensor.dim %[[LHS]], %[[C1]]
// CHECK-DAG:  %[[DST:.*]] = tensor.empty(%[[DIM]])
// CHECK:      linalg.generic
// CHECK-SAME:   indexing_maps = [#[[SCALAR_MAP]], #[[ID_MAP]], #[[ID_MAP]], #[[ID_MAP]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel"]
// CHECK-SAME:   ins(%[[PRED]], %[[LHS]], %[[RHS]] : tensor<i1>, tensor<2x?xf32>, tensor<2x?xf32>)
// CHECK-SAME:   outs(%[[DST]] : tensor<2x?xf32>)
// CHECK-SAME:   {someattr}
// CHECK:      ^bb0(%[[PRED_:.*]]: i1, %[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %{{.*}}: f32):
// CHECK:        %[[RES:.*]] = arith.select %[[PRED_]], %[[LHS_]], %[[RHS_]] : f32
// CHECK:        linalg.yield %[[RES]]

// CHECK-PRIMITIVE-LABEL: func @select_scalar_pred_dyn
// CHECK-PRIMITIVE-SAME:  (%[[PRED:.*]]: tensor<i1>, %[[LHS:.*]]: tensor<2x?xf32>, %[[RHS:.*]]: tensor<2x?xf32>)
// CHECK-PRIMITIVE-DAG:  %[[C1:.*]] = arith.constant 1
// CHECK-PRIMITIVE-DAG:  %[[DIM:.*]] = tensor.dim %[[LHS]], %[[C1]]
// CHECK-PRIMITIVE-DAG:  %[[DST:.*]] = tensor.empty(%[[DIM]])
// CHECK-PRIMITIVE-DAG:  %[[PRED_ELEM:.*]] = tensor.extract %[[PRED]]
// CHECK-PRIMITIVE:      linalg.map
// CHECK-PRIMITIVE-SAME:   ins(%[[LHS]], %[[RHS]] : tensor<2x?xf32>, tensor<2x?xf32>)
// CHECK-PRIMITIVE-SAME:   outs(%[[DST]] : tensor<2x?xf32>)
// CHECK-PRIMITIVE-SAME:   {someattr}
// CHECK-PRIMITIVE:      (%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32) {
// CHECK-PRIMITIVE:        %[[RES:.*]] = arith.select %[[PRED_ELEM]], %[[LHS_]], %[[RHS_]] : f32
// CHECK-PRIMITIVE:        linalg.yield %[[RES]]

// -----

// CHECK-LABEL: func @select_mixed
func.func @select_mixed(%pred: tensor<2x?xi1>, %lhs: tensor<?x2xf32>,
             %rhs: tensor<2x2xf32>) -> tensor<?x2xf32> {
  %0 = "stablehlo.select"(%pred, %lhs, %rhs)
         : (tensor<2x?xi1>, tensor<?x2xf32>, tensor<2x2xf32>) -> (tensor<?x2xf32>)
  func.return %0 : tensor<?x2xf32>
}

// -----

// CHECK-LABEL: func @bitcast_convert
func.func @bitcast_convert(%input: tensor<2x2xi32>) -> tensor<2x2xf32> {
  %result = "stablehlo.bitcast_convert"(%input) : (tensor<2x2xi32>) -> tensor<2x2xf32>
  func.return %result : tensor<2x2xf32>
}
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.bitcast %[[OPERAND_IN]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.bitcast

// -----

// CHECK-LABEL: func @bitcast_convert_dynamic
func.func @bitcast_convert_dynamic(%input: tensor<?x?xi32>) -> tensor<?x?xf32> {
  %result = "stablehlo.bitcast_convert"(%input) : (tensor<?x?xi32>) -> tensor<?x?xf32>
  func.return %result : tensor<?x?xf32>
}
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.bitcast %[[OPERAND_IN]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.bitcast

// -----

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @bitcast_convert_expand
func.func @bitcast_convert_expand(%input: tensor<6xi32>) -> tensor<6x4xi8> {
  %result = "stablehlo.bitcast_convert"(%input) : (tensor<6xi32>) -> tensor<6x4xi8>
  func.return %result : tensor<6x4xi8>
}

// CHECK: %[[C8:.*]] = arith.constant 8 : i32
// CHECK: tensor.empty() : tensor<6x4xi8>
// CHECK: %[[RESULT:.*]] = linalg.generic {
// CHECK:    indexing_maps = [#[[MAP0]], #[[MAP1]]],
// CHECK:    iterator_types = ["parallel", "parallel"]}
// CHECK:    ^bb0(%[[IN:.*]]: i32, %[[OUT:.*]]: i8):
// CHECK:      %[[IOTA:.*]] = linalg.index 1 : index
// CHECK:      %[[IOTA_CASTED:.*]] = arith.index_cast %[[IOTA]] : index to i32
// CHECK:      %[[AMT:.*]] = arith.muli %[[IOTA_CASTED]], %[[C8]] : i32
// CHECK:      %[[SHIFT:.*]] = arith.shrui %[[IN]], %[[AMT]] : i32
// CHECK:      %[[TRUNC:.*]] = arith.trunci %[[SHIFT]] : i32 to i8
// CHECK:      linalg.yield %[[TRUNC]] : i8
// CHECK:    } -> tensor<6x4xi8>
// CHECK:    return %[[RESULT]] : tensor<6x4xi8>

// -----

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func @bitcast_convert_contract
func.func @bitcast_convert_contract(%input: tensor<7x4xi8>) -> tensor<7xi32> {
  %result = "stablehlo.bitcast_convert"(%input) : (tensor<7x4xi8>) -> tensor<7xi32>
  func.return %result : tensor<7xi32>
}
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<7xi32>
// CHECK: linalg.fill ins(%[[C0]] : i32) outs(%[[EMPTY]] : tensor<7xi32>) -> tensor<7xi32>
// CHECK: %[[RESULT:.*]] = linalg.generic {
// CHECK:    indexing_maps = [#[[MAP0]], #[[MAP1]]],
// CHECK:    iterator_types = ["parallel", "reduction"]}
// CHECK:    ^bb0(%[[IN:.*]]: i8, %[[OUT:.*]]: i32):
// CHECK:      %[[IOTA:.*]] = linalg.index 1 : index
// CHECK:      %[[IOTA_CASTED:.*]] = arith.index_cast %[[IOTA]] : index to i32
// CHECK:      %[[AMT:.*]] = arith.muli %[[IOTA_CASTED]], %[[C8]] : i3
// CHECK:      %[[EXT:.*]] = arith.extui %[[IN]] : i8 to i32
// CHECK:      %[[SHIFT:.*]] = arith.shli %[[EXT]], %[[AMT]] : i32
// CHECK:      %[[OR:.*]] = arith.ori %[[SHIFT]], %[[OUT]] : i32
// CHECK:      linalg.yield %[[OR]] : i32
// CHECK: } -> tensor<7xi32>
// CHECK: return %[[RESULT]] : tensor<7xi32>

// -----

// CHECK-LABEL:   func @concatenate(
// CHECK-SAME:   %[[VAL_0:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[VAL_1:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[VAL_2:[a-zA-Z0-9_]*]]
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[C0]] : tensor<?x?xi32>
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_0]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_9:.*]] = tensor.dim %[[VAL_1]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_15:.*]] = tensor.dim %[[VAL_2]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_16:.*]] = arith.addi %[[VAL_7]], %[[VAL_9]] : index
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_15]] : index
// CHECK:           %[[VAL_23:.*]] = tensor.empty(%[[VAL_5]], %[[VAL_17]]) : tensor<?x?xi32>
// CHECK:           %[[VAL_24:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%[[VAL_23]] : tensor<?x?xi32>) {
// CHECK:           ^bb0(%[[VAL_25:.*]]: i32):
// CHECK:             %[[VAL_26:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_28:.*]] = linalg.index 1 : index
// CHECK:             %[[VAL_30:.*]] = tensor.dim %[[VAL_0]], %[[C1]] : tensor<?x?xi32>
// CHECK:             %[[VAL_32:.*]] = arith.cmpi ult, %[[VAL_28]], %[[VAL_30]] : index
// CHECK:             %[[VAL_33:.*]] = scf.if %[[VAL_32]] -> (i32) {
// CHECK:               %[[VAL_35:.*]] = tensor.extract %[[VAL_0]][%[[VAL_26]], %[[VAL_28]]] : tensor<?x?xi32>
// CHECK:               scf.yield %[[VAL_35]] : i32
// CHECK:             } else {
// CHECK:               %[[VAL_37:.*]] = tensor.dim %[[VAL_1]], %[[C1]] : tensor<?x?xi32>
// CHECK:               %[[VAL_38:.*]] = arith.addi %[[VAL_30]], %[[VAL_37]] : index
// CHECK:               %[[VAL_39:.*]] = arith.cmpi ult, %[[VAL_28]], %[[VAL_38]] : index
// CHECK:               %[[VAL_40:.*]] = scf.if %[[VAL_39]] -> (i32) {
// CHECK:                 %[[VAL_41:.*]] = arith.subi %[[VAL_28]], %[[VAL_30]] : index
// CHECK:                 %[[VAL_42:.*]] = tensor.extract %[[VAL_1]][%[[VAL_26]], %[[VAL_41]]] : tensor<?x?xi32>
// CHECK:                 scf.yield %[[VAL_42]] : i32
// CHECK:               } else {
// CHECK:                 %[[VAL_43:.*]] = arith.subi %[[VAL_28]], %[[VAL_38]] : index
// CHECK:                 %[[VAL_44:.*]] = tensor.extract %[[VAL_2]][%[[VAL_26]], %[[VAL_43]]] : tensor<?x?xi32>
// CHECK:                 scf.yield %[[VAL_44]] : i32
// CHECK:               }
// CHECK:               scf.yield %[[VAL_45:.*]] : i32
// CHECK:             }
// CHECK:             linalg.yield %[[VAL_46:.*]] : i32
// CHECK:           } -> tensor<?x?xi32>
// CHECK:           return %[[VAL_47:.*]] : tensor<?x?xi32>
// CHECK:         }
func.func @concatenate(%a: tensor<?x?xi32>, %b: tensor<?x?xi32>, %c: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %concat = "stablehlo.concatenate"(%a, %b, %c) {
      dimension = 1
    } : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
    func.return %concat : tensor<?x?xi32>
}

// -----

// CHECK-LABEL:   func @concatenate_unsigned(
// CHECK-SAME:   %[[VAL_0:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[VAL_1:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[VAL_2:[a-zA-Z0-9_]*]]
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : tensor<?x?xui32> to tensor<?x?xi32>
// CHECK-DAG:       %[[VAL_4:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : tensor<?x?xui32> to tensor<?x?xi32>
// CHECK-DAG:       %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : tensor<?x?xui32> to tensor<?x?xi32>
// CHECK:           %[[VAL_8:.*]] = tensor.dim %[[VAL_3]], %[[C0]] : tensor<?x?xi32>
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_3]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_14:.*]] = tensor.dim %[[VAL_4]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_18:.*]] = tensor.dim %[[VAL_5]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_19:.*]] = arith.addi %[[VAL_10]], %[[VAL_14]] : index
// CHECK:           %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_18]] : index
// CHECK:           %[[VAL_26:.*]] = tensor.empty(%[[VAL_8]], %[[VAL_20]]) : tensor<?x?xi32>
// CHECK:           %[[VAL_27:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%[[VAL_26]] : tensor<?x?xi32>) {
// CHECK:           ^bb0(%[[VAL_28:.*]]: i32):
// CHECK:             %[[VAL_29:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_30:.*]] = linalg.index 1 : index
// CHECK:             %[[VAL_33:.*]] = tensor.dim %[[VAL_3]], %[[C1]] : tensor<?x?xi32>
// CHECK:             %[[VAL_35:.*]] = arith.cmpi ult, %[[VAL_30]], %[[VAL_33]] : index
// CHECK:             %[[VAL_36:.*]] = scf.if %[[VAL_35]] -> (i32) {
// CHECK:               %[[VAL_38:.*]] = tensor.extract %[[VAL_3]][%[[VAL_29]], %[[VAL_30]]] : tensor<?x?xi32>
// CHECK:               scf.yield %[[VAL_38]] : i32
// CHECK:             } else {
// CHECK:               %[[VAL_40:.*]] = tensor.dim %[[VAL_4]], %[[C1]] : tensor<?x?xi32>
// CHECK:               %[[VAL_41:.*]] = arith.addi %[[VAL_33]], %[[VAL_40]] : index
// CHECK:               %[[VAL_42:.*]] = arith.cmpi ult, %[[VAL_30]], %[[VAL_41]] : index
// CHECK:               %[[VAL_43:.*]] = scf.if %[[VAL_42]] -> (i32) {
// CHECK:                 %[[VAL_44:.*]] = arith.subi %[[VAL_30]], %[[VAL_33]] : index
// CHECK:                 %[[VAL_45:.*]] = tensor.extract %[[VAL_4]][%[[VAL_29]], %[[VAL_44]]] : tensor<?x?xi32>
// CHECK:                 scf.yield %[[VAL_45]] : i32
// CHECK:               } else {
// CHECK:                 %[[VAL_46:.*]] = arith.subi %[[VAL_30]], %[[VAL_41]] : index
// CHECK:                 %[[VAL_47:.*]] = tensor.extract %[[VAL_5]][%[[VAL_29]], %[[VAL_46]]] : tensor<?x?xi32>
// CHECK:                 scf.yield %[[VAL_47]] : i32
// CHECK:               }
// CHECK:               scf.yield %[[VAL_48:.*]] : i32
// CHECK:             }
// CHECK:             linalg.yield %[[VAL_49:.*]] : i32
// CHECK:           } -> tensor<?x?xi32>
// CHECK:           %[[VAL_50:.*]] = builtin.unrealized_conversion_cast %[[VAL_51:.*]] : tensor<?x?xi32> to tensor<?x?xui32>
// CHECK:           return %[[VAL_50]] : tensor<?x?xui32>
// CHECK:         }
func.func @concatenate_unsigned(%a: tensor<?x?xui32>, %b: tensor<?x?xui32>, %c: tensor<?x?xui32>) -> tensor<?x?xui32> {
    %concat = "stablehlo.concatenate"(%a, %b, %c) {
      dimension = 1
    } : (tensor<?x?xui32>, tensor<?x?xui32>, tensor<?x?xui32>) -> tensor<?x?xui32>
    func.return %concat : tensor<?x?xui32>
}

// -----

// CHECK-LABEL: signed_divide
func.func @signed_divide(%lhs: tensor<2x2xi32>, %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK-DAG:   %[[VAL_7:.*]] = arith.constant -1 : i32
  // CHECK-DAG:   %[[VAL_8:.*]] = arith.constant -2147483648 : i32
  // CHECK-DAG:   %[[VAL_9:.*]] = arith.constant 0 : i32
  // CHECK-DAG:   %[[VAL_10:.*]] = arith.constant 1 : i32
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32):
  // CHECK:   %[[VAL_11:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_9]] : i32
  // CHECK:   %[[VAL_13:.*]] = arith.cmpi eq, %[[VAL_4]], %[[VAL_8]] : i32
  // CHECK:   %[[VAL_15:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_7]] : i32
  // CHECK:   %[[VAL_16:.*]] = arith.andi %[[VAL_13]], %[[VAL_15]] : i1
  // CHECK:   %[[VAL_17:.*]] = arith.ori %[[VAL_11]], %[[VAL_16]] : i1
  // CHECK:   %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_10]], %[[VAL_5]] : i32
  // CHECK:   %[[VAL_19:.*]] = arith.divsi %[[VAL_4]], %[[VAL_18]] : i32
  // CHECK:   %[[VAL_20:.*]] = arith.select %[[VAL_16]], %[[VAL_8]], %[[VAL_19]] : i32
  // CHECK:   %[[VAL_21:.*]] = arith.select %[[VAL_11]], %[[VAL_7]], %[[VAL_20]] : i32
  // CHECK:   linalg.yield %[[VAL_21]] : i32
  %0 = "stablehlo.divide"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: unsigned_divide
func.func @unsigned_divide(%lhs: tensor<2x2xui32>, %rhs: tensor<2x2xui32>) -> tensor<2x2xui32> {
  // CHECK-DAG:   %[[VAL_9:.*]] = arith.constant -1 : i32
  // CHECK-DAG:   %[[VAL_11:.*]] = arith.constant 0 : i32
  // CHECK-DAG:   %[[VAL_12:.*]] = arith.constant 1 : i32
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32):
  // CHECK:   %[[VAL_13:.*]] = arith.cmpi eq, %[[VAL_7]], %[[VAL_11]] : i32
  // CHECK:   %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_12]], %[[VAL_7]] : i32
  // CHECK:   %[[VAL_15:.*]] = arith.divui %[[VAL_6]], %[[VAL_14]] : i32
  // CHECK:   %[[VAL_16:.*]] = arith.select %[[VAL_13]], %[[VAL_9]], %[[VAL_15]] : i32
  // CHECK:   linalg.yield %[[VAL_16]] : i32
  %0 = "stablehlo.divide"(%lhs, %rhs) : (tensor<2x2xui32>, tensor<2x2xui32>) -> tensor<2x2xui32>
  func.return %0 : tensor<2x2xui32>
}

// -----

// CHECK-LABEL: complex_divide
func.func @complex_divide(%lhs: tensor<2xcomplex<f32>>,
                     %rhs: tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.div
  %0 = "stablehlo.divide"(%lhs, %rhs) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  func.return %0 : tensor<2xcomplex<f32>>
}

// -----

func.func @shift_left(%lhs: tensor<2x2xi32>,
                 %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %result = "stablehlo.shift_left"(%lhs, %rhs)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %result : tensor<2x2xi32>
}
// CHECK-LABEL: func @shift_left
// CHECK-DAG:    %[[ZERO:.*]] = arith.constant 0
// CHECK-DAG:    %[[BITS:.*]] = arith.constant 32
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS:.*]]: i32, %[[RHS:.*]]: i32, %{{.*}}: i32):
// CHECK-DAG:    %[[SHIFT:.*]] = arith.shli %[[LHS]], %[[RHS]] : i32
// CHECK-DAG:    %[[NOT_SATURATING:.*]] = arith.cmpi ult, %[[RHS]], %[[BITS]]
// CHECK-NEXT:   %[[RESULT:.*]] = arith.select %[[NOT_SATURATING]], %[[SHIFT]], %[[ZERO]]
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

func.func @shift_right_arithmetic(%lhs: tensor<2x2xi32>,
                             %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %result = "stablehlo.shift_right_arithmetic"(%lhs, %rhs)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %result : tensor<2x2xi32>
}
// CHECK-LABEL: func @shift_right_arithmetic
// CHECK-DAG:    %[[BITS:.*]] = arith.constant 32
// CHECK-DAG:    %[[MAX_SHIFT:.*]] = arith.constant 31
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS:.*]]: i32, %[[RHS:.*]]: i32, %{{.*}}: i32):
// CHECK-DAG:    %[[SHIFT:.*]] = arith.shrsi %[[LHS]], %[[RHS]] : i32
// CHECK-DAG:    %[[MAX_SHIFTED:.*]] = arith.shrsi %[[LHS]], %[[MAX_SHIFT]] : i32
// CHECK-DAG:    %[[NOT_SATURATING:.*]] = arith.cmpi ult, %[[RHS]], %[[BITS]]
// CHECK-NEXT:   %[[RESULT:.*]] = arith.select %[[NOT_SATURATING]], %[[SHIFT]], %[[MAX_SHIFTED]]
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

func.func @shift_right_logical(%lhs: tensor<2x2xi32>,
                          %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %result = "stablehlo.shift_right_logical"(%lhs, %rhs)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %result : tensor<2x2xi32>
}
// CHECK-LABEL: func @shift_right_logical
// CHECK-DAG:    %[[ZERO:.*]] = arith.constant 0
// CHECK-DAG:    %[[BITS:.*]] = arith.constant 32
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS:.*]]: i32, %[[RHS:.*]]: i32, %{{.*}}: i32):
// CHECK-DAG:    %[[SHIFT:.*]] = arith.shrui %[[LHS]], %[[RHS]] : i32
// CHECK-DAG:    %[[NOT_SATURATING:.*]] = arith.cmpi ult, %[[RHS]], %[[BITS]]
// CHECK-NEXT:   %[[RESULT:.*]] = arith.select %[[NOT_SATURATING]], %[[SHIFT]], %[[ZERO]]
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-LABEL: func @einsum_basic
func.func @einsum_basic(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "ijk,ikm->ijm", someattr}: (tensor<3x4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
  func.return %0 : tensor<3x4x6xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<3x4x5xf32>, %[[RHS:.*]]: tensor<3x5x6xf32>)
// CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<3x4x6xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<3x4x5xf32>, tensor<3x5x6xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<3x4x6xf32>)
// CHECK-SAME: {someattr}
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   %[[RES:.*]] = arith.addf %[[OUT_]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func @einsum_pointwisemul
func.func @einsum_pointwisemul(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "abc,abc->abc"} : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
  func.return %0 : tensor<3x4x5xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<3x4x5xf32>, %[[RHS:.*]]: tensor<3x4x5xf32>)
// CHECK: tensor.empty() : tensor<3x4x5xf32>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP0]], #[[MAP0]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<3x4x5xf32>, tensor<3x4x5xf32>)
// CHECK-SAME: outs(%[[DST:.+]] : tensor<3x4x5xf32>)
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[RES:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: func @einsum_matmul
func.func @einsum_matmul(%arg0: tensor<7x9xf32>, %arg1: tensor<9x5xf32>) -> tensor<7x5xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "ae,ed->ad"}: (tensor<7x9xf32>, tensor<9x5xf32>) -> tensor<7x5xf32>
  func.return %0 : tensor<7x5xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<7x9xf32>, %[[RHS:.*]]: tensor<9x5xf32>)
// CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<7x5xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<7x9xf32>, tensor<9x5xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<7x5xf32>)
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   %[[RES:.*]] = arith.addf %[[OUT_]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5)>
// CHECK: func @einsum_broadcast4
func.func @einsum_broadcast4(%arg0: tensor<3x4x5x6x7xf32>, %arg1: tensor<7x8xf32>) -> tensor<3x4x5x6x8xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "abcdh,hg->abcdg"}: (tensor<3x4x5x6x7xf32>, tensor<7x8xf32>) -> tensor<3x4x5x6x8xf32>
  func.return %0 : tensor<3x4x5x6x8xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<3x4x5x6x7xf32>, %[[RHS:.*]]: tensor<7x8xf32>)
// CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<3x4x5x6x8xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<3x4x5x6x7xf32>, tensor<7x8xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<3x4x5x6x8xf32>)
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   %[[RES:.*]] = arith.addf %[[OUT_]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: func @einsum_ellipsis
func.func @einsum_ellipsis(%arg0: tensor<1x512x128xf32>, %arg1: tensor<128x256xf32>) -> tensor<1x512x256xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "...x,xy->...y"} : (tensor<1x512x128xf32>, tensor<128x256xf32>) -> tensor<1x512x256xf32>
  func.return %0 : tensor<1x512x256xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<1x512x128xf32>, %[[RHS:.*]]: tensor<128x256xf32>)
// CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<1x512x256xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<1x512x128xf32>, tensor<128x256xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<1x512x256xf32>)
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   %[[RES:.*]] = arith.addf %[[OUT_]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: func @einsum_dynamic_size_broadcast_dot
func.func @einsum_dynamic_size_broadcast_dot(%arg0: tensor<?x?x4xf32>, %arg1: tensor<4x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "abc,cd->abd"} : (tensor<?x?x4xf32>, tensor<4x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<?x?x4xf32>, %[[RHS:.*]]: tensor<4x?xf32>)
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[DIM0:.+]] = tensor.dim %[[LHS]], %[[C0]] : tensor<?x?x4xf32>
// CHECK: %[[DIM1:.+]] = tensor.dim %[[LHS]], %[[C1]] : tensor<?x?x4xf32>
// CHECK: %[[DIM2:.+]] = tensor.dim %[[RHS]], %[[C1:.+]] : tensor<4x?xf32>
// CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]], %[[DIM2]]) : tensor<?x?x?xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<?x?x4xf32>, tensor<4x?xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<?x?x?xf32>)
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   %[[RES:.*]] = arith.addf %[[OUT_]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, 0)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func @broadcast_in_dim
func.func @broadcast_in_dim(%operand: tensor<5x7x1xf32>) -> tensor<7x10x6x4x5xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%operand)
         {broadcast_dimensions = dense<[4,0,2]> : tensor<3xi64>}
         : (tensor<5x7x1xf32>) -> tensor<7x10x6x4x5xf32>
  func.return %0 : tensor<7x10x6x4x5xf32>
}
// CHECK: tensor.empty() : tensor<7x10x6x4x5xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast_in_dim
// CHECK-PRIMITIVE: tensor.collapse_shape
// CHECK-PRIMITIVE: linalg.transpose
// CHECK-PRIMITIVE:   permutation = [1, 0]
// CHECK-PRIMITIVE: tensor.empty() : tensor<7x10x6x4x5xf32>
// CHECK-PRIMITIVE: linalg.broadcast
// CHECK-PRIMITIVE:   dimensions = [1, 2, 3]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, 0)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func @broadcast_in_dim_ui32
func.func @broadcast_in_dim_ui32(%operand: tensor<5x7x1xui32>) -> tensor<7x10x6x4x5xui32> {
  %0 = "stablehlo.broadcast_in_dim"(%operand)
         {broadcast_dimensions = dense<[4,0,2]> : tensor<3xi64>}
         : (tensor<5x7x1xui32>) -> tensor<7x10x6x4x5xui32>
  func.return %0 : tensor<7x10x6x4x5xui32>
}
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<5x7x1xui32> to tensor<5x7x1xi32>
// CHECK: tensor.empty() : tensor<7x10x6x4x5xi32>
// CHECK: %[[RES:.*]] = linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : i32
// CHECK: builtin.unrealized_conversion_cast %[[RES]] : tensor<7x10x6x4x5xi32> to tensor<7x10x6x4x5xui32>

// CHECK-PRIMITIVE-LABEL: func @broadcast_in_dim_ui32
// CHECK-PRIMITIVE: tensor.collapse_shape
// CHECK-PRIMITIVE: linalg.transpose
// CHECK-PRIMITIVE:   permutation = [1, 0]
// CHECK-PRIMITIVE: tensor.empty() : tensor<7x10x6x4x5xi32>
// CHECK-PRIMITIVE: %[[RES:.*]] = linalg.broadcast
// CHECK-PRIMITIVE:   dimensions = [1, 2, 3]
// CHECK-PRIMITIVE: builtin.unrealized_conversion_cast %[[RES]] : tensor<7x10x6x4x5xi32> to tensor<7x10x6x4x5xui32>

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @broadcast_in_dim_with_one_to_one
func.func @broadcast_in_dim_with_one_to_one(
         %operand: tensor<1xf32>) -> tensor<1x5xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%operand)
         {broadcast_dimensions = dense<[0]> : tensor<1xi64>}
         : (tensor<1xf32>) -> tensor<1x5xf32>
  func.return %0 : tensor<1x5xf32>
}
// CHECK: tensor.empty() : tensor<1x5xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast_in_dim_with_one_to_one
// CHECK-PRIMITIVE-NOT: tensor.collapse_shape
// CHECK-PRIMITIVE-NOT: linalg.transpose
// CHECK-PRIMITIVE:     linalg.broadcast
// CHECK-PRIMITIVE:       dimensions = [1]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d0, d1)>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @broadcast_in_dim_with_transpose
func.func @broadcast_in_dim_with_transpose(
         %operand: tensor<2x3x4xf32>) -> tensor<3x4x2x5xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%operand)
         {broadcast_dimensions = dense<[2, 0, 1]> : tensor<3xi64>}
         : (tensor<2x3x4xf32>) -> tensor<3x4x2x5xf32>
  func.return %0 : tensor<3x4x2x5xf32>
}
// CHECK: tensor.empty() : tensor<3x4x2x5xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast_in_dim_with_transpose
// CHECK-PRIMITIVE: tensor.empty() : tensor<3x4x2xf32>
// CHECK-PRIMITIVE: linalg.transpose
// CHECK-PRIMITIVE:   permutation = [1, 2, 0]
// CHECK-PRIMITIVE: tensor.empty() : tensor<3x4x2x5xf32>
// CHECK-PRIMITIVE: linalg.broadcast
// CHECK-PRIMITIVE:   dimensions = [3]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2) -> ()>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @broadcast_in_dim_scalar
func.func @broadcast_in_dim_scalar(%operand: tensor<f32>) -> tensor<7x10x6xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%operand)
        {broadcast_dimensions = dense<[]> : tensor<0xi64>}
        : (tensor<f32>) -> tensor<7x10x6xf32>
  func.return %0 : tensor<7x10x6xf32>
}
// CHECK: tensor.empty() : tensor<7x10x6xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast_in_dim_scalar
// CHECK-PRIMITIVE: tensor.empty() : tensor<7x10x6xf32>
// CHECK-PRIMITIVE: linalg.broadcast
// CHECK-PRIMITIVE:   dimensions = [0, 1, 2]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1, d2) -> ()>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @broadcast_scalar
func.func @broadcast_scalar(%arg: tensor<f32>) -> tensor<4x2x1xf32> {
  %0 = "stablehlo.broadcast"(%arg) {broadcast_sizes = dense<[4, 2, 1]> : tensor<3xi64>} : (tensor<f32>) -> tensor<4x2x1xf32>
  func.return %0: tensor<4x2x1xf32>
}
// CHECK: tensor.empty() : tensor<4x2x1xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast_scalar
// CHECK-PRIMITIVE: tensor.empty() : tensor<4x2x1xf32>
// CHECK-PRIMITIVE: linalg.broadcast
// CHECK-PRIMITIVE-SAME: ins(
// CHECK-PRIMITIVE-SAME: outs(
// CHECK-PRIMITIVE-SAME: dimensions = [0, 1, 2]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-LABEL: func @broadcast
func.func @broadcast(%arg: tensor<4x?x16xf32>) -> tensor<4x2x1x4x?x16xf32> {
  %0 = "stablehlo.broadcast"(%arg) {broadcast_sizes = dense<[4, 2, 1]> : tensor<3xi64>} : (tensor<4x?x16xf32>) -> tensor<4x2x1x4x?x16xf32>
  func.return %0: tensor<4x2x1x4x?x16xf32>
}
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[DIM:.*]] = tensor.dim %{{.*}}, %[[C1]] : tensor<4x?x16xf32>
// CHECK: %{{.*}} = tensor.empty(%[[DIM]]) : tensor<4x2x1x4x?x16xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast
// CHECK-PRIMITIVE-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-PRIMITIVE: %[[DIM:.*]] = tensor.dim %{{.*}}, %[[C1]] : tensor<4x?x16xf32>
// CHECK-PRIMITIVE: %{{.*}} = tensor.empty(%[[DIM]]) : tensor<4x2x1x4x?x16xf32>
// CHECK-PRIMITIVE: linalg.broadcast
// CHECK-PRIMITIVE:   dimensions = [0, 1, 2]

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @iota_f32
func.func @iota_f32() -> tensor<7x10xf32> {
  %result = "stablehlo.iota"() {iota_dimension = 1 : i64, someattr} : () -> (tensor<7x10xf32>)
  func.return %result : tensor<7x10xf32>
}
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-SAME: {someattr}
// CHECK-NEXT: ^bb0(%{{.*}}: f32):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   %[[FLOAT_CAST:.*]] = arith.sitofp %[[INT_CAST]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[FLOAT_CAST]] : f32

// CHECK-PRIMITIVE-LABEL: func @iota_f32
// CHECK-PRIMITIVE: %[[EMPTY:.*]] = tensor.empty()
// CHECK-PRIMITIVE: linalg.map outs(%[[EMPTY]] : tensor<7x10xf32>
// CHECK-PRIMITIVE-SAME: {someattr}
// CHECK-PRIMITIVE:        %[[INDEX:.*]] = linalg.index 1
// CHECK-PRIMITIVE-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i64
// CHECK-PRIMITIVE-NEXT:   %[[FLOAT_CAST:.*]] = arith.sitofp %[[INT_CAST]] : i64 to f32
// CHECK-PRIMITIVE-NEXT:   linalg.yield %[[FLOAT_CAST]]

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @iota_i32
func.func @iota_i32() -> tensor<7x10xi32> {
  %result = "stablehlo.iota"() {iota_dimension = 1 : i64} : () -> (tensor<7x10xi32>)
  func.return %result : tensor<7x10xi32>
}
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   linalg.yield %[[INT_CAST]] : i32

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @iota_ui32
func.func @iota_ui32() -> tensor<7x10xui32> {
  %result = "stablehlo.iota"() {iota_dimension = 1 : i64} : () -> (tensor<7x10xui32>)
  func.return %result : tensor<7x10xui32>
}
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   linalg.yield %[[INT_CAST]] : i32
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<7x10xi32> to tensor<7x10xui32>

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @iota_complexf32
func.func @iota_complexf32() -> tensor<7x10xcomplex<f32>> {
  %result = "stablehlo.iota"() {iota_dimension = 1 : i64} : () -> (tensor<7x10xcomplex<f32>>)
  func.return %result : tensor<7x10xcomplex<f32>>
}
// CHECK-DAG:    %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%{{.*}}: complex<f32>):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   %[[FLOAT_CAST:.*]] = arith.sitofp %[[INT_CAST]] : i32 to f32
// CHECK-NEXT:   %[[COMPLEX_CAST:.*]] = complex.create %[[FLOAT_CAST]], %[[ZERO]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[COMPLEX_CAST]] : complex<f32>

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @dynamic_iota_f32
// CHECK-SAME: %[[SHAPE:.*]]: tensor<?xi32>
func.func @dynamic_iota_f32(%shape: tensor<?xi32>) -> tensor<?x?x8xf32> {
  %result = "stablehlo.dynamic_iota"(%shape) {iota_dimension = 1 : i64} : (tensor<?xi32>) -> (tensor<?x?x8xf32>)
  func.return %result : tensor<?x?x8xf32>
}
// CHECK: %[[V1:.*]] = tensor.extract %[[SHAPE]][%c0]
// CHECK: %[[I1:.*]] = arith.index_cast %[[V1]] : i32 to index
// CHECK: %[[V2:.*]] = tensor.extract %[[SHAPE]][%c1]
// CHECK: %[[I2:.*]] = arith.index_cast %[[V2]] : i32 to index
// CHECK: tensor.empty(%[[I1]], %[[I2]]) : tensor<?x?x8xf32>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%{{.*}}: f32):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   %[[FLOAT_CAST:.*]] = arith.sitofp %[[INT_CAST]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[FLOAT_CAST]] : f32

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @dyanmic_iota_ui32
// CHECK-SAME: %[[SHAPE:.*]]: tensor<?xi32>
func.func @dyanmic_iota_ui32(%shape: tensor<?xi32>) -> tensor<?x?x8xui32> {
  %result = "stablehlo.dynamic_iota"(%shape) {iota_dimension = 1 : i64} : (tensor<?xi32>) -> (tensor<?x?x8xui32>)
  func.return %result : tensor<?x?x8xui32>
}
// CHECK: %[[V1:.*]] = tensor.extract %[[SHAPE]][%c0]
// CHECK: %[[I1:.*]] = arith.index_cast %[[V1]] : i32 to index
// CHECK: %[[V2:.*]] = tensor.extract %[[SHAPE]][%c1]
// CHECK: %[[I2:.*]] = arith.index_cast %[[V2]] : i32 to index
// CHECK: tensor.empty(%[[I1]], %[[I2]]) : tensor<?x?x8xi32>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   linalg.yield %[[FLOAT_CAST]] : i32
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<?x?x8xi32> to tensor<?x?x8xui32>

// -----

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @float_pow
func.func @float_pow(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: f32
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[RESULT:[a-zA-Z0-9_]*]] = math.powf %[[ARG0]], %[[ARG1]]
  // CHECK: linalg.yield %[[RESULT]]
  %0 = "stablehlo.power"(%lhs, %rhs) : (tensor<2x2xf32>,
                                   tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_pow
func.func @complex_pow(%lhs: tensor<2x2xcomplex<f32>>,
                %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: complex<f32>
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: complex<f32>
  // CHECK: %[[RESULT:[a-zA-Z0-9_]*]] = complex.pow %[[ARG0]], %[[ARG1]]
  // CHECK: linalg.yield %[[RESULT]]
  %0 = "stablehlo.power"(%lhs, %rhs) : (tensor<2x2xcomplex<f32>>,
                                   tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @integer_pow
func.func @integer_pow(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
                    // CHECK: linalg.generic
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: i32
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: i32
  // CHECK: %[[FOR_RESULT:[a-zA-Z0-9_]*]]:3 = scf.for {{.*}} to %c6 step %c1
  // CHECK-SAME: iter_args(
  // CHECK-SAME:   %[[ITER0:.*]] = %c1
  // CHECK-SAME:   %[[ITER1:.*]] = %[[ARG0]],
  // CHECK-SAME:   %[[ITER2:.*]] = %[[ARG1]]
  // CHECK-SAME: ) -> (i32, i32, i32) {
  //   CHECK: %[[AND:[a-zA-Z0-9_]*]] = arith.andi %[[ITER2]], %c1
  //   CHECK: %[[COND:[a-zA-Z0-9_]*]] = arith.cmpi eq, %[[AND]], %c1
  //   CHECK: %[[MUL:[a-zA-Z0-9_]*]] = arith.muli %[[ITER0]], %[[ITER1]]
  //   CHECK: %[[ACCUM:[a-zA-Z0-9_]*]] = arith.select %[[COND]], %[[MUL]], %[[ITER0]]
  //   CHECK: %[[BASE:[a-zA-Z0-9_]*]] = arith.muli %[[ITER1]], %[[ITER1]]
  //   CHECK: %[[EXP:[a-zA-Z0-9_]*]] = arith.shrui %[[ITER2]], %c1
  //   CHECK: scf.yield %[[ACCUM]], %[[BASE]], %[[EXP]]
  // CHECK: %[[RHS_PARITY:.*]] = arith.remsi %[[ARG1]], %c2
  // CHECK: %[[RHS_EVEN:.*]] = arith.cmpi eq, %[[RHS_PARITY]], %c0
  // CHECK: %[[RHS_NEG:.*]] = arith.cmpi slt, %[[ARG1]], %c0
  // CHECK: %[[LHS_ONE:.*]] = arith.cmpi eq, %[[ARG0]], %c1
  // CHECK: %[[LHS_NEG_ONE:.*]] = arith.cmpi eq, %[[ARG0]], %c-1
  // CHECK: %[[VAL5:.*]] = arith.extui %[[LHS_ONE]] : i1 to i32
  // CHECK: %[[VAL6:.*]] = arith.select %[[RHS_EVEN]], %c1{{.*}}, %c-1
  // CHECK: %[[VAL7:.*]] = arith.select %[[LHS_NEG_ONE]], %[[VAL6]], %[[VAL5]]
  // CHECK: %[[RESULT:.*]] = arith.select %[[RHS_NEG]], %[[VAL7]], %[[FOR_RESULT]]#0
  // CHECK: linalg.yield %[[RESULT]]
  %0 = "stablehlo.power"(%lhs, %rhs) : (tensor<2x2xi32>,
                                   tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}


// -----

func.func @map_mixed(%arg0: tensor<?xf32>,
                     %arg1: tensor<4xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.map"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>}
  : (tensor<?xf32>, tensor<4xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: @map_mixed
// CHECK: linalg.generic

// CHECK-PRIMITIVE-LABEL: @map_mixed
// CHECK-PRIMITIVE: linalg.map

// -----

func.func @map_one_arg(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.map"(%arg0) ({
  ^bb0(%arg2: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg2 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>}
  : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: @map_one_arg
// CHECK: linalg.generic

// CHECK-PRIMITIVE-LABEL: @map_one_arg
// CHECK-PRIMITIVE: linalg.map

// -----

// CHECK-LABEL: @reduce_add_unranked
// CHECK-PRIMITIVE-LABEL: @reduce_add_unranked
func.func @reduce_add_unranked(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> tensor<*xi32> {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4 : tensor<i32>):
    %1 = stablehlo.add %arg3, %arg4 : tensor<i32>
    "stablehlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>, someattr} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}
// CHECK: stablehlo.reduce
// CHECK-PRIMITIVE: stablehlo.reduce

// -----

func.func @reduce_dynamic_output(%arg0: tensor<5x4xi32>, %arg1: tensor<i32>) -> tensor<?xi32> {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4 : tensor<i32>):
    %1 = stablehlo.maximum %arg3, %arg4 : tensor<i32>
    "stablehlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5x4xi32>, tensor<i32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// Regression test: just check that this lowers successfully.
// CHECK-LABEL: @reduce_dynamic_output
// CHECK: linalg.generic

// CHECK-PRIMITIVE-LABEL: @reduce_dynamic_output
// CHECK-PRIMITIVE: linalg.reduce

// -----

func.func @pad_cst(%arg0: tensor<12x4xf32>) -> tensor<18x12xf32> {
  %0 = arith.constant dense<0.0> : tensor<f32>
  %1 = "stablehlo.pad"(%arg0, %0) {
    edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
    edge_padding_low = dense<[4, 5]> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<12x4xf32>, tensor<f32>) -> tensor<18x12xf32>
  func.return %1 : tensor<18x12xf32>
}
// CHECK-LABEL: func @pad_cst
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
//   CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK: tensor.pad %[[ARG0]] low[4, 5] high[2, 3]
//       CHECK:  tensor.yield %[[CST]] : f32
//       CHECK: } : tensor<12x4xf32> to tensor<18x12xf32>

// -----

func.func @pad_tensor(%arg0: tensor<12x4xf32>, %arg1: tensor<f32>) -> tensor<18x12xf32> {
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
    edge_padding_low = dense<[4, 5]> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<12x4xf32>, tensor<f32>) -> tensor<18x12xf32>
  func.return %0 : tensor<18x12xf32>
}
// CHECK-LABEL: func @pad_tensor
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]
//   CHECK-DAG:   %[[PAD:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
//       CHECK:   tensor.pad %[[ARG0]] low[4, 5] high[2, 3]
//       CHECK:     tensor.yield %[[PAD]] : f32
//       CHECK:   } : tensor<12x4xf32> to tensor<18x12xf32>

// -----

func.func @pad_interior(%arg0: tensor<12x4xui32>, %arg1: tensor<ui32>) -> tensor<29x15xui32> {
  %0 = arith.constant dense<0> : tensor<ui32>
  %1 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
    edge_padding_low = dense<[4, 5]> : tensor<2xi64>,
    interior_padding = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<12x4xui32>, tensor<ui32>) -> tensor<29x15xui32>
  func.return %1 : tensor<29x15xui32>
}
// CHECK-LABEL: func @pad_interior
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]
//   CHECK-DAG: %[[CAST0:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<12x4xui32> to tensor<12x4xi32>
//   CHECK-DAG: %[[CAST1:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : tensor<ui32> to tensor<i32>
//   CHECK-DAG: %[[PAD:.+]] = tensor.extract %[[CAST1]][] : tensor<i32>
//       CHECK: %[[INIT:.+]] = tensor.empty() : tensor<29x15xi32>
//       CHECK: %[[FILL:.+]] = linalg.fill ins(%[[PAD]] : i32) outs(%[[INIT]] : tensor<29x15xi32>) -> tensor<29x15xi32>
//       CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[CAST0]] into %[[FILL]][4, 5] [12, 4] [2, 2] : tensor<12x4xi32> into tensor<29x15xi32>

// -----

func.func @pad_interior_negative(%arg0: tensor<12x4xui32>, %arg1: tensor<ui32>) -> tensor<25x9xui32> {
  %0 = arith.constant dense<0> : tensor<ui32>
  %1 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_high = dense<[-2, 3]> : tensor<2xi64>,
    edge_padding_low = dense<[4, -1]> : tensor<2xi64>,
    interior_padding = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<12x4xui32>, tensor<ui32>) -> tensor<25x9xui32>
  func.return %1 : tensor<25x9xui32>
}
// CHECK-LABEL: func @pad_interior_negative
//       CHECK: %[[PAD:.*]] = tensor.insert_slice %{{.+}} into %{{.+}}[4, 0] [12, 4] [2, 2] : tensor<12x4xi32> into tensor<29x10xi32>
//       CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[PAD]][0, 1] [25, 9] [1, 1] : tensor<29x10xi32> to tensor<25x9xi32>

// -----

func.func @torch_index_select(%arg0: tensor<5x1x5xi32>,
                         %arg1: tensor<2xi32>) ->  tensor<2x1x5xi32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) {
    dim = 0 : i64,
    batch_dims = 0 : i64,
    someattr
  } : (tensor<5x1x5xi32>, tensor<2xi32>) -> tensor<2x1x5xi32>
  func.return %0 : tensor<2x1x5xi32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//      CHECK: func @torch_index_select
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[INDEX:[a-zA-Z0-9_]*]]
//      CHECK: %[[INIT1:.+]] = tensor.empty() :
//      CHECK: %[[INIT2:.+]] = tensor.empty() :
//      CHECK: linalg.generic {
// CHECK-SAME:   indexing_maps
// CHECK-SAME:   #[[MAP0]], #[[MAP1]], #[[MAP2]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[INDEX]], %[[INIT1]] :
// CHECK-SAME: outs(%[[INIT2]] :
// CHECK-SAME: {someattr}
//      CHECK: ^{{.+}}(%[[VAL:.+]]: i32, %{{.+}}: i32, %{{.+}}: i32):
//      CHECK:   %[[CAST:.+]] = arith.index_cast %[[VAL]] : i32 to index
//      CHECK:   %[[J:.+]] = linalg.index 1
//      CHECK:   %[[K:.+]] = linalg.index 2
//      CHECK:   %[[VAL2:.+]] = tensor.extract %[[INPUT]][%[[CAST]], %[[J]], %[[K]]] : tensor<5x1x5xi32>
//      CHECK:   linalg.yield %[[VAL2]] : i32

// -----

// CHECK-LABEL: @real_real
// CHECK-SAME: (%[[ARG0:.*]]:
func.func @real_real(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %1 = "stablehlo.real"(%arg0) : (tensor<?xf32>) -> (tensor<?xf32>)
  // CHECK: return %[[ARG0]]
  func.return %1 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @imag_real
func.func @imag_real(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %1 = "stablehlo.imag"(%arg0) : (tensor<?xf32>) -> (tensor<?xf32>)
  // CHECK: %[[CST:.*]] = arith.constant 0
  // CHECK: linalg.generic
  // CHECK: yield %[[CST]]
  func.return %1 : tensor<?xf32>
}

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>,
                  %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    someattr
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}
// The iterations are (Batch Dim, LHS Other Dim, RHS Other dim, Contracting Dim)
// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0)>
// Output is the iterators excluding contracting
// CHECK: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK: func @dot_general(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>)
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C0]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]])
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// Only contracting dims are reductions
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<?x?x?xf32>)
// CHECK-SAME: {someattr}
// CHECK:   ^bb0(%[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
// CHECK:     %[[MUL:.*]] = arith.mulf %[[ARG2]], %[[ARG3]] : f32
// CHECK:     %[[SUM:.*]] = arith.addf %[[ARG4]], %[[MUL]] : f32
// CHECK:     linalg.yield %[[SUM]] : f32
// CHECK: } -> tensor<?x?x?xf32>

// -----

func.func @dot_general_unsigned(%arg0: tensor<?x?x?xui32>,
                  %arg1: tensor<?x?x?xui32>) -> tensor<?x?x?xui32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    someattr
  } : (tensor<?x?x?xui32>, tensor<?x?x?xui32>) -> tensor<?x?x?xui32>
  func.return %0 : tensor<?x?x?xui32>
}

// CHECK-LABEL: func @dot_general_unsigned(
// CHECK: linalg.generic
// CHECK-SAME: ins({{.*}} : tensor<?x?x?xi32>, tensor<?x?x?xi32>)
// CHECK-SAME: outs({{.*}} : tensor<?x?x?xi32>)

// -----

func.func @dot_general_complex(%arg0: tensor<?x?x?xcomplex<f32>>,
                  %arg1: tensor<?x?x?xcomplex<f32>>) -> tensor<?x?x?xcomplex<f32>> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    someattr
  } : (tensor<?x?x?xcomplex<f32>>, tensor<?x?x?xcomplex<f32>>) -> tensor<?x?x?xcomplex<f32>>
  func.return %0 : tensor<?x?x?xcomplex<f32>>
}

// CHECK-LABEL: func @dot_general_complex(
// CHECK: linalg.generic
// CHECK: complex.mul
// CHECK: complex.add

// -----

func.func @dot_general_multiple_batch_dimensions(%arg0: tensor<3x4x2x4xi32>,
             %arg1: tensor<3x4x3x2xi32>) -> tensor<3x4x4x3xi32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [3]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    someattr
  } : (tensor<3x4x2x4xi32>, tensor<3x4x3x2xi32>) -> tensor<3x4x4x3xi32>
  return %0 : tensor<3x4x4x3xi32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
// CHECK: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @dot_general_multiple_batch_dimensions(
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<3x4x2x4xi32>, tensor<3x4x3x2xi32>)
// CHECK-SAME: outs({{.*}} : tensor<3x4x4x3xi32>)
// CHECK-SAME: {someattr}

// -----

// CHECK-LABEL: func @reduce_precision(
// CHECK-DAG: %[[C2:.*]] = arith.constant 1048576 : i32
// CHECK-DAG: %[[C_21:.*]] = arith.constant 20 : i32
// CHECK-DAG: %[[C3:.*]] = arith.constant 524287 : i32
// CHECK-DAG: %[[C4:.*]] = arith.constant -1048576 : i32
// CHECK-DAG: %[[C5:.*]] = arith.constant 2139095040 : i32
// CHECK-DAG: %[[C6:.*]] = arith.constant 1090519040 : i32
// CHECK-DAG: %[[C7:.*]] = arith.constant 1040187392 : i32
// CHECK-DAG: %[[C8:.*]] = arith.constant -2147483648 : i32
// CHECK-DAG: %[[C9:.*]] = arith.constant 2147483647 : i32
// CHECK: linalg.generic
// CHECK: %[[X_AS_INT:.*]] = arith.bitcast %[[IN:.*]] : f32 to i32
// CHECK: %[[ABS_X:.*]] = arith.andi %[[X_AS_INT]], %[[C9]]
// CHECK: %[[IS_NAN:.*]] = arith.cmpi ugt, %[[ABS_X]], %[[C5]]
// CHECK: %[[MASKED:.*]] = arith.andi %[[X_AS_INT]], %[[C2]] : i32
// CHECK: %[[V0:.*]] = arith.shrui %[[MASKED]], %[[C_21]] : i32
// CHECK: %[[V1:.*]] = arith.addi %[[V0]], %[[C3]] : i32
// CHECK: %[[V2:.*]] = arith.addi %[[X_AS_INT]], %[[V1]] : i32
// CHECK: %[[V3:.*]] = arith.andi %[[V2]], %[[C4]] : i32
// CHECK: %[[V4:.*]] = arith.andi %[[V3]], %[[C5]] : i32
// CHECK: %[[V5:.*]] = arith.cmpi ugt, %[[V4]], %[[C6]] : i32
// CHECK: %[[V6:.*]] = arith.cmpi ule, %[[V4]], %[[C7]] : i32
// CHECK: %[[V7:.*]] = arith.andi %[[V3]], %[[C8]] : i32
// CHECK: %[[V8:.*]] = arith.ori %[[V7]], %[[C5]] : i32
// CHECK: %[[V9:.*]] = arith.select %[[V5]], %[[V8]], %[[V3]] : i32
// CHECK: %[[V10:.*]] = arith.select %[[V6]], %[[V7]], %[[V9]] : i32
// CHECK: %[[CONVERTED:.*]] = arith.bitcast %[[V10]] : i32 to f32
// CHECK: %[[RESULT:.*]] = arith.select %[[IS_NAN]], %[[IN]], %[[CONVERTED]]
// CHECK: linalg.yield %[[RESULT]]

// CHECK-PRIMITIVE-LABEL: func @reduce_precision(
// CHECK-PRIMITIVE: linalg.map
func.func @reduce_precision(%arg0: tensor<1x2x3x4xf32>)
                            -> tensor<1x2x3x4xf32> {
  %0 = "stablehlo.reduce_precision"(%arg0) {exponent_bits=3:i32, mantissa_bits=3:i32} : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  return %0 : tensor<1x2x3x4xf32>
}


// -----

// CHECK-LABEL: set_dimension_size
// CHECK-SAME: %[[VALUE:.*]]: tensor<2x?xf32, #stablehlo.bounds<?, 2>
func.func @set_dimension_size(
  %value: tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>,
  %dimension: tensor<i32>)
  -> tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>> {
  // CHECK: tensor.extract_slice %[[VALUE]][0, 0] [2, %{{.*}}] [1, 1] : tensor<2x?xf32, #stablehlo.bounds<?, 2>> to tensor<2x?xf32, #stablehlo.bounds<?, 2>>
  %0 = "stablehlo.set_dimension_size"(%value, %dimension) { dimension = 1 }
    : (tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>, tensor<i32>)
    -> tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>
  func.return %0 : tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>
}

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @transpose
func.func @transpose(%arg0: tensor<2x3x9x5xi32>) -> tensor<3x2x5x9xi32> {
  %0 = "stablehlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>}
        : (tensor<2x3x9x5xi32>) -> tensor<3x2x5x9xi32>
  func.return %0 : tensor<3x2x5x9xi32>
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]

// CHECK-PRIMITIVE-LABEL: func @transpose
// CHECK-PRIMITIVE: linalg.transpose

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @transpose_dynamic
func.func @transpose_dynamic(%arg0: tensor<?x?x9x?xi32>) -> tensor<?x?x?x9xi32> {
  %0 = "stablehlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>, someattr}
        : (tensor<?x?x9x?xi32>) -> tensor<?x?x?x9xi32>
  func.return %0 : tensor<?x?x?x9xi32>
}
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK: %[[D0:.*]] = tensor.dim %arg0, %[[C0]]
// CHECK: %[[D1:.*]] = tensor.dim %arg0, %[[C1]]
// CHECK: %[[D3:.*]] = tensor.dim %arg0, %[[C3]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]], %[[D0]], %[[D3]]) : tensor<?x?x?x9xi32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-SAME: ins(%arg0 : tensor<?x?x9x?xi32>) outs(%[[INIT]] : tensor<?x?x?x9xi32>)
// CHECK-SAME: {someattr}

// CHECK-PRIMITIVE-LABEL: func @transpose_dynamic
// CHECK-PRIMITIVE-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-PRIMITIVE-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-PRIMITIVE-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-PRIMITIVE: %[[D0:.*]] = tensor.dim %arg0, %[[C0]]
// CHECK-PRIMITIVE: %[[D1:.*]] = tensor.dim %arg0, %[[C1]]
// CHECK-PRIMITIVE: %[[D3:.*]] = tensor.dim %arg0, %[[C3]]
// CHECK-PRIMITIVE: %[[INIT:.*]] = tensor.empty(%[[D1]], %[[D0]], %[[D3]]) : tensor<?x?x?x9xi32>
// CHECK-PRIMITIVE: linalg.transpose
// CHECK-PRIMITIVE-SAME: ins(%arg0 : tensor<?x?x9x?xi32>)
// CHECK-PRIMITIVE-SAME: outs(%[[INIT]] : tensor<?x?x?x9xi32>)
// CHECK-PRIMITIVE-SAME: permutation = [1, 0, 3, 2]
// CHECK-PRIMITIVE-SAME: {someattr}

func.func @transpose_unsigned(%arg0: tensor<2x2xui32>) -> tensor<2x2xui32> {
  %0 = "stablehlo.transpose"(%arg0) {
    permutation = dense<[1, 0]> : tensor<2xi64>,
    result_layout = dense<[0, 1]> : tensor<2xindex>
  } : (tensor<2x2xui32>) -> tensor<2x2xui32>
  return %0 : tensor<2x2xui32>
}

// Regression test. Just check that unsigned ints lower successfully.
// CHECK-LABEL: func @transpose_unsigned
// CHECK-PRIMITIVE-LABEL: func @transpose_unsigned

// -----

// CHECK-LABEL: func @minf
func.func @minf(%lhs: tensor<2x2xf32>, %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.minimum"(%lhs, %rhs) {someattr}
          : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}
// CHECK: tensor.empty() : tensor<2x2xf32>
// CHECK: linalg.generic
// CHECK-SAME: {someattr}
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.minf %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.minf

// -----

// CHECK-LABEL: func @maxi
func.func @maxi(%lhs: tensor<2x2xi32>, %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = "stablehlo.maximum"(%lhs, %rhs)
          : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}
// CHECK: tensor.empty() : tensor<2x2xi32>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.maxsi %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.maxsi

// -----

// CHECK-LABEL: func @maxu
func.func @maxu(%lhs: tensor<2x2xui32>, %rhs: tensor<2x2xui32>) -> tensor<2x2xui32> {
  %0 = "stablehlo.maximum"(%lhs, %rhs)
          : (tensor<2x2xui32>, tensor<2x2xui32>) -> tensor<2x2xui32>
  func.return %0 : tensor<2x2xui32>
}
// CHECK: tensor.empty() : tensor<2x2xi32>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.maxui %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.maxui

// -----

// CHECK-LABEL: func @maxi1
func.func @maxi1(%lhs: tensor<?x?xi1>, %rhs: tensor<?x?xi1>) -> tensor<?x?xi1> {
  %0 = "stablehlo.maximum"(%lhs, %rhs)
          : (tensor<?x?xi1>, tensor<?x?xi1>) -> tensor<?x?xi1>
  func.return %0 : tensor<?x?xi1>
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i1, %[[RHS_IN:.*]]: i1, %{{.*}}: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.maxui %[[LHS_IN]], %[[RHS_IN]] : i1
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.maxui


// -----

func.func @dot_matmul(%arg0: tensor<2x3xf32>,
                 %arg1: tensor<3x?xf32>) -> tensor<2x?xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) {someattr}
           : (tensor<2x3xf32>, tensor<3x?xf32>) -> tensor<2x?xf32>
  func.return %0 : tensor<2x?xf32>
}
// CHECK-LABEL: func @dot_matmul(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x3xf32>, %[[ARG1:.*]]: tensor<3x?xf32>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]])
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matmul
// CHECK-SAME: {someattr}
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x3xf32>, tensor<3x?xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x?xf32>)

// -----

func.func @dot_matmul_complex(%arg0: tensor<2x3xcomplex<f32>>,
                 %arg1: tensor<3x?xcomplex<f32>>) -> tensor<2x?xcomplex<f32>> {
  %0 = "stablehlo.dot"(%arg0, %arg1) {someattr}
           : (tensor<2x3xcomplex<f32>>, tensor<3x?xcomplex<f32>>) -> tensor<2x?xcomplex<f32>>
  func.return %0 : tensor<2x?xcomplex<f32>>
}
// CHECK-LABEL: func @dot_matmul_complex(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x3xcomplex<f32>>, %[[ARG1:.*]]: tensor<3x?xcomplex<f32>>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]]) : tensor<2x?x
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matmul
// CHECK-SAME: {someattr}
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x3xcomplex<f32>>, tensor<3x?xcomplex<f32>>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x?xcomplex<f32>>)

// -----

func.func @dot_matmul_i8_i8_i32(%arg0: tensor<2x3xi8>,
                 %arg1: tensor<3x?xi8>) -> tensor<2x?xi32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<2x3xi8>,
                                   tensor<3x?xi8>) -> tensor<2x?xi32>
  func.return %0 : tensor<2x?xi32>
}
// CHECK-LABEL: func @dot_matmul_i8_i8_i32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x3xi8>, %[[ARG1:.*]]: tensor<3x?xi8>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]]) : tensor<2x?x
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matmul
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x3xi8>, tensor<3x?xi8>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x?xi32>)

// -----

func.func @dot_matmul_i16_i16_i32(%arg0: tensor<2x3xi16>,
                 %arg1: tensor<3x?xi16>) -> tensor<2x?xi32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<2x3xi16>,
                                   tensor<3x?xi16>) -> tensor<2x?xi32>
  func.return %0 : tensor<2x?xi32>
}
// CHECK-LABEL: func @dot_matmul_i16_i16_i32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x3xi16>, %[[ARG1:.*]]: tensor<3x?xi16>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]]) : tensor<2x?x
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matmul
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x3xi16>, tensor<3x?xi16>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x?xi32>)

// -----

func.func @dot_matmul_i32_i32_i32(%arg0: tensor<2x3xi32>,
                 %arg1: tensor<3x?xi32>) -> tensor<2x?xi32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<2x3xi32>,
                                   tensor<3x?xi32>) -> tensor<2x?xi32>
  func.return %0 : tensor<2x?xi32>
}
// CHECK-LABEL: func @dot_matmul_i32_i32_i32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x3xi32>, %[[ARG1:.*]]: tensor<3x?xi32>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]]) : tensor<2x?x
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matmul
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x3xi32>, tensor<3x?xi32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x?xi32>)

// -----

func.func @dot_matvec(%arg0: tensor<?x3xf32>,
                 %arg1: tensor<3xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<?x3xf32>,
                                   tensor<3xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK-LABEL: func @dot_matvec(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x3xf32>, %[[ARG1:.*]]: tensor<3xf32>)
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]])
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matvec
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<?x3xf32>, tensor<3xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<?xf32>)

// -----

func.func @dot_vecmat(%arg0: tensor<3xf32>,
                 %arg1: tensor<3x?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<3xf32>,
                                   tensor<3x?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK-LABEL: func @dot_vecmat(
// CHECK-SAME: %[[ARG0:.*]]: tensor<3xf32>, %[[ARG1:.*]]: tensor<3x?xf32>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]])
// CHECK: linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.vecmat
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<3xf32>, tensor<3x?xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<?xf32>)

// -----

func.func @dot_dot(%arg0: tensor<?xf32>,
              %arg1: tensor<?xf32>) -> tensor<f32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: func @dot_dot(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<?xf32>)
// CHECK: %[[INIT:.*]] = tensor.empty()
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.dot
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<?xf32>, tensor<?xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<f32>)

// -----

func.func @dot_dot_unsigned(%arg0: tensor<?xui32>,
              %arg1: tensor<?xui32>) -> tensor<ui32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<?xui32>, tensor<?xui32>) -> tensor<ui32>
  func.return %0 : tensor<ui32>
}
// CHECK-LABEL: func @dot_dot_unsigned(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?xui32>, %[[ARG1:.*]]: tensor<?xui32>)
// CHECK: %[[INIT:.*]] = tensor.empty()
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}outs(%[[INIT]]
// CHECK: linalg.dot
// CHECK-SAME: ins(%{{.*}} : tensor<?xi32>, tensor<?xi32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<i32>)

// -----

// CHECK-LABEL: @clamp_static
// CHECK-SAME: %[[LB:.*]]: tensor<4xf32>, %[[X:.*]]: tensor<4xf32>, %[[UB:.*]]: tensor<4xf32>
func.func @clamp_static(%lb : tensor<4xf32>, %x : tensor<4xf32>, %ub : tensor<4xf32>)
    -> tensor<4xf32> {
  // CHECK: %[[INIT:.*]] = tensor.empty
  // CHECK: %[[RESULT:.*]] = linalg.generic {{.*}} ins(%[[LB]], %[[X]], %[[UB]] : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%[[INIT]] : tensor<4xf32>)
  // CHECK: ^bb0(%[[SCALAR_LB:.*]]: f32, %[[SCALAR_X:.*]]: f32, %[[SCALAR_UB:.*]]: f32, %{{.*}}: f32):
  // CHECK:   %[[MAX:.*]] = arith.maxf %[[SCALAR_LB]], %[[SCALAR_X]] : f32
  // CHECK:   %[[MIN:.*]] = arith.minf %[[MAX]], %[[SCALAR_UB]] : f32
  // CHECK:   linalg.yield %[[MIN]]
  // CHECK: } -> tensor<4xf32>
  // CHECK: return %[[RESULT]] : tensor<4xf32>
  %0 = "stablehlo.clamp"(%lb, %x, %ub) : (tensor<4xf32>, tensor<4xf32>,
      tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK-PRIMITIVE-LABEL: @clamp_static
// CHECK-PRIMITIVE-SAME: %[[LB:.*]]: tensor<4xf32>, %[[X:.*]]: tensor<4xf32>, %[[UB:.*]]: tensor<4xf32>

// CHECK-PRIMITIVE: %[[INIT:.*]] = tensor.empty
// CHECK-PRIMITIVE: %[[RESULT:.*]] = linalg.map ins(%[[LB]], %[[X]], %[[UB]] : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%[[INIT]] : tensor<4xf32>)
// CHECK-PRIMITIVE: (%[[SCALAR_LB:.*]]: f32, %[[SCALAR_X:.*]]: f32, %[[SCALAR_UB:.*]]: f32)
// CHECK-PRIMITIVE:   %[[MAX:.*]] = arith.maxf %[[SCALAR_LB]], %[[SCALAR_X]] : f32
// CHECK-PRIMITIVE:   %[[MIN:.*]] = arith.minf %[[MAX]], %[[SCALAR_UB]] : f32
// CHECK-PRIMITIVE:   linalg.yield %[[MIN]]
// CHECK-PRIMITIVE: return %[[RESULT]] : tensor<4xf32>

// -----

// CHECK-LABEL: @clamp_dynamic
// CHECK-SAME: %[[LB:.*]]: tensor<?xf32>, %[[X:.*]]: tensor<?xf32>, %[[UB:.*]]: tensor<?xf32>
func.func @clamp_dynamic(%lb : tensor<?xf32>, %x : tensor<?xf32>, %ub : tensor<?xf32>)
    -> tensor<?xf32> {
  // CHECK: %[[INIT:.*]] = tensor.empty
  // CHECK: %[[RESULT:.*]] = linalg.generic {{.*}} ins(%[[LB]], %[[X]], %[[UB]] : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%[[INIT]] : tensor<?xf32>)
  // CHECK: ^bb0(%[[SCALAR_LB:.*]]: f32, %[[SCALAR_X:.*]]: f32, %[[SCALAR_UB:.*]]: f32, %{{.*}}: f32):
  // CHECK:   %[[MAX:.*]] = arith.maxf %[[SCALAR_LB]], %[[SCALAR_X]] : f32
  // CHECK:   %[[MIN:.*]] = arith.minf %[[MAX]], %[[SCALAR_UB]] : f32
  // CHECK:   linalg.yield %[[MIN]]
  // CHECK: } -> tensor<?xf32>
  // CHECK: return %[[RESULT]] : tensor<?xf32>
  %0 = "stablehlo.clamp"(%lb, %x, %ub) : (tensor<?xf32>, tensor<?xf32>,
      tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-PRIMITIVE-LABEL: @clamp_dynamic
// CHECK-PRIMITIVE: linalg.map

// -----

func.func @clamp_mixed(%lb : tensor<4xf32>, %x : tensor<?xf32>, %ub : tensor<?xf32>)
    -> tensor<?xf32> {
  %0 = "stablehlo.clamp"(%lb, %x, %ub) : (tensor<4xf32>, tensor<?xf32>,
      tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: @clamp_mixed
// CHECK: linalg.generic

// CHECK-PRIMITIVE-LABEL: @clamp_mixed
// CHECK-PRIMITIVE: linalg.map

// -----

func.func @clamp_scalar(%lb : tensor<f32>, %x : tensor<?xf32>, %ub : tensor<f32>)
    -> tensor<?xf32> {
  %0 = "stablehlo.clamp"(%lb, %x, %ub) : (tensor<f32>, tensor<?xf32>,
      tensor<f32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: @clamp_scalar
// CHECK: linalg.generic

// CHECK-PRIMITIVE-LABEL: @clamp_scalar
// CHECK-PRIMITIVE-SAME: %[[LB:.*]]: tensor<f32>, %[[X:.*]]: tensor<?xf32>, %[[UB:.*]]: tensor<f32>

// CHECK-PRIMITIVE-DAG: %[[INIT:.*]] = tensor.empty
// CHECK-PRIMITIVE-DAG: %[[SCALAR_LB:.*]] = tensor.extract %[[LB]]
// CHECK-PRIMITIVE-DAG: %[[SCALAR_UB:.*]] = tensor.extract %[[UB]]
// CHECK-PRIMITIVE: %[[RESULT:.*]] = linalg.map ins(%[[X]] : tensor<?xf32>) outs(%[[INIT]] : tensor<?xf32>)
// CHECK-PRIMITIVE: (%[[SCALAR_X:.*]]: f32)
// CHECK-PRIMITIVE:   %[[MAX:.*]] = arith.maxf %[[SCALAR_LB]], %[[SCALAR_X]] : f32
// CHECK-PRIMITIVE:   %[[MIN:.*]] = arith.minf %[[MAX]], %[[SCALAR_UB]] : f32
// CHECK-PRIMITIVE:   linalg.yield %[[MIN]]
// CHECK-PRIMITIVE: return %[[RESULT]]


// -----

func.func @clamp_scalar_mixed(%lb : tensor<f32>, %x : tensor<?xf32>, %ub : tensor<?xf32>)
    -> tensor<?xf32> {
  %0 = "stablehlo.clamp"(%lb, %x, %ub) : (tensor<f32>, tensor<?xf32>,
      tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: @clamp_scalar_mixed
// CHECK: linalg.generic

// CHECK-PRIMITIVE-LABEL: @clamp_scalar_mixed
// CHECK-PRIMITIVE: linalg.map
