// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-stablehlo-preprocessing-lower-complex))" %s | FileCheck %s

// CHECK-LABEL: @add
func.func @add(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>, %arg3 : tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %2 = "stablehlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)
  %3 = "stablehlo.complex"(%arg2, %arg3) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = stablehlo.add %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = stablehlo.add %arg1, %arg3
  %4 = "stablehlo.add"(%2, %3) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)
  %5 = stablehlo.real %4 : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %6 = stablehlo.imag %4 : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return [[VAL0]], [[VAL1]]
  func.return %5, %6 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: @add_unranked
func.func @add_unranked(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>, %arg2 : tensor<*xf32>, %arg3 : tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %2 = "stablehlo.complex"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)
  %3 = "stablehlo.complex"(%arg2, %arg3) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = stablehlo.add %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = stablehlo.add %arg1, %arg3
  %4 = "stablehlo.add"(%2, %3) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> (tensor<*xcomplex<f32>>)
  %5 = stablehlo.real %4 : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)
  %6 = stablehlo.imag %4 : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)

  // CHECK: return [[VAL0]], [[VAL1]]
  func.return %5, %6 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: @sub
func.func @sub(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>, %arg3 : tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %2 = "stablehlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)
  %3 = "stablehlo.complex"(%arg2, %arg3) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = stablehlo.subtract %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = stablehlo.subtract %arg1, %arg3
  %4 = "stablehlo.subtract"(%2, %3) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)
  %5 = stablehlo.real %4 : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %6 = stablehlo.imag %4 : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return [[VAL0]], [[VAL1]]
  func.return %5, %6 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: @sub_unranked
func.func @sub_unranked(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>, %arg2 : tensor<*xf32>, %arg3 : tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %2 = "stablehlo.complex"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)
  %3 = "stablehlo.complex"(%arg2, %arg3) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = stablehlo.subtract %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = stablehlo.subtract %arg1, %arg3
  %4 = "stablehlo.subtract"(%2, %3) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> (tensor<*xcomplex<f32>>)
  %5 = stablehlo.real %4 : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)
  %6 = stablehlo.imag %4 : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)

  // CHECK: return [[VAL0]], [[VAL1]]
  func.return %5, %6 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: @mul
func.func @mul(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>, %arg3 : tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %2 = "stablehlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)
  %3 = "stablehlo.complex"(%arg2, %arg3) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = stablehlo.multiply %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = stablehlo.multiply %arg1, %arg3
  // CHECK-DAG: [[VAL2:%.+]] = stablehlo.subtract [[VAL0]], [[VAL1]]
  // CHECK-DAG: [[VAL3:%.+]] = stablehlo.multiply %arg0, %arg3
  // CHECK-DAG: [[VAL4:%.+]] = stablehlo.multiply %arg1, %arg2
  // CHECK-DAG: [[VAL5:%.+]] = stablehlo.add [[VAL3]], [[VAL4]]
  %4 = "stablehlo.multiply"(%2, %3) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)
  %5 = stablehlo.real %4 : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %6 = stablehlo.imag %4 : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return %2, %5 : tensor<2xf32>, tensor<2xf32>
  func.return %5, %6 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: @mul_unranked
func.func @mul_unranked(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>, %arg2 : tensor<*xf32>, %arg3 : tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %2 = "stablehlo.complex"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)
  %3 = "stablehlo.complex"(%arg2, %arg3) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = stablehlo.multiply %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = stablehlo.multiply %arg1, %arg3
  // CHECK-DAG: [[VAL2:%.+]] = stablehlo.subtract [[VAL0]], [[VAL1]]
  // CHECK-DAG: [[VAL3:%.+]] = stablehlo.multiply %arg0, %arg3
  // CHECK-DAG: [[VAL4:%.+]] = stablehlo.multiply %arg1, %arg2
  // CHECK-DAG: [[VAL5:%.+]] = stablehlo.add [[VAL3]], [[VAL4]]
  %4 = "stablehlo.multiply"(%2, %3) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> (tensor<*xcomplex<f32>>)
  %5 = stablehlo.real %4 : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)
  %6 = stablehlo.imag %4 : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)

  // CHECK: return %2, %5 : tensor<*xf32>, tensor<*xf32>
  func.return %5, %6 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: @div
func.func @div(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>, %arg3 : tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %2 = "stablehlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)
  %3 = "stablehlo.complex"(%arg2, %arg3) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = stablehlo.negate %arg3

  // Compute the numerator's real component:
  //   numerator.real = lhs.real * rhs.real  lhs.imag * rhs.imag
  // CHECK-DAG: [[VAL1:%.+]] = stablehlo.multiply %arg0, %arg2
  // CHECK-DAG: [[VAL2:%.+]] = stablehlo.multiply %arg1, [[VAL0]]
  // CHECK-DAG: [[VAL3:%.+]] = stablehlo.subtract [[VAL1]], [[VAL2]]

  // Compute the real valued denominator as rhs * con(rhs):
  //   denominator = rhs.real * rhs.real + rhs.imag * rhs.imag
  // CHECK-DAG: [[VAL4:%.+]] = stablehlo.multiply %arg2, %arg2
  // CHECK-DAG: [[VAL5:%.+]] = stablehlo.multiply %arg3, %arg3
  // CHECK-DAG: [[VAL6:%.+]] = stablehlo.add [[VAL4]], [[VAL5]]

  // Compute the numerator's imaginary component:
  //   numerator.imag = lhs.imag * rhs.real - lhs.real * rhs.imag
  // CHECK-DAG: [[VAL7:%.+]] = stablehlo.multiply %arg1, %arg2
  // CHECK-DAG: [[VAL8:%.+]] = stablehlo.multiply %arg0, [[VAL0]]
  // CHECK-DAG: [[VAL9:%.+]] = stablehlo.add [[VAL8]], [[VAL7]]

  // Divide the numerator by the real valued denominator.
  // CHECK-DAG: [[VAL10:%.+]] = stablehlo.divide [[VAL3]], [[VAL6]]
  // CHECK-DAG: [[VAL11:%.+]] = stablehlo.divide [[VAL9]], [[VAL6]]
  %4 = "stablehlo.divide"(%2, %3) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)

  %5 = stablehlo.real %4 : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %6 = stablehlo.imag %4 : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return [[VAL10]], [[VAL11]]
  func.return %5, %6 : tensor<2xf32>, tensor<2xf32>
}

// -----

// CHECK-LABEL: @div_unranked
func.func @div_unranked(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>, %arg2 : tensor<*xf32>, %arg3 : tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %2 = "stablehlo.complex"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)
  %3 = "stablehlo.complex"(%arg2, %arg3) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = stablehlo.negate %arg3

  // Compute the numerator's real component:
  //   numerator.real = lhs.real * rhs.real  lhs.imag * rhs.imag
  // CHECK-DAG: [[VAL1:%.+]] = stablehlo.multiply %arg0, %arg2
  // CHECK-DAG: [[VAL2:%.+]] = stablehlo.multiply %arg1, [[VAL0]]
  // CHECK-DAG: [[VAL3:%.+]] = stablehlo.subtract [[VAL1]], [[VAL2]]

  // Compute the real valued denominator as rhs * con(rhs):
  //   denominator = rhs.real * rhs.real + rhs.imag * rhs.imag
  // CHECK-DAG: [[VAL4:%.+]] = stablehlo.multiply %arg2, %arg2
  // CHECK-DAG: [[VAL5:%.+]] = stablehlo.multiply %arg3, %arg3
  // CHECK-DAG: [[VAL6:%.+]] = stablehlo.add [[VAL4]], [[VAL5]]

  // Compute the numerator's imaginary component:
  //   numerator.imag = lhs.imag * rhs.real - lhs.real * rhs.imag
  // CHECK-DAG: [[VAL7:%.+]] = stablehlo.multiply %arg1, %arg2
  // CHECK-DAG: [[VAL8:%.+]] = stablehlo.multiply %arg0, [[VAL0]]
  // CHECK-DAG: [[VAL9:%.+]] = stablehlo.add [[VAL8]], [[VAL7]]

  // Divide the numerator by the real valued denominator.
  // CHECK-DAG: [[VAL10:%.+]] = stablehlo.divide [[VAL3]], [[VAL6]]
  // CHECK-DAG: [[VAL11:%.+]] = stablehlo.divide [[VAL9]], [[VAL6]]

  %4 = "stablehlo.divide"(%2, %3) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> (tensor<*xcomplex<f32>>)

  %5 = stablehlo.real %4 : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)
  %6 = stablehlo.imag %4 : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)

  // CHECK: return [[VAL10]], [[VAL11]]
  func.return %5, %6 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: @abs
func.func @abs(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>) -> (tensor<2xf32>) {
  %0 = "stablehlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = stablehlo.multiply %arg0, %arg0
  // CHECK-DAG: [[VAL1:%.+]] = stablehlo.multiply %arg1, %arg1
  // CHECK-DAG: [[VAL2:%.+]] = stablehlo.add [[VAL0]], [[VAL1]]
  // CHECK-DAG: [[VAL3:%.+]] = stablehlo.sqrt [[VAL2]]
  %1 = "stablehlo.abs"(%0) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return [[VAL3]]
  func.return %1 : tensor<2xf32>
}

// CHECK-LABEL: @exp
func.func @exp(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %0 = "stablehlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[EXP:%.+]] = stablehlo.exponential %arg0
  // CHECK-DAG: [[COS:%.+]] = stablehlo.cosine %arg1
  // CHECK-DAG: [[SIN:%.+]] = stablehlo.sine %arg1
  // CHECK-DAG: [[OUTR:%.+]] = stablehlo.multiply [[COS]], [[EXP]]
  // CHECK-DAG: [[OUTI:%.+]] = stablehlo.multiply [[SIN]], [[EXP]]
  %1 = stablehlo.exponential %0 : (tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)

  %2 = stablehlo.real %1 : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %3 = stablehlo.imag %1 : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: [[OUTR]], [[OUTI]]
  func.return %2, %3 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: @exp_complex
func.func @exp_complex(%arg0 : tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>) {
  // CHECK-DAG: [[REAL:%.+]] = stablehlo.real %arg0
  // CHECK-DAG: [[IMAG:%.+]] = stablehlo.imag %arg0
  // CHECK-DAG: [[EXP:%.+]] = stablehlo.exponential [[REAL]]
  // CHECK-DAG: [[COS:%.+]] = stablehlo.cosine [[IMAG]]
  // CHECK-DAG: [[SIN:%.+]] = stablehlo.sine [[IMAG]]
  // CHECK-DAG: [[OUTR:%.+]] = stablehlo.multiply [[COS]], [[EXP]]
  // CHECK-DAG: [[OUTI:%.+]] = stablehlo.multiply [[SIN]], [[EXP]]
  // CHECK-DAG: [[OUT:%.+]] = stablehlo.complex [[OUTR]], [[OUTI]]
  %0 = stablehlo.exponential %arg0 : (tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)

  // CHECK: [[OUT]]
  func.return %0 : tensor<2xcomplex<f32>>
}

// CHECK-LABEL: @exp_unranked
func.func @exp_unranked(%arg0 : tensor<*xcomplex<f32>>) -> (tensor<*xcomplex<f32>>) {
  // CHECK-DAG: [[REAL:%.+]] = stablehlo.real %arg0
  // CHECK-DAG: [[IMAG:%.+]] = stablehlo.imag %arg0
  // CHECK-DAG: [[EXP:%.+]] = stablehlo.exponential [[REAL]]
  // CHECK-DAG: [[COS:%.+]] = stablehlo.cosine [[IMAG]]
  // CHECK-DAG: [[SIN:%.+]] = stablehlo.sine [[IMAG]]
  // CHECK-DAG: [[OUTR:%.+]] = stablehlo.multiply [[COS]], [[EXP]]
  // CHECK-DAG: [[OUTI:%.+]] = stablehlo.multiply [[SIN]], [[EXP]]
  // CHECK-DAG: [[OUT:%.+]] = stablehlo.complex [[OUTR]], [[OUTI]]
  %0 = stablehlo.exponential %arg0 : (tensor<*xcomplex<f32>>) -> (tensor<*xcomplex<f32>>)

  // CHECK: [[OUT]]
  func.return %0 : tensor<*xcomplex<f32>>
}

// CHECK-LABEL: @compare_eq
// CHECK: ([[LHS:%.+]]: tensor<2xcomplex<f32>>, [[RHS:%.+]]: tensor<2xcomplex<f32>>)
func.func @compare_eq(%lhs : tensor<2xcomplex<f32>>, %rhs: tensor<2xcomplex<f32>>) -> (tensor<2xi1>) {
  // CHECK-DAG: [[REAL_LHS:%.+]] = stablehlo.real [[LHS]]
  // CHECK-DAG: [[REAL_RHS:%.+]] = stablehlo.real [[RHS]]
  // CHECK-DAG: [[OUTR:%.+]] = stablehlo.compare EQ, [[REAL_LHS]], [[REAL_RHS]]
  // CHECK-DAG: [[IMAG_LHS:%.+]] = stablehlo.imag [[LHS]]
  // CHECK-DAG: [[IMAG_RHS:%.+]] = stablehlo.imag [[RHS]]
  // CHECK-DAG: [[OUTI:%.+]] = stablehlo.compare EQ, [[IMAG_LHS]], [[IMAG_RHS]]
  // CHECK-DAG: [[OUT:%.+]] = stablehlo.and [[OUTR]], [[OUTI]]
  %0 = "stablehlo.compare"(%lhs, %rhs) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xi1>

  // CHECK: return [[OUT]]
  func.return %0 : tensor<2xi1>
}

// CHECK-LABEL: @compare_ne
// CHECK: ([[LHS:%.+]]: tensor<2xcomplex<f32>>, [[RHS:%.+]]: tensor<2xcomplex<f32>>)
func.func @compare_ne(%lhs : tensor<2xcomplex<f32>>, %rhs: tensor<2xcomplex<f32>>) -> (tensor<2xi1>) {
  // CHECK-DAG: [[REAL_LHS:%.+]] = stablehlo.real [[LHS]]
  // CHECK-DAG: [[REAL_RHS:%.+]] = stablehlo.real [[RHS]]
  // CHECK-DAG: [[OUTR:%.+]] = stablehlo.compare NE, [[REAL_LHS]], [[REAL_RHS]]
  // CHECK-DAG: [[IMAG_LHS:%.+]] = stablehlo.imag [[LHS]]
  // CHECK-DAG: [[IMAG_RHS:%.+]] = stablehlo.imag [[RHS]]
  // CHECK-DAG: [[OUTI:%.+]] = stablehlo.compare NE, [[IMAG_LHS]], [[IMAG_RHS]]
  // CHECK-DAG: [[OUT:%.+]] = stablehlo.or [[OUTR]], [[OUTI]]
  %0 = "stablehlo.compare"(%lhs, %rhs) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xi1>

  // CHECK: return [[OUT]]
  func.return %0 : tensor<2xi1>
}

// CHECK-LABEL: @sin
func.func @sin(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  // CHECK-DAG: %[[TWO:.+]] = stablehlo.constant dense<2.000000e+00>
  // CHECK-DAG: %[[SIN:.+]] = stablehlo.sine %arg0
  // CHECK-DAG: %[[EXP:.+]] = stablehlo.exponential %arg1
  // CHECK-DAG: %[[NEG:.+]] = stablehlo.negate %arg1
  // CHECK-DAG: %[[NEXP:.+]] = stablehlo.exponential %[[NEG]]
  // CHECK-DAG: %[[ADD:.+]] = stablehlo.add %[[EXP]], %[[NEXP]]
  // CHECK-DAG: %[[MUL:.+]] = stablehlo.multiply %[[SIN]], %[[ADD]]
  // CHECK-DAG: %[[RDIV:.+]] = stablehlo.divide %[[MUL]], %[[TWO]]
  // CHECK-DAG: %[[COS:.+]] = stablehlo.cosine %arg0
  // CHECK-DAG: %[[SUB:.+]] = stablehlo.subtract %[[EXP]], %[[NEXP]]
  // CHECK-DAG: %[[IMUL:.+]] = stablehlo.multiply %[[COS]], %[[SUB]]
  // CHECK-DAG: %[[IDIV:.+]] = stablehlo.divide %[[IMUL]], %[[TWO]]
  %0 = "stablehlo.complex"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> (tensor<10xcomplex<f32>>)
  %1 = stablehlo.sine %0 : (tensor<10xcomplex<f32>>) -> (tensor<10xcomplex<f32>>)
  %2 = stablehlo.real %1 : (tensor<10xcomplex<f32>>) -> (tensor<10xf32>)
  %3 = stablehlo.imag %1 : (tensor<10xcomplex<f32>>) -> (tensor<10xf32>)

  // CHECK: return %[[RDIV]], %[[IDIV]]
  func.return %2, %3 : tensor<10xf32>, tensor<10xf32>
}

// CHECK-LABEL: @cos
func.func @cos(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  // CHECK-DAG: %[[TWO:.+]] = stablehlo.constant dense<2.000000e+00>
  // CHECK-DAG: %[[COS:.+]] = stablehlo.cosine %arg0
  // CHECK-DAG: %[[EXP:.+]] = stablehlo.exponential %arg1
  // CHECK-DAG: %[[NEG:.+]] = stablehlo.negate %arg1
  // CHECK-DAG: %[[NEXP:.+]] = stablehlo.exponential %[[NEG]]
  // CHECK-DAG: %[[ADD:.+]] = stablehlo.add %[[EXP]], %[[NEXP]]
  // CHECK-DAG: %[[MUL:.+]] = stablehlo.multiply %[[COS]], %[[ADD]]
  // CHECK-DAG: %[[RDIV:.+]] = stablehlo.divide %[[MUL]], %[[TWO]]
  // CHECK-DAG: %[[SIN:.+]] = stablehlo.sine %arg0
  // CHECK-DAG: %[[SUB:.+]] = stablehlo.subtract %[[NEXP]], %[[EXP]]
  // CHECK-DAG: %[[IMUL:.+]] = stablehlo.multiply %[[SIN]], %[[SUB]]
  // CHECK-DAG: %[[IDIV:.+]] = stablehlo.divide %[[IMUL]], %[[TWO]]
  %0 = "stablehlo.complex"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> (tensor<10xcomplex<f32>>)
  %1 = stablehlo.cosine %0 : (tensor<10xcomplex<f32>>) -> (tensor<10xcomplex<f32>>)
  %2 = stablehlo.real %1 : (tensor<10xcomplex<f32>>) -> (tensor<10xf32>)
  %3 = stablehlo.imag %1 : (tensor<10xcomplex<f32>>) -> (tensor<10xf32>)

  // CHECK: return %[[RDIV]], %[[IDIV]]
  func.return %2, %3 : tensor<10xf32>, tensor<10xf32>
}

// CHECK-LABEL: @dot_complex
func.func @dot_complex(%arg0: tensor<2x3xcomplex<f32>>, %arg1:  tensor<3x4xcomplex<f32>>) -> (tensor<2x4xcomplex<f32>>) {
  // CHECK-DAG: [[ROP0:%.+]] = stablehlo.real %arg0
  // CHECK-DAG: [[IOP0:%.+]] = stablehlo.imag %arg0
  // CHECK-DAG: [[ROP1:%.+]] = stablehlo.real %arg1
  // CHECK-DAG: [[IOP1:%.+]] = stablehlo.imag %arg1
  // CHECK-DAG: %[[RR:.+]] = stablehlo.dot [[ROP0]], [[ROP1]]
  // CHECK-DAG: %[[II:.+]] = stablehlo.dot [[IOP0]], [[IOP1]]
  // CHECK-DAG: %[[RPART:.+]] = stablehlo.subtract %[[RR]], %[[II]]
  // CHECK-DAG: %[[RI:.+]] = stablehlo.dot [[ROP0]], [[IOP1]]
  // CHECK-DAG: %[[IR:.+]] = stablehlo.dot [[IOP0]], [[ROP1]]
  // CHECK-DAG: %[[IPART:.+]] = stablehlo.add %[[RI]], %[[IR]]
  // CHECK-DAG: %[[CMPLX:.+]] = stablehlo.complex %[[RPART]], %[[IPART]]
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x3xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> tensor<2x4xcomplex<f32>>
  // CHECK: return %[[CMPLX]]
  return %0 : tensor<2x4xcomplex<f32>>
}
