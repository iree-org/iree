// RUN: iree-opt --iree-stablehlo-canonicalize --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @add
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<2xi32>, [[ARG1:%.+]]: tensor<f32>)
func.func @add(%arg0: tensor<2xi32>, %arg1: tensor<f32>)
  -> (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %cn0 = stablehlo.constant dense<-0.0> : tensor<f32>
  %c0_2 = stablehlo.constant dense<0> : tensor<2xi32>
  %c1 = stablehlo.constant dense<5> : tensor<i32>
  %c2 = stablehlo.constant dense<3.0> : tensor<f32>
  %c3 = stablehlo.constant dense<[1, 2]> : tensor<2xi32>

  %0 = stablehlo.add %c0, %c1 : tensor<i32>
  %1 = stablehlo.add %c1, %c1 : tensor<i32>
  %2 = stablehlo.add %c2, %c2 : tensor<f32>
  %3 = stablehlo.add %arg1, %cn0 : tensor<f32>

  %4 = stablehlo.add %c0_2, %arg0 : tensor<2xi32>
  %5 = stablehlo.add %c3, %arg0 : tensor<2xi32>
  %6 = stablehlo.add %c3, %c3 : tensor<2xi32>

  // CHECK-DAG:  [[C0:%.+]] = stablehlo.constant dense<5> : tensor<i32>
  // CHECK-DAG:  [[C1:%.+]] = stablehlo.constant dense<10> : tensor<i32>
  // CHECK-DAG:  [[C2:%.+]] = stablehlo.constant dense<6.000000e+00> : tensor<f32>
  // CHECK-DAG:  [[C3:%.+]] = stablehlo.constant dense<[2, 4]> : tensor<2xi32>
  // CHECK-DAG:  [[C4:%.+]] = stablehlo.constant dense<[1, 2]> : tensor<2xi32>

  // CHECK-DAG:  [[A0:%.+]] = stablehlo.add [[ARG0]], [[C4]] : tensor<2xi32>

  // CHECK-NEXT: return [[C0]], [[C1]], [[C2]], [[ARG1]], [[ARG0]], [[A0]], [[C3]]
  return %0, %1, %2, %3, %4, %5, %6 : tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>
}

// -----

// CHECK-LABEL: func.func @subtract
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<2xi32>, [[ARG1:%.+]]: tensor<f32>)
func.func @subtract(%arg0: tensor<2xi32>, %arg1: tensor<f32>)
  -> (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %cp0 = stablehlo.constant dense<0.0> : tensor<f32>
  %c0_2 = stablehlo.constant dense<0> : tensor<2xi32>
  %c1 = stablehlo.constant dense<5> : tensor<i32>
  %c2 = stablehlo.constant dense<3.0> : tensor<f32>
  %c3 = stablehlo.constant dense<[1, 2]> : tensor<2xi32>
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %c5 = stablehlo.constant dense<[0, 1]> : tensor<2xi32>

  %0 = stablehlo.subtract %c1, %c0 : tensor<i32>
  %1 = stablehlo.subtract %c1, %c4 : tensor<i32>

  %2 = stablehlo.subtract %arg1, %cp0 : tensor<f32>
  %3 = stablehlo.subtract %arg1, %arg1 : tensor<f32>

  %4 = stablehlo.subtract %arg0, %arg0 : tensor<2xi32>

  %5 = stablehlo.subtract %c3, %c5 : tensor<2xi32>

  // CHECK-DAG:  [[C0:%.+]] = stablehlo.constant dense<5> : tensor<i32>
  // CHECK-DAG:  [[C1:%.+]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK-DAG:  [[C2:%.+]] = stablehlo.constant dense<0> : tensor<2xi32>
  // CHECK-DAG:  [[C3:%.+]] = stablehlo.constant dense<1> : tensor<2xi32>

  // CHECK-DAG:  [[S0:%.+]] = stablehlo.subtract [[ARG1]], [[ARG1]] : tensor<f32>

  // CHECK-NEXT: return [[C0]], [[C1]], [[ARG1]], [[S0]], [[C2]], [[C3]]
  return %0, %1, %2, %3, %4, %5 : tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>
}

// -----

// CHECK-LABEL: func.func @multiply
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<2xi32>, [[ARG1:%.+]]: tensor<f32>)
func.func @multiply(%arg0: tensor<2xi32>, %arg1: tensor<f32>)
  -> (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %cp0 = stablehlo.constant dense<0.0> : tensor<f32>
  %c0_2 = stablehlo.constant dense<0> : tensor<2xi32>
  %c1 = stablehlo.constant dense<5> : tensor<i32>
  %c2 = stablehlo.constant dense<3.0> : tensor<f32>
  %c3 = stablehlo.constant dense<[1, 2]> : tensor<2xi32>
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %c5 = stablehlo.constant dense<1> : tensor<2xi32>

  %0 = stablehlo.multiply %c1, %c0 : tensor<i32>
  %1 = stablehlo.multiply %c4, %c4 : tensor<i32>

  %2 = stablehlo.multiply %arg1, %cp0 : tensor<f32>
  %3 = stablehlo.multiply %c2, %c2 : tensor<f32>

  %4 = stablehlo.multiply %arg0, %c0_2 : tensor<2xi32>
  %5 = stablehlo.multiply %arg0, %c5 : tensor<2xi32>
  %6 = stablehlo.multiply %c3, %arg0 : tensor<2xi32>

  // CHECK-DAG:  [[C0:%.+]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-DAG:  [[C1:%.+]] = stablehlo.constant dense<16> : tensor<i32>
  // CHECK-DAG:  [[C2:%.+]] = stablehlo.constant dense<9.000000e+00> : tensor<f32>
  // CHECK-DAG:  [[C3:%.+]] = stablehlo.constant dense<0> : tensor<2xi32>
  // CHECK-DAG:  [[C4:%.+]] = stablehlo.constant dense<[1, 2]> : tensor<2xi32>
  // CHECK-DAG:  [[CP0:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>

  // CHECK-DAG:  [[M0:%.+]] = stablehlo.multiply [[ARG1]], [[CP0]] : tensor<f32>
  // CHECK-DAG:  [[M1:%.+]] = stablehlo.multiply [[ARG0]], [[C4]] : tensor<2xi32>

  // CHECK-NEXT: return [[C0]], [[C1]], [[M0]], [[C2]], [[C3]], [[ARG0]], [[M1]]
  return %0, %1, %2, %3, %4, %5, %6 : tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>
}

// -----

// CHECK-LABEL: func.func @compare_signed_arg
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<i32>)
func.func @compare_signed_arg(%arg0: tensor<i32>)
  -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %c5 = stablehlo.constant dense<5> : tensor<i32>

  %0 = stablehlo.compare EQ, %arg0, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = stablehlo.compare GT, %arg0, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = stablehlo.compare LE, %arg0, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %3 = stablehlo.compare NE, %arg0, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  %4 = stablehlo.compare EQ, %c5, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %5 = stablehlo.compare LT, %c5, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %6 = stablehlo.compare GE, %c5, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %7 = stablehlo.compare NE, %c5, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  // CHECK-DAG:  [[C0:%.+]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-DAG:  [[C1:%.+]] = stablehlo.constant dense<true> : tensor<i1>
  // CHECK-DAG:  [[C5:%.+]] = stablehlo.constant dense<5> : tensor<i32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.compare EQ, [[ARG0]], [[C5]], SIGNED
  // CHECK-DAG:  [[R1:%.+]] = stablehlo.compare GT, [[ARG0]], [[C5]], SIGNED
  // CHECK-DAG:  [[R2:%.+]] = stablehlo.compare LE, [[ARG0]], [[C5]], SIGNED
  // CHECK-DAG:  [[R3:%.+]] = stablehlo.compare NE, [[ARG0]], [[C5]], SIGNED

  // CHECK-NEXT: return [[C1]], [[C0]], [[C1]], [[C0]], [[R0]], [[R1]], [[R2]], [[R3]]
  return %0, %1, %2, %3, %4, %5, %6, %7 :
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>
}

// -----

// CHECK-LABEL: func.func @compare_unsigned_arg
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<i32>)
func.func @compare_unsigned_arg(%arg0: tensor<i32>)
  -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %c5 = stablehlo.constant dense<5> : tensor<i32>

  %0 = stablehlo.compare EQ, %arg0, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = stablehlo.compare GT, %arg0, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = stablehlo.compare LE, %arg0, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %3 = stablehlo.compare NE, %arg0, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  %4 = stablehlo.compare EQ, %c5, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %5 = stablehlo.compare LT, %c5, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %6 = stablehlo.compare GE, %c5, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %7 = stablehlo.compare NE, %c5, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  // CHECK-DAG:  [[C0:%.+]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-DAG:  [[C1:%.+]] = stablehlo.constant dense<true> : tensor<i1>
  // CHECK-DAG:  [[C5:%.+]] = stablehlo.constant dense<5> : tensor<i32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.compare EQ, [[ARG0]], [[C5]], UNSIGNED
  // CHECK-DAG:  [[R1:%.+]] = stablehlo.compare GT, [[ARG0]], [[C5]], UNSIGNED
  // CHECK-DAG:  [[R2:%.+]] = stablehlo.compare LE, [[ARG0]], [[C5]], UNSIGNED
  // CHECK-DAG:  [[R3:%.+]] = stablehlo.compare NE, [[ARG0]], [[C5]], UNSIGNED

  // CHECK-NEXT: return [[C1]], [[C0]], [[C1]], [[C0]], [[R0]], [[R1]], [[R2]], [[R3]]
  return %0, %1, %2, %3, %4, %5, %6, %7 :
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>
}

// -----

// CHECK-LABEL: func.func @compare_folds
func.func @compare_folds()
  -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>) {
  %cn1 = stablehlo.constant dense<-1> : tensor<i32>
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %c5 = stablehlo.constant dense<5> : tensor<i32>

  %0 = stablehlo.compare EQ, %cn1, %cn1, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = stablehlo.compare GT, %c5, %c5, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = stablehlo.compare GE, %c4, %cn1, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %3 = stablehlo.compare LE, %c4, %c5, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  %4 = stablehlo.compare EQ, %cn1, %cn1, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %5 = stablehlo.compare GT, %c5, %cn1, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %6 = stablehlo.compare GE, %c5, %c4, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %7 = stablehlo.compare LE, %cn1, %c5, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  // CHECK-DAG:  [[C0:%.+]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-DAG:  [[C1:%.+]] = stablehlo.constant dense<true> : tensor<i1>

  // CHECK-NEXT: return [[C1]], [[C0]], [[C1]], [[C1]], [[C1]], [[C0]], [[C1]], [[C0]]
  return %0, %1, %2, %3, %4, %5, %6, %7 :
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>
}

// -----

// CHECK-LABEL: func.func @broadcast_in_dim
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<3x3xi32>)
func.func @broadcast_in_dim(%arg0: tensor<3x3xi32>)
  -> (tensor<6xi32>, tensor<3xf32>, tensor<3x3xi32>, tensor<3x3xi32>, tensor<3x3xi32>, tensor<3x3x1xi32>, tensor<3x2x3x3xi32>) {
  %c0 = stablehlo.constant dense<5> : tensor<i32>
  %c1 = stablehlo.constant dense<3.0> : tensor<f32>
  %c2 = stablehlo.constant dense<1> : tensor<1x3xi32>

  %0 = stablehlo.broadcast_in_dim %c0, dims = [] : (tensor<i32>) -> tensor<6xi32>
  %1 = stablehlo.broadcast_in_dim %c1, dims = [] : (tensor<f32>) -> tensor<3xf32>
  %2 = stablehlo.broadcast_in_dim %c2, dims = [1, 0] : (tensor<1x3xi32>) -> tensor<3x3xi32>

  %3 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<3x3xi32>) -> tensor<3x3xi32>
  %4 = stablehlo.broadcast_in_dim %arg0, dims = [1, 0] : (tensor<3x3xi32>) -> tensor<3x3xi32>
  %5 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<3x3xi32>) -> tensor<3x3x1xi32>

  %6 = stablehlo.broadcast_in_dim %arg0, dims = [1, 0] : (tensor<3x3xi32>) -> tensor<3x3x2xi32>
  %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2, 1] : (tensor<3x3x2xi32>) -> tensor<3x2x3x3xi32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.constant dense<5> : tensor<6xi32>
  // CHECK-DAG:  [[R1:%.+]] = stablehlo.constant dense<3.000000e+00> : tensor<3xf32>
  // CHECK-DAG:  [[R2:%.+]] = stablehlo.constant dense<1> : tensor<3x3xi32>

  // CHECK-DAG:  [[R4:%.+]] = stablehlo.transpose [[ARG0]], dims = [1, 0] : (tensor<3x3xi32>) -> tensor<3x3xi32>
  // CHECK-DAG:  [[R5:%.+]] = stablehlo.reshape [[ARG0]] : (tensor<3x3xi32>) -> tensor<3x3x1xi32>
  // CHECK-DAG:  [[R6:%.+]] = stablehlo.broadcast_in_dim [[ARG0]], dims = [2, 0] : (tensor<3x3xi32>) -> tensor<3x2x3x3xi32>

  // CHECK-NEXT: return [[R0]], [[R1]], [[R2]], [[ARG0]], [[R4]], [[R5]], [[R6]]
  return %0, %1, %2, %3, %4, %5, %7 : tensor<6xi32>, tensor<3xf32>, tensor<3x3xi32>, tensor<3x3xi32>, tensor<3x3xi32>, tensor<3x3x1xi32>, tensor<3x2x3x3xi32>
}

// -----

// CHECK-LABEL: func.func @concatenate
func.func @concatenate() -> (tensor<6xi32>, tensor<3xi32>, tensor<3x3xi32>, tensor<2x5xi32>) {
  %c0 = stablehlo.constant dense<[0, 1]> : tensor<2xi32>
  %c1 = stablehlo.constant dense<[2, 3, 4]> : tensor<3xi32>
  %c2 = stablehlo.constant dense<[5]> : tensor<1xi32>

  %c3 = stablehlo.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>
  %c4 = stablehlo.constant dense<[[6, 7, 8]]> : tensor<1x3xi32>
  %c5 = stablehlo.constant dense<[[11, 12], [13, 14]]> : tensor<2x2xi32>

  %0 = stablehlo.concatenate %c0, %c1, %c2, dim = 0 : (tensor<2xi32>, tensor<3xi32>, tensor<1xi32>) -> tensor<6xi32>
  %1 = stablehlo.concatenate %c0, %c2, dim = 0 : (tensor<2xi32>, tensor<1xi32>) -> tensor<3xi32>

  %2 = stablehlo.concatenate %c3, %c4, dim = 0 : (tensor<2x3xi32>, tensor<1x3xi32>) -> tensor<3x3xi32>
  %3 = stablehlo.concatenate %c3, %c5, dim = 1 : (tensor<2x3xi32>, tensor<2x2xi32>) -> tensor<2x5xi32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.constant dense<[0, 1, 2, 3, 4, 5]> : tensor<6xi32>
  // CHECK-DAG:  [[R1:%.+]] = stablehlo.constant dense<[0, 1, 5]> : tensor<3xi32>
  // CHECK-DAG:  [[R2:%.+]] = stablehlo.constant dense<{{\[\[0, 1, 2\], \[3, 4, 5\], \[6, 7, 8\]\]}}> : tensor<3x3xi32>
  // CHECK-DAG:  [[R3:%.+]] = stablehlo.constant dense<{{\[\[0, 1, 2, 11, 12\], \[3, 4, 5, 13, 14\]\]}}> : tensor<2x5xi32>
  // CHECK-NEXT: return [[R0]], [[R1]], [[R2]], [[R3]]
  return %0, %1, %2, %3 : tensor<6xi32>, tensor<3xi32>, tensor<3x3xi32>, tensor<2x5xi32>
}

// -----

// CHECK-LABEL: func.func @convert
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<2xf32>)
func.func @convert(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %r = stablehlo.convert %arg0 : tensor<2xf32>

  // CHECK: return [[ARG0]]
  return %r : tensor<2xf32>
}

// -----

// CHECK-LABEL: func.func @complex
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<2xf32>, [[ARG1:%.+]]: tensor<2xf32>)
func.func @complex(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %c = stablehlo.complex %arg0, %arg1 : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  %r = stablehlo.real %c : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %i = stablehlo.imag %c : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return [[ARG0]], [[ARG1]]
  return %r, %i : tensor<2xf32>, tensor<2xf32>
}

// -----

// CHECK-LABEL: func.func @dynamic_reshape
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1xf32>, [[ARG1:%.+]]: tensor<?x?xf32>, [[ARG2:%.+]]: tensor<2xi32>)
func.func @dynamic_reshape(%arg0: tensor<1xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<2xi32>)
          -> (tensor<1x1xf32>, tensor<2x1xf32>, tensor<1x2xi32>) {
  %c0 = stablehlo.constant dense<[2, 1]> : tensor<2xi32>

  %0 = stablehlo.dynamic_reshape %arg0, %arg2 : (tensor<1xf32>, tensor<2xi32>) -> tensor<1x1xf32>
  %1 = stablehlo.dynamic_reshape %arg1, %c0 : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<2x1xf32>
  %2 = stablehlo.dynamic_reshape %arg2, %arg2 : (tensor<2xi32>, tensor<2xi32>) -> tensor<1x2xi32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.reshape [[ARG0]] : (tensor<1xf32>) -> tensor<1x1xf32>
  // CHECK-DAG:  [[R1:%.+]] = stablehlo.reshape [[ARG1]] : (tensor<?x?xf32>) -> tensor<2x1xf32>
  // CHECK-DAG:  [[R2:%.+]] = stablehlo.reshape [[ARG2]] : (tensor<2xi32>) -> tensor<1x2xi32>
  // CHECK-NEXT: return [[R0]], [[R1]], [[R2]]
  return %0, %1, %2 : tensor<1x1xf32>, tensor<2x1xf32>, tensor<1x2xi32>
}

// -----

// CHECK-LABEL: func.func @get_dimension_size
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x2x3xf32>, [[ARG1:%.+]]: tensor<?x2xf32>)
func.func @get_dimension_size(%arg0: tensor<1x2x3xf32>, %arg1: tensor<?x2xf32>)
          -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) {
  %a = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<1x2x3xf32>) -> tensor<i32>
  %b = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<1x2x3xf32>) -> tensor<i32>
  %c = stablehlo.get_dimension_size %arg0, dim = 2 : (tensor<1x2x3xf32>) -> tensor<i32>

  %d = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x2xf32>) -> tensor<i32>
  %e = stablehlo.get_dimension_size %arg1, dim = 1 : (tensor<?x2xf32>) -> tensor<i32>

  // CHECK-DAG:  [[CST1:%.+]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK-DAG:  [[CST2:%.+]] = stablehlo.constant dense<2> : tensor<i32>
  // CHECK-DAG:  [[CST3:%.+]] = stablehlo.constant dense<3> : tensor<i32>
  // CHECK-DAG:  [[DYN:%.+]]  = stablehlo.get_dimension_size [[ARG1]], dim = 0 : (tensor<?x2xf32>) -> tensor<i32>
  // CHECK-NEXT: return [[CST1]], [[CST2]], [[CST3]], [[DYN]], [[CST2]]
  return %a, %b, %c, %d, %e : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
}

// -----

// CHECK-LABEL: func.func @get_tuple_element
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<f32>, [[ARG1:%.+]]: tensor<i32>, [[ARG2:%.+]]: tuple<tensor<f32>, tensor<f16>>)
func.func @get_tuple_element(%arg0: tensor<f32>, %arg1: tensor<i32>, %arg2: tuple<tensor<f32>, tensor<f16>>)
          -> (tensor<f32>, tensor<i32>, tensor<f16>) {
  %t = stablehlo.tuple %arg0, %arg1 : tuple<tensor<f32>, tensor<i32>>

  %a = stablehlo.get_tuple_element %t[0] : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  %b = stablehlo.get_tuple_element %t[1] : (tuple<tensor<f32>, tensor<i32>>) -> tensor<i32>

  %c = stablehlo.get_tuple_element %arg2[1] : (tuple<tensor<f32>, tensor<f16>>) -> tensor<f16>

  // CHECK:      [[GTE:%.+]] = stablehlo.get_tuple_element [[ARG2]][1] : (tuple<tensor<f32>, tensor<f16>>) -> tensor<f16>
  // CHECK-NEXT: return [[ARG0]], [[ARG1]], [[GTE]]
  return %a, %b, %c : tensor<f32>, tensor<i32>, tensor<f16>
}

// -----

// CHECK-LABEL: func.func @reshape
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1xf32>)
func.func @reshape(%arg0: tensor<1xf32>)
          -> (tensor<1xf32>, tensor<1xi32>, tensor<i32>, tensor<2x2xi32>) {
  %c0 = stablehlo.constant dense<2> : tensor<i32>
  %c1 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

  %0 = stablehlo.reshape %arg0 : (tensor<1xf32>) -> tensor<1xf32>
  %1 = stablehlo.reshape %c0 : (tensor<i32>) -> tensor<1xi32>
  %2 = stablehlo.reshape %1 : (tensor<1xi32>) -> tensor<i32>
  %3 = stablehlo.reshape %c1 : (tensor<4xi32>) -> tensor<2x2xi32>

  // CHECK-DAG:  [[CST1:%.+]] = stablehlo.constant dense<2> : tensor<i32>
  // CHECK-DAG:  [[CST2:%.+]] = stablehlo.constant dense<2> : tensor<1xi32>
  // CHECK-DAG:  [[CST3:%.+]] = stablehlo.constant dense<{{\[\[1, 2\], \[3, 4\]\]}}> : tensor<2x2xi32>
  // CHECK-NEXT: return [[ARG0]], [[CST2]], [[CST1]], [[CST3]]
  return %0, %1, %2, %3 : tensor<1xf32>, tensor<1xi32>, tensor<i32>, tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func.func @transpose
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<2xf32>, [[ARG1:%.+]]: tensor<1x2xf32>, [[ARG2:%.+]]: tensor<f32>)
func.func @transpose(%arg0: tensor<2xf32>, %arg1: tensor<1x2xf32>, %arg2: tensor<f32>)
          -> (tensor<2xf32>, tensor<1x2xf32>, tensor<2x1xf32>, tensor<f32>) {
  %a = stablehlo.transpose %arg0, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
  %b = stablehlo.transpose %arg1, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %c = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
  %d = stablehlo.transpose %arg2, dims = [] : (tensor<f32>) -> tensor<f32>

  // CHECK-NEXT: [[X:%.+]] = stablehlo.transpose [[ARG1]], dims = [1, 0]
  // CHECK-NEXT: return [[ARG0]], [[ARG1]], [[X]], [[ARG2]]
  return %a, %b, %c, %d : tensor<2xf32>, tensor<1x2xf32>, tensor<2x1xf32>, tensor<f32>
}
