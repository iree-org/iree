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

// CHECK-LABEL: func.func @select
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<2xi32>, [[ARG1:%.+]]: tensor<2xi32>, [[ARGC:%.+]]: tensor<2xi1>)
func.func @select(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>, %argC: tensor<2xi1>)
  -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<4xi32>) {
  %c0 = stablehlo.constant dense<false> : tensor<i1>
  %c1 = stablehlo.constant dense<true> : tensor<i1>

  %c0x2 = stablehlo.constant dense<false> : tensor<2xi1>
  %c1x2 = stablehlo.constant dense<true> : tensor<2xi1>

  %cond = stablehlo.constant dense<[false, true, false, true]> : tensor<4xi1>
  %foo = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %bar = stablehlo.constant dense<[5, 6, 7, 8]> : tensor<4xi32>

  %0 = stablehlo.select %argC, %arg0, %arg0 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %1 = stablehlo.select %c0, %arg0, %arg1 : (tensor<i1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %2 = stablehlo.select %c1, %arg0, %arg1 : (tensor<i1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %3 = stablehlo.select %c0x2, %arg0, %arg1 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %4 = stablehlo.select %c1x2, %arg0, %arg1 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %5 = stablehlo.select %argC, %arg0, %arg1 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>

  %6 = stablehlo.select %cond, %foo, %bar : (tensor<4xi1>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.select [[ARGC]], [[ARG0]], [[ARG1]]
  // CHECK-DAG:  [[C0:%.+]] = stablehlo.constant dense<[5, 2, 7, 4]> : tensor<4xi32>

  // CHECK-NEXT: return [[ARG0]], [[ARG1]], [[ARG0]], [[ARG1]], [[ARG0]], [[R0]], [[C0]]
  return %0, %1, %2, %3, %4, %5, %6 :
         tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<4xi32>
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

// CHECK-LABEL: @merge_consecutive_reshapes
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
func.func @merge_consecutive_reshapes(%arg0: tensor<4x4xi32>) -> tensor<16xi32> {
  %0 = stablehlo.reshape %arg0 : (tensor<4x4xi32>) -> tensor<2x8xi32>
  %1 = stablehlo.reshape %0 : (tensor<2x8xi32>) -> tensor<16xi32>
  // CHECK: [[R0:%.+]] = stablehlo.reshape %[[ARG0]] : (tensor<4x4xi32>) -> tensor<16xi32>
  return %1 : tensor<16xi32>
}

// -----

// CHECK-LABEL: func.func @transpose
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<2xf32>, [[ARG1:%.+]]: tensor<3x2xf32>, [[ARG2:%.+]]: tensor<f32>)
func.func @transpose(%arg0: tensor<2xf32>, %arg1: tensor<3x2xf32>, %arg2: tensor<f32>)
          -> (tensor<2xf32>, tensor<3x2xf32>, tensor<2x3xf32>, tensor<f32>) {
  %a = stablehlo.transpose %arg0, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
  %b = stablehlo.transpose %arg1, dims = [0, 1] : (tensor<3x2xf32>) -> tensor<3x2xf32>
  %c = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
  %d = stablehlo.transpose %arg2, dims = [] : (tensor<f32>) -> tensor<f32>

  // CHECK-NEXT: [[X:%.+]] = stablehlo.transpose [[ARG1]], dims = [1, 0]
  // CHECK-NEXT: return [[ARG0]], [[ARG1]], [[X]], [[ARG2]]
  return %a, %b, %c, %d : tensor<2xf32>, tensor<3x2xf32>, tensor<2x3xf32>, tensor<f32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic(%arg0: tensor<4xf32>, %arg1: tensor<2xi64>) -> tensor<5x4xf32> {
  // CHECK: %[[RESULT:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<4xf32>) -> tensor<5x4xf32>
  %0 = stablehlo.dynamic_broadcast_in_dim %arg0, %arg1, dims = [1] : (tensor<4xf32>, tensor<2xi64>) -> tensor<5x4xf32>
  // CHECK: return %[[RESULT]] : tensor<5x4xf32>
  func.return %0 : tensor<5x4xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_shape
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_shape(%arg0: tensor<i32>) -> tensor<4x32xi32> {
  %0 = stablehlo.constant dense<[4, 32]> : tensor<2xi32>
  // CHECK: %[[RESULT:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<i32>) -> tensor<4x32xi32>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [] : (tensor<i32>, tensor<2xi32>) -> tensor<?x32xi32>
  %2 = stablehlo.dynamic_reshape %1, %0 : (tensor<?x32xi32>, tensor<2xi32>) -> tensor<4x32xi32>
  // CHECK: return %[[RESULT]] : tensor<4x32xi32>
  func.return %2 : tensor<4x32xi32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_index_shape
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_index_shape(%arg0: tensor<f32>) -> tensor<4x32xf32> {
  %0 = shape.const_shape [4, 32] : tensor<2xindex>
  // CHECK: %[[RESULT:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<4x32xf32>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x32xf32>
  %2 = stablehlo.dynamic_reshape %1, %0 : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<4x32xf32>
  // CHECK: return %[[RESULT]] : tensor<4x32xf32>
  func.return %2 : tensor<4x32xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_requires_cast
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_requires_cast(%arg0: tensor<f32>) -> tensor<?x?xf32> {
  %0 = shape.const_shape [4, 32] : tensor<2xindex>
  // CHECK: %[[BCAST:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<4x32xf32>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK: %[[RESULT:.*]] = tensor.cast %[[BCAST]] : tensor<4x32xf32> to tensor<?x?xf32>
  // CHECK: return %[[RESULT]] : tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_almost_not_actually_dynamic
func.func @dynamic_broadcast_in_dim_op_almost_not_actually_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<2xi64>) -> tensor<5x4xf32> {
  // CHECK: %[[RESULT:.+]] = stablehlo.dynamic_broadcast_in_dim %arg0, %arg1, dims = [1] : (tensor<?xf32>, tensor<2xi64>) -> tensor<5x4xf32>
  %0 = stablehlo.dynamic_broadcast_in_dim %arg0, %arg1, dims = [1] : (tensor<?xf32>, tensor<2xi64>) -> tensor<5x4xf32>
  // CHECK: return %[[RESULT]] : tensor<5x4xf32>
  func.return %0 : tensor<5x4xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_all_dims_non_expanding
func.func @dynamic_broadcast_in_dim_all_dims_non_expanding(%arg0: tensor<*xf32>, %arg1: tensor<1xindex>) -> tensor<?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<*xf32>
  %1 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {
    broadcast_dimensions = dense<0> : tensor<1xi64>,
    known_expanding_dimensions = dense<> : tensor<0xi64>,
    known_nonexpanding_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: %[[RES:.*]] = tensor.cast %[[ARG]] : tensor<*xf32> to tensor<?xf32>
  // CHECK: return %[[RES]] : tensor<?xf32>
  func.return %1 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func.func @gather_to_slice
func.func @gather_to_slice(%arg0: tensor<5x6x7xf32>) -> tensor<3x6x5xf32> {
  %0 = arith.constant dense<[1, 2]> : tensor<2xi32>
  %1 = "stablehlo.gather"(%arg0, %0) {
    dimension_numbers = #stablehlo.gather<
      index_vector_dim = 0,
      offset_dims = [0, 1, 2],
      start_index_map = [0, 2],
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[3, 6, 5]> : tensor<3xi64>} : (tensor<5x6x7xf32>, tensor<2xi32>) -> tensor<3x6x5xf32>
  return %1 : tensor<3x6x5xf32>
  // CHECK:      %[[RET:.*]] = stablehlo.slice %arg0 [1:4, 0:6, 2:7]
  // CHECK-SAME:    : (tensor<5x6x7xf32>) -> tensor<3x6x5xf32>
  // CHECK-NEXT: return %[[RET]] : tensor<3x6x5xf32>
}

// -----

// CHECK-LABEL: func.func @gather_scalar_index_to_slice
func.func @gather_scalar_index_to_slice(%arg0: tensor<5x6x7xf32>) -> tensor<5x6x4xf32> {
  %0 = arith.constant dense<1> : tensor<i32>
  %1 = "stablehlo.gather"(%arg0, %0) {
    dimension_numbers = #stablehlo.gather<
      index_vector_dim = 0,
      offset_dims = [0, 1, 2],
      start_index_map = [2],
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[5, 6, 4]> : tensor<3xi64>} : (tensor<5x6x7xf32>, tensor<i32>) -> tensor<5x6x4xf32>
  return %1 : tensor<5x6x4xf32>
  // CHECK:      %[[RET:.*]] = stablehlo.slice %arg0 [0:5, 0:6, 1:5]
  // CHECK-SAME:    : (tensor<5x6x7xf32>) -> tensor<5x6x4xf32>
  // CHECK-NEXT: return %[[RET]] : tensor<5x6x4xf32>
}

// -----

// CHECK-LABEL: func.func @gather_to_slice_reshape
func.func @gather_to_slice_reshape(%arg0: tensor<5x6x7xf32>) -> tensor<3x6xf32> {
  %0 = arith.constant dense<[1, 2]> : tensor<2xi32>
  %1 = "stablehlo.gather"(%arg0, %0) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [2],
      index_vector_dim = 0,
      offset_dims = [0, 1],
      start_index_map = [0, 2],
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[3, 6, 1]> : tensor<3xi64>} : (tensor<5x6x7xf32>, tensor<2xi32>) -> tensor<3x6xf32>
  return %1 : tensor<3x6xf32>
  // CHECK:      %[[V0:.*]] = stablehlo.slice %arg0 [1:4, 0:6, 2:3]
  // CHECK-SAME:    : (tensor<5x6x7xf32>) -> tensor<3x6x1xf32>
  // CHECK-NEXT: %[[V1:.*]] = stablehlo.reshape %[[V0]] : (tensor<3x6x1xf32>) -> tensor<3x6xf32>
  // CHECK-NEXT: return %[[V1]] : tensor<3x6xf32>
}

// -----

// CHECK-LABEL: func.func @gather_to_slice_indices_clamp_upperbound
func.func @gather_to_slice_indices_clamp_upperbound(%arg0 : tensor<4x2xui32>) -> tensor<2xui32> {
  %0 = arith.constant dense<4> : tensor<1xi32>
  %1 = "stablehlo.gather"(%arg0, %0) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [0],
      index_vector_dim = 0,
      collapsed_slice_dims = [0],
      start_index_map = [0]
    >, indices_are_sorted = true,
    slice_sizes = dense<[1, 2]> : tensor<2xi64>} : (tensor<4x2xui32>, tensor<1xi32>) -> tensor<2xui32>
  return %1 : tensor<2xui32>
  // CHECK:      %[[V0:.*]] = stablehlo.slice %arg0 [3:4, 0:2]
  // CHECK-SAME:    : (tensor<4x2xui32>) -> tensor<1x2xui32>
  // CHECK-NEXT: %[[V1:.*]] = stablehlo.reshape %[[V0]] : (tensor<1x2xui32>) -> tensor<2xui32>
  // CHECK-NEXT: return %[[V1]] : tensor<2xui32>
}

// -----

// CHECK-LABEL: func.func @gather_to_slice_indices_clamp_lowerbound
func.func @gather_to_slice_indices_clamp_lowerbound(%arg0 : tensor<4x2xui32>) -> tensor<2xui32> {
  %0 = arith.constant dense<-1> : tensor<1xi32>
  %1 = "stablehlo.gather"(%arg0, %0) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [0],
      index_vector_dim = 0,
      collapsed_slice_dims = [0],
      start_index_map = [0]
    >, indices_are_sorted = true,
    slice_sizes = dense<[1, 2]> : tensor<2xi64>} : (tensor<4x2xui32>, tensor<1xi32>) -> tensor<2xui32>
  return %1 : tensor<2xui32>
  // CHECK:      %[[V0:.*]] = stablehlo.slice %arg0 [0:1, 0:2]
  // CHECK-SAME:    : (tensor<4x2xui32>) -> tensor<1x2xui32>
  // CHECK-NEXT: %[[V1:.*]] = stablehlo.reshape %[[V0]] : (tensor<1x2xui32>) -> tensor<2xui32>
  // CHECK-NEXT: return %[[V1]] : tensor<2xui32>
}

// -----

// CHECK-LABEL: @transpose_is_reshape
func.func @transpose_is_reshape(%arg0: tensor<1x4x5x1xf32>) -> tensor<1x4x1x5xf32> {
  // CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %arg0 : (tensor<1x4x5x1xf32>) -> tensor<1x4x1x5xf32>
  %0 = stablehlo.transpose %arg0, dims = [3, 1, 0, 2] : (tensor<1x4x5x1xf32>) -> tensor<1x4x1x5xf32>
  return %0 : tensor<1x4x1x5xf32>
}

// CHECK-LABEL: @transpose_is_not_reshape
func.func @transpose_is_not_reshape(%arg0: tensor<1x4x5x2xf32>) -> tensor<2x4x1x5xf32> {
  // CHECK-NOT: stablehlo.reshape
  %0 = stablehlo.transpose %arg0, dims = [3, 1, 0, 2] : (tensor<1x4x5x2xf32>) -> tensor<2x4x1x5xf32>
  return %0 : tensor<2x4x1x5xf32>
}

// -----

// CHECK-LABEL: func.func @reduce_noop_1
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<4x8xf32>)
func.func @reduce_noop_1(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) across dimensions = [] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
    reducer(%arg1: tensor<f32>, %arg2: tensor<f32>) {
    %4 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %4 : tensor<f32>
  }
  // CHECK: return [[ARG0]] : tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @reduce_noop_2
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<4x8xi32>, [[ARG1:%.+]]: tensor<i32>)
func.func @reduce_noop_2(%arg0: tensor<4x8xi32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.reduce(%arg0 init: %0) across dimensions = [0, 1] : (tensor<4x8xi32>, tensor<i32>) -> tensor<i32>
    reducer(%b1: tensor<i32>, %b2: tensor<i32>) {
    stablehlo.return %arg1 : tensor<i32>
  }
  // CHECK: return [[ARG1]] : tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: func.func @reduce_zero_ext
func.func @reduce_zero_ext(%arg0: tensor<0xi1>) -> tensor<i32> {
  %0 = stablehlo.constant dense<false> : tensor<i1>
  %1 = stablehlo.constant dense<false> : tensor<0xi1>
  %2 = stablehlo.compare  NE, %arg0, %1, UNSIGNED : (tensor<0xi1>, tensor<0xi1>) -> tensor<0xi1>
  %3 = stablehlo.convert %2 : (tensor<0xi1>) -> tensor<0xi32>
  %4 = stablehlo.constant dense<0> : tensor<i32>
  %5 = stablehlo.reduce(%3 init: %4) across dimensions = [0] : (tensor<0xi32>, tensor<i32>) -> tensor<i32>
    reducer(%arg1: tensor<i32>, %arg2: tensor<i32>)  {
    %6 = stablehlo.add %arg1, %arg2 : tensor<i32>
    stablehlo.return %6 : tensor<i32>
  }

  // CHECK: [[CST:%.+]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: return [[CST]] : tensor<i32>
  return %5 : tensor<i32>
}

// -----

// CHECK-LABEL: func.func @add_zero_ext
func.func @add_zero_ext(%arg0 : tensor<5x0xi32>, %arg1 : tensor<5x0xi32>) -> tensor<5x0xi32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<5x0xi32>
  func.return %0 : tensor<5x0xi32>
}
// CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<5x0xi32>
// CHECK:   return %[[EMPTY]]

// -----

// CHECK-LABEL: func.func @add_zero_ext_dynamic
func.func @add_zero_ext_dynamic(%arg0 : tensor<?x0xi32>, %arg1 : tensor<?x0xi32>) -> tensor<?x0xi32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<?x0xi32>
  func.return %0 : tensor<?x0xi32>
}
// CHECK-NOT:   tensor.empty()

// -----

// CHECK-LABEL: func.func @scatter_zero_ext
func.func @scatter_zero_ext(%arg0 : tensor<f32>, %arg1 : tensor<1x0xi32>, %arg2 : tensor<1xf32>) -> tensor<f32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [],
      scatter_dims_to_operand_dims = [],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<f32>, tensor<1x0xi32>, tensor<1xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<1x0xi32>
// CHECK:   %[[SCATTER:.+]] = "stablehlo.scatter"(%arg0, %0, %arg2)
// CHECK:   return %[[SCATTER]]

// -----

  func.func public @sort_zero_extent(%arg0: tensor<0xi16> {jax.arg_info = "a", mhlo.sharding = "{replicated}"}) -> (tensor<0xi32> {jax.result_info = ""}) {
    %0 = stablehlo.iota dim = 0 : tensor<0xi32>
    %1:2 = "stablehlo.sort"(%arg0, %0) ({
    ^bb0(%arg1: tensor<i16>, %arg2: tensor<i16>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = stablehlo.compare  LT, %arg1, %arg2,  SIGNED : (tensor<i16>, tensor<i16>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<0xi16>, tensor<0xi32>) -> (tensor<0xi16>, tensor<0xi32>)
    return %1#1 : tensor<0xi32>
  }

// CHECK-LABEL: @sort_zero_extent
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<0xi32>
// CHECK: return %[[EMPTY]]

// -----

// CHECK-LABEL: @while_zero_extent
// CHECK: %[[R0:.+]] = tensor.empty() : tensor<75x0xf32>
// CHECK: %[[R1:.+]] = tensor.empty() : tensor<75x0xf32>
// CHECK: %[[R2:.+]]:2 = stablehlo.while
// CHECK: return %[[R2]]#0, %[[R0]]


func.func public @while_zero_extent(%arg0: tensor<i32>, %arg1: tensor<3xf32>, %arg2: tensor<75x0xf32>) -> (tensor<i32>, tensor<75x0xf32>) {
  %0 = stablehlo.constant dense<1> : tensor<i32>
  %1 = stablehlo.constant dense<75> : tensor<i32>
  %2 = stablehlo.constant dense<0> : tensor<i32>
  %3:2 = stablehlo.while(%iterArg = %2, %iterArg_2 = %arg2) : tensor<i32>, tensor<75x0xf32>
   cond {
    %4 = stablehlo.compare  LT, %iterArg, %1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %44 = stablehlo.add %iterArg, %0 : tensor<i32>
    stablehlo.return %44, %iterArg_2 : tensor<i32>, tensor<75x0xf32>
  }
  return %3#0, %3#1 : tensor<i32>, tensor<75x0xf32>
}

// -----

func.func @push_shape_ops_to_end(%arg0 : tensor<12xf32>) -> tensor<3x4x2x1xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<12xf32>) -> tensor<3x4xf32>
  %1 = stablehlo.broadcast %0, sizes = [1, 2] : (tensor<3x4xf32>) -> tensor<1x2x3x4xf32>
  %2 = stablehlo.cosine %1 : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  %3 = stablehlo.transpose %2, dims = [2, 3, 1, 0]  : (tensor<1x2x3x4xf32>) -> tensor<3x4x2x1xf32>
  %4 = stablehlo.abs %3 : (tensor<3x4x2x1xf32>) -> tensor<3x4x2x1xf32>
  return %4 : tensor<3x4x2x1xf32>
}

// CHECK-LABEL: @push_shape_ops_to_end
// CHECK: %[[COS:.+]] = stablehlo.cosine %arg0 : tensor<12xf32>
// CHECK: %[[ABS:.+]] = stablehlo.abs %[[COS]] : tensor<12xf32>
// CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %[[ABS]] : (tensor<12xf32>) -> tensor<3x4xf32>
// CHECK: %[[BROADCAST:.+]] = stablehlo.broadcast %[[RESHAPE]], sizes = [1, 2] : (tensor<3x4xf32>) -> tensor<1x2x3x4xf32>
// CHECK: %[[TRANSPOSE:.+]] = stablehlo.transpose %[[BROADCAST]], dims = [2, 3, 1, 0] : (tensor<1x2x3x4xf32>) -> tensor<3x4x2x1xf32>
// CHECK: return %[[TRANSPOSE]]

// -----

func.func @reorder_with_type_change(%arg0 : tensor<3x4xi32>) -> tensor<12xi64> {
  %0 = stablehlo.reshape %arg0 : (tensor<3x4xi32>) -> tensor<12xi32>
  %1 = stablehlo.convert %0 : (tensor<12xi32>) -> tensor<12xi64>
  return %1 : tensor<12xi64>
}

// CHECK-LABEL: @reorder_with_type_change
// CHECK: %[[CONVERT:.+]] = stablehlo.convert %arg0 : (tensor<3x4xi32>) -> tensor<3x4xi64>
// CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %[[CONVERT]] : (tensor<3x4xi64>) -> tensor<12xi64>
// CHECK: return %[[RESHAPE]]

// -----

func.func @do_not_reorder_with_other_uses(%arg0: tensor<2x2xf64>, %arg1: tensor<4xf32>, %arg2: tensor<f64>) -> (tensor<f64>, tensor<4xf32>) {
  %0 = stablehlo.reshape %arg0 : (tensor<2x2xf64>) -> tensor<4xf64>
  %1 = stablehlo.convert %0 : (tensor<4xf64>) -> tensor<4xf32>
  %2 = stablehlo.subtract %arg1, %1 : tensor<4xf32>
  %3 = stablehlo.reduce(%0 init: %arg2) across dimensions = [0] : (tensor<4xf64>, tensor<f64>) -> tensor<f64>
    reducer(%arg3: tensor<f64>, %arg4: tensor<f64>)  {
    %4 = stablehlo.add %arg3, %arg4 : tensor<f64>
    stablehlo.return %4 : tensor<f64>
  }
  return %3, %2 : tensor<f64>, tensor<4xf32>
}

// CHECK-LABEL: do_not_reorder_with_other_uses
// CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %arg0 : (tensor<2x2xf64>) -> tensor<4xf64>
// CHECK: %[[CONVERT:.+]] = stablehlo.convert %[[RESHAPE]] : (tensor<4xf64>) -> tensor<4xf32>
