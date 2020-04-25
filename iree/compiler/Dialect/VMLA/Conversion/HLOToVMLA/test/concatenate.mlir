// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @concatenate_0
func @concatenate_0() -> (tensor<2x5xi32>) attributes { sym_visibility = "private" } {
  // CHECK-DAG: %[[ARG0:.+]] = "vmla.constant"() {{.+}} tensor<2x2xi32>
  %c0 = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK-DAG: %[[ARG1:.+]] = "vmla.constant"() {{.+}} tensor<2x3xi32>
  %c1 = constant dense<[[5, 6, 7], [8, 9, 10]]> : tensor<2x3xi32>
  // CHECK: %[[DST:.+]] = "vmla.buffer.alloc"(%c40)
  // CHECK: "vmla.copy"(
  // CHECK-SAME: %[[ARG0]], %rs2_2, %c0, %c0,
  // CHECK-SAME: %[[DST]], %rs2_5, %c0, %c0,
  // CHECK-SAME: %c2, %c2
  // CHECK-SAME: ) {element_type = i32}
  // CHECK: "vmla.copy"(
  // CHECK-SAME: %[[ARG1]], %rs2_3, %c0, %c0,
  // CHECK-SAME: %[[DST]], %rs2_5, %c0, %c2,
  // CHECK-SAME: %c2, %c3
  // CHECK-SAME: ) {element_type = i32}
  %0 = "xla_hlo.concatenate"(%c0, %c1) {dimension = 1} : (tensor<2x2xi32>, tensor<2x3xi32>) -> tensor<2x5xi32>
  // CHECK-NEXT: return %[[DST]]
  return %0: tensor<2x5xi32>
}

// -----

// CHECK-LABEL: @concatenate_1
func @concatenate_1() -> (tensor<2x5xi32>) attributes { sym_visibility = "private" } {
  // CHECK-DAG: %[[ARG0:.+]] = "vmla.constant"() {{.+}} tensor<2x3xi32>
  %c1 = constant dense<[[5, 6, 7], [8, 9, 10]]> : tensor<2x3xi32>
  // CHECK-DAG: %[[ARG1:.+]] = "vmla.constant"() {{.+}} tensor<2x2xi32>
  %c0 = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK: %[[DST:.+]] = "vmla.buffer.alloc"(%c40)
  // CHECK: "vmla.copy"(
  // CHECK-SAME: %[[ARG0]], %rs2_3, %c0, %c0,
  // CHECK-SAME: %[[DST]], %rs2_5, %c0, %c0,
  // CHECK-SAME: %c2, %c3
  // CHECK-SAME: ) {element_type = i32}
  // CHECK: "vmla.copy"(
  // CHECK-SAME: %[[ARG1]], %rs2_2, %c0, %c0,
  // CHECK-SAME: %[[DST]], %rs2_5, %c0, %c3,
  // CHECK-SAME: %c2, %c2
  // CHECK-SAME: ) {element_type = i32}
  %0 = "xla_hlo.concatenate"(%c1, %c0) {dimension = 1} : (tensor<2x3xi32>, tensor<2x2xi32>) -> tensor<2x5xi32>
  // CHECK-NEXT: return %[[DST]]
  return %0: tensor<2x5xi32>
}

// -----

// CHECK-LABEL: @concatenate_2
func @concatenate_2() -> (tensor<2x7xi32>) attributes { sym_visibility = "private" } {
  // CHECK-DAG: %[[ARG0:.+]] = "vmla.constant"() {{.+}} tensor<2x2xi32>
  %c0 = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK-DAG: %[[ARG1:.+]] = "vmla.constant"() {{.+}} tensor<2x3xi32>
  %c1 = constant dense<[[5, 6, 7], [8, 9, 10]]> : tensor<2x3xi32>
  // CHECK-DAG: %[[ARG2:.+]] = "vmla.constant"() {{.+}} tensor<2x2xi32>
  %c2 = constant dense<[[11, 12], [13, 14]]> : tensor<2x2xi32>
  // CHECK: %[[DST:.+]] = "vmla.buffer.alloc"(%c56)
  // CHECK: "vmla.copy"(
  // CHECK-SAME: %[[ARG0]], %rs2_2, %c0, %c0,
  // CHECK-SAME: %[[DST]], %rs2_7, %c0, %c0,
  // CHECK-SAME: %c2, %c2
  // CHECK-SAME: ) {element_type = i32}
  // CHECK: "vmla.copy"(
  // CHECK-SAME: %[[ARG1]], %rs2_3, %c0, %c0,
  // CHECK-SAME: %[[DST]], %rs2_7, %c0, %c2,
  // CHECK-SAME: %c2, %c3
  // CHECK-SAME: ) {element_type = i32}
  // CHECK: "vmla.copy"(
  // CHECK-SAME: %[[ARG2]], %rs2_2, %c0, %c0,
  // CHECK-SAME: %[[DST]], %rs2_7, %c0, %c5,
  // CHECK-SAME: %c2, %c2
  // CHECK-SAME: ) {element_type = i32}
  %0 = "xla_hlo.concatenate"(%c0, %c1, %c2) {dimension = 1} : (tensor<2x2xi32>, tensor<2x3xi32>, tensor<2x2xi32>) -> tensor<2x7xi32>
  // CHECK-NEXT: return %[[DST]]
  return %0: tensor<2x7xi32>
}

// -----

// CHECK-LABEL: @concatenate_3
func @concatenate_3() -> (tensor<4x2xi32>) attributes { sym_visibility = "private" } {
  // CHECK-DAG: %[[ARG0:.+]] = "vmla.constant"() {{.+}} tensor<2x2xi32>
  %c0 = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK-DAG: %[[ARG1:.+]] = "vmla.constant"() {{.+}} tensor<2x2xi32>
  %c2 = constant dense<[[11, 12], [13, 14]]> : tensor<2x2xi32>
  // CHECK: %[[DST:.+]] = "vmla.buffer.alloc"(%c32)
  // CHECK: "vmla.copy"(
  // CHECK-SAME: %[[ARG0]], %rs2_2, %c0, %c0,
  // CHECK-SAME: %[[DST]], %rs4_2, %c0, %c0,
  // CHECK-SAME: %c2, %c2
  // CHECK-SAME: ) {element_type = i32}
  // CHECK: "vmla.copy"(
  // CHECK-SAME: %[[ARG1]], %rs2_2, %c0, %c0,
  // CHECK-SAME: %[[DST]], %rs4_2, %c2, %c0,
  // CHECK-SAME: %c2, %c2
  // CHECK-SAME: ) {element_type = i32}
  %0 = "xla_hlo.concatenate"(%c0, %c2) {dimension = 0} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<4x2xi32>
  // CHECK-NEXT: return %[[DST]]
  return %0: tensor<4x2xi32>
}
