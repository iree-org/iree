// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @concatenate_0
func @concatenate_0() -> (tensor<2x5xi32>) {
  // CHECK-DAG: [[ARG0:%.+]] = "vmla.constant"() {{.+}} tensor<2x2xi32>
  %c0 = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK-DAG: [[ARG1:%.+]] = "vmla.constant"() {{.+}} tensor<2x3xi32>
  %c1 = constant dense<[[5, 6, 7], [8, 9, 10]]> : tensor<2x3xi32>
  // CHECK: [[DST:%.+]] = "vmla.buffer.alloc"(%c40_i32)
  // CHECK-NEXT: [[DST_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: [[ARG0_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: "vmla.copy"(
  // CHECK-SAME: [[ARG0]], [[ARG0_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: [[DST]], [[DST_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: %c2_i32, %c2_i32
  // CHECK-SAME: ) {element_type = i32}
  // CHECK-NEXT: [[ARG1_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: "vmla.copy"(
  // CHECK-SAME: [[ARG1]], [[ARG1_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: [[DST]], [[DST_SHAPE]], %c0_i32, %c2_i32,
  // CHECK-SAME: %c2_i32, %c3_i32
  // CHECK-SAME: ) {element_type = i32}
  %0 = "xla_hlo.concatenate"(%c0, %c1) {dimension = 1} : (tensor<2x2xi32>, tensor<2x3xi32>) -> tensor<2x5xi32>
  // CHECK-NEXT: return [[DST]]
  return %0: tensor<2x5xi32>
}

// -----

// CHECK-LABEL: @concatenate_1
func @concatenate_1() -> (tensor<2x5xi32>) {
  // CHECK-DAG: [[ARG0:%.+]] = "vmla.constant"() {{.+}} tensor<2x3xi32>
  %c1 = constant dense<[[5, 6, 7], [8, 9, 10]]> : tensor<2x3xi32>
  // CHECK-DAG: [[ARG1:%.+]] = "vmla.constant"() {{.+}} tensor<2x2xi32>
  %c0 = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK: [[DST:%.+]] = "vmla.buffer.alloc"(%c40_i32)
  // CHECK-NEXT: [[DST_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: [[ARG0_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: "vmla.copy"(
  // CHECK-SAME: [[ARG0]], [[ARG0_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: [[DST]], [[DST_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: %c2_i32, %c3_i32
  // CHECK-SAME: ) {element_type = i32}
  // CHECK-NEXT: [[ARG1_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: "vmla.copy"(
  // CHECK-SAME: [[ARG1]], [[ARG1_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: [[DST]], [[DST_SHAPE]], %c0_i32, %c3_i32,
  // CHECK-SAME: %c2_i32, %c2_i32
  // CHECK-SAME: ) {element_type = i32}
  %0 = "xla_hlo.concatenate"(%c1, %c0) {dimension = 1} : (tensor<2x3xi32>, tensor<2x2xi32>) -> tensor<2x5xi32>
  // CHECK-NEXT: return [[DST]]
  return %0: tensor<2x5xi32>
}

// -----

// CHECK-LABEL: @concatenate_2
func @concatenate_2() -> (tensor<2x7xi32>) {
  // CHECK-DAG: [[ARG0:%.+]] = "vmla.constant"() {{.+}} tensor<2x2xi32>
  %c0 = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK-DAG: [[ARG1:%.+]] = "vmla.constant"() {{.+}} tensor<2x3xi32>
  %c1 = constant dense<[[5, 6, 7], [8, 9, 10]]> : tensor<2x3xi32>
  // CHECK-DAG: [[ARG2:%.+]] = "vmla.constant"() {{.+}} tensor<2x2xi32>
  %c2 = constant dense<[[11, 12], [13, 14]]> : tensor<2x2xi32>
  // CHECK: [[DST:%.+]] = "vmla.buffer.alloc"(%c56_i32)
  // CHECK-NEXT: [[DST_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: [[ARG0_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: "vmla.copy"(
  // CHECK-SAME: [[ARG0]], [[ARG0_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: [[DST]], [[DST_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: %c2_i32, %c2_i32
  // CHECK-SAME: ) {element_type = i32}
  // CHECK-NEXT: [[ARG1_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: "vmla.copy"(
  // CHECK-SAME: [[ARG1]], [[ARG1_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: [[DST]], [[DST_SHAPE]], %c0_i32, %c2_i32,
  // CHECK-SAME: %c2_i32, %c3_i32
  // CHECK-SAME: ) {element_type = i32}
  // CHECK-NEXT: [[ARG2_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: "vmla.copy"(
  // CHECK-SAME: [[ARG2]], [[ARG2_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: [[DST]], [[DST_SHAPE]], %c0_i32, %c5_i32,
  // CHECK-SAME: %c2_i32, %c2_i32
  // CHECK-SAME: ) {element_type = i32}
  %0 = "xla_hlo.concatenate"(%c0, %c1, %c2) {dimension = 1} : (tensor<2x2xi32>, tensor<2x3xi32>, tensor<2x2xi32>) -> tensor<2x7xi32>
  // CHECK-NEXT: return [[DST]]
  return %0: tensor<2x7xi32>
}

// -----

// CHECK-LABEL: @concatenate_3
func @concatenate_3() -> (tensor<4x2xi32>) {
  // CHECK-DAG: [[ARG0:%.+]] = "vmla.constant"() {{.+}} tensor<2x2xi32>
  %c0 = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK-DAG: [[ARG1:%.+]] = "vmla.constant"() {{.+}} tensor<2x2xi32>
  %c2 = constant dense<[[11, 12], [13, 14]]> : tensor<2x2xi32>
  // CHECK: [[DST:%.+]] = "vmla.buffer.alloc"(%c32_i32)
  // CHECK-NEXT: [[DST_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: [[ARG0_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: "vmla.copy"(
  // CHECK-SAME: [[ARG0]], [[ARG0_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: [[DST]], [[DST_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: %c2_i32, %c2_i32
  // CHECK-SAME: ) {element_type = i32}
  // CHECK-NEXT: [[ARG1_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: "vmla.copy"(
  // CHECK-SAME: [[ARG1]], [[ARG1_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: [[DST]], [[DST_SHAPE]], %c2_i32, %c0_i32,
  // CHECK-SAME: %c2_i32, %c2_i32
  // CHECK-SAME: ) {element_type = i32}
  %0 = "xla_hlo.concatenate"(%c0, %c2) {dimension = 0} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<4x2xi32>
  // CHECK-NEXT: return [[DST]]
  return %0: tensor<4x2xi32>
}
