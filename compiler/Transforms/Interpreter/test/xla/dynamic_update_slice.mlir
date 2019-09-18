// RUN: iree-opt --lower-xla-to-iree-interpreter --mlir-print-op-generic %s --split-input-file | FileCheck %s --dump-input=fail

// -----

// CHECK-LABEL: func @dynamic_update_slice.1D() -> tensor<4xi32> {
func @dynamic_update_slice.1D() -> tensor<4xi32> {
  // CHECK-DAG: [[C:%[a-z_0-9]+]] = "std.constant"() {value = dense<5> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst = "std.constant"() {value = dense<5> : tensor<1xi32>} : () -> tensor<1xi32>

  // CHECK-DAG: [[C0:%[a-z_0-9]+]] = "std.constant"() {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : () -> tensor<4xi32>
  %cst_0 = "std.constant"() {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : () -> tensor<4xi32>

  // CHECK-DAG: [[C1:%[a-z_0-9]+]] = "std.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "std.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>

  // CHECK-DAG: [[R0:%[a-z_0-9]+]] = "iree.tensor_to_memref"([[C0]]) : (tensor<4xi32>) -> memref<4xi32>
  // CHECK-DAG: [[R1:%[a-z_0-9]+]] = "iree.tensor_to_memref"([[C]]) : (tensor<1xi32>) -> memref<1xi32>
  // CHECK-DAG: [[R2:%[a-z_0-9]+]] = "iree.tensor_to_memref"([[C1]]) : (tensor<i32>) -> memref<i32>
  // CHECK-DAG: [[R3:%[a-z_0-9]+]] = "iree.constant"() {value = dense<1> : tensor<1xi64>} : () -> memref<1xi64>
  // CHECK-DAG: [[R4:%[a-z_0-9]+]] = "iree.constant"() {value = dense<1> : tensor<1xi64>} : () -> memref<1xi64>
  // CHECK-DAG: [[R5:%[a-z_0-9]+]] = "iree_hl_interp.reshape"([[R2]], [[R4]]) : (memref<i32>, memref<1xi64>) -> memref<1xi32>
  // CHECK-DAG: [[R6:%[a-z_0-9]+]] = "iree_hl_interp.concat"([[R5]]) {dimension = 0 : i32} : (memref<1xi32>) -> memref<1xi32>
  // CHECK-DAG: [[R7:%[a-z_0-9]+]] = "iree.constant"() {value = dense<0> : tensor<1xi64>} : () -> memref<1xi64>
  // CHECK-DAG: [[R8:%[a-z_0-9]+]] = "iree_hl_interp.clone"([[R0]]) : (memref<4xi32>) -> memref<4xi32>
  // CHECK-NEXT: "iree_hl_interp.copy"([[R1]], [[R7]], [[R8]], [[R6]], [[R3]]) : (memref<1xi32>, memref<1xi64>, memref<4xi32>, memref<1xi32>, memref<1xi64>) -> ()
  %0 = "xla_hlo.dynamic-update-slice"(%cst_0, %cst, %cst_1) : (tensor<4xi32>, tensor<1xi32>, tensor<i32>) -> tensor<4xi32>

  // CHECK-NEXT: [[R9:%[a-z_0-9]+]] = "iree.memref_to_tensor"([[R8]]) : (memref<4xi32>) -> tensor<4xi32>
  // CHECK-NEXT: "std.return"([[R9]]) : (tensor<4xi32>) -> ()
  "std.return"(%0) : (tensor<4xi32>) -> ()
}

// -----

// CHECK-LABEL: func @dynamic_update_slice.2D() -> tensor<2x4xi32> {
func @dynamic_update_slice.2D() -> tensor<2x4xi32> {
  // CHECK-DAG: [[C:%[a-z_0-9]+]] = "std.constant"() {value = dense<12> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %cst = "std.constant"() {value = dense<12> : tensor<1x1xi32>} : () -> tensor<1x1xi32>

  // CHECK-DAG: [[C0:%[a-z_0-9]+]] = "std.constant"() {value = dense<{{\[\[}}1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi32>} : () -> tensor<2x4xi32>
  %cst_0 = "std.constant"() {value = dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi32>} : () -> tensor<2x4xi32>

  // CHECK-DAG: [[C1:%[a-z_0-9]+]] = "std.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "std.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>

  // CHECK-DAG: [[C2:%[a-z_0-9]+]] = "std.constant"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %cst_2 = "std.constant"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>

  // CHECK-DAG: [[R0:%[a-z_0-9]+]] = "iree.tensor_to_memref"([[C0]]) : (tensor<2x4xi32>) -> memref<2x4xi32>
  // CHECK-DAG: [[R1:%[a-z_0-9]+]] = "iree.tensor_to_memref"([[C]]) : (tensor<1x1xi32>) -> memref<1x1xi32>
  // CHECK-DAG: [[R2:%[a-z_0-9]+]] = "iree.tensor_to_memref"([[C1]]) : (tensor<i32>) -> memref<i32>
  // CHECK-DAG: [[R3:%[a-z_0-9]+]] = "iree.tensor_to_memref"([[C2]]) : (tensor<i32>) -> memref<i32>
  // CHECK-DAG: [[R4:%[a-z_0-9]+]] = "iree.constant"() {value = dense<1> : tensor<2xi64>} : () -> memref<2xi64>
  // CHECK-DAG: [[R5:%[a-z_0-9]+]] = "iree.constant"() {value = dense<1> : tensor<1xi64>} : () -> memref<1xi64>
  // CHECK-DAG: [[R6:%[a-z_0-9]+]] = "iree_hl_interp.reshape"([[R2]], [[R5]]) : (memref<i32>, memref<1xi64>) -> memref<1xi32>
  // CHECK-DAG: [[R7:%[a-z_0-9]+]] = "iree.constant"() {value = dense<1> : tensor<1xi64>} : () -> memref<1xi64>
  // CHECK-DAG: [[R8:%[a-z_0-9]+]] = "iree_hl_interp.reshape"([[R3]], [[R7]]) : (memref<i32>, memref<1xi64>) -> memref<1xi32>
  // CHECK-DAG: [[R9:%[a-z_0-9]+]] = "iree_hl_interp.concat"([[R6]], [[R8]]) {dimension = 0 : i32} : (memref<1xi32>, memref<1xi32>) -> memref<2xi32>
  // CHECK-DAG: [[R10:%[a-z_0-9]+]] = "iree.constant"() {value = dense<0> : tensor<2xi64>} : () -> memref<2xi64>
  // CHECK-NEXT: [[R11:%[a-z_0-9]+]] = "iree_hl_interp.clone"([[R0]]) : (memref<2x4xi32>) -> memref<2x4xi32>
  // CHECK-NEXT: "iree_hl_interp.copy"([[R1]], [[R10]], [[R11]], [[R9]], [[R4]]) : (memref<1x1xi32>, memref<2xi64>, memref<2x4xi32>, memref<2xi32>, memref<2xi64>) -> ()
  %0 = "xla_hlo.dynamic-update-slice"(%cst_0, %cst, %cst_1, %cst_2) : (tensor<2x4xi32>, tensor<1x1xi32>, tensor<i32>, tensor<i32>) -> tensor<2x4xi32>

  // CHECK-NEXT: [[R12:%[a-z_0-9]+]] = "iree.memref_to_tensor"([[R11]]) : (memref<2x4xi32>) -> tensor<2x4xi32>
  // CHECK-NEXT: "std.return"([[R12]]) : (tensor<2x4xi32>) -> ()
  "std.return"(%0) : (tensor<2x4xi32>) -> ()
}

// -----

// CHECK-LABEL: func @dynamic_update_slice.1D.notlast() -> tensor<4xi32> {
func @dynamic_update_slice.1D.notlast() -> tensor<4xi32> {
  // CHECK-DAG: [[C:%[a-z_0-9]+]] = "std.constant"() {value = dense<5> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst = "std.constant"() {value = dense<5> : tensor<1xi32>} : () -> tensor<1xi32>

  // CHECK-DAG: [[C0:%[a-z_0-9]+]] = "std.constant"() {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : () -> tensor<4xi32>
  %cst_0 = "std.constant"() {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : () -> tensor<4xi32>

  // CHECK-DAG: [[C1:%[a-z_0-9]+]] = "std.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "std.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>

  // CHECK-DAG: [[R0:%[a-z_0-9]+]] = "iree.tensor_to_memref"([[C0]]) : (tensor<4xi32>) -> memref<4xi32>
  // CHECK-DAG: [[R1:%[a-z_0-9]+]] = "iree.tensor_to_memref"([[C]]) : (tensor<1xi32>) -> memref<1xi32>
  // CHECK-DAG: [[R2:%[a-z_0-9]+]] = "iree.tensor_to_memref"([[C1]]) : (tensor<i32>) -> memref<i32>
  // CHECK-DAG: [[R3:%[a-z_0-9]+]] = "iree.constant"() {value = dense<1> : tensor<1xi64>} : () -> memref<1xi64>
  // CHECK-DAG: [[R4:%[a-z_0-9]+]] = "iree.constant"() {value = dense<1> : tensor<1xi64>} : () -> memref<1xi64>
  // CHECK-DAG: [[R5:%[a-z_0-9]+]] = "iree_hl_interp.reshape"([[R2]], [[R4]]) : (memref<i32>, memref<1xi64>) -> memref<1xi32>
  // CHECK-DAG: [[R6:%[a-z_0-9]+]] = "iree_hl_interp.concat"([[R5]]) {dimension = 0 : i32} : (memref<1xi32>) -> memref<1xi32>
  // CHECK-DAG: [[R7:%[a-z_0-9]+]] = "iree.constant"() {value = dense<0> : tensor<1xi64>} : () -> memref<1xi64>
  // CHECK-DAG: [[R8:%[a-z_0-9]+]] = "iree_hl_interp.clone"([[R0]]) : (memref<4xi32>) -> memref<4xi32>
  // CHECK-NEXT: "iree_hl_interp.copy"([[R1]], [[R7]], [[R8]], [[R6]], [[R3]]) : (memref<1xi32>, memref<1xi64>, memref<4xi32>, memref<1xi32>, memref<1xi64>) -> ()
  %0 = "xla_hlo.dynamic-update-slice"(%cst_0, %cst, %cst_1) : (tensor<4xi32>, tensor<1xi32>, tensor<i32>) -> tensor<4xi32>

  // CHECK-NEXT: [[R9:%[a-z_0-9]+]] = "iree.memref_to_tensor"([[R8]]) : (memref<4xi32>) -> tensor<4xi32>
  // CHECK-NEXT: [[R11:%[a-z_0-9]+]] = "xla_hlo.add"([[C0]], [[R9]]) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = "xla_hlo.add"(%cst_0, %0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  // CHECK-DAG: "std.return"([[R10]]) : (tensor<4xi32>) -> ()
  "std.return"(%1) : (tensor<4xi32>) -> ()
}

// -----

// CHECK-LABEL: func @dynamic_update_slice.2D.notlast() -> tensor<2x4xi32> {
func @dynamic_update_slice.2D.notlast() -> tensor<2x4xi32> {
  // CHECK-DAG: [[C:%[a-z_0-9]+]] = "std.constant"() {value = dense<12> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %cst = "std.constant"() {value = dense<12> : tensor<1x1xi32>} : () -> tensor<1x1xi32>

  // CHECK-DAG: [[C0:%[a-z_0-9]+]] = "std.constant"() {value = dense<{{\[\[}}1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi32>} : () -> tensor<2x4xi32>
  %cst_0 = "std.constant"() {value = dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi32>} : () -> tensor<2x4xi32>

  // CHECK-DAG: [[C1:%[a-z_0-9]+]] = "std.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "std.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>

  // CHECK-DAG: [[C2:%[a-z_0-9]+]] = "std.constant"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %cst_2 = "std.constant"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>

  // CHECK-DAG: [[R0:%[a-z_0-9]+]] = "iree.tensor_to_memref"([[C0]]) : (tensor<2x4xi32>) -> memref<2x4xi32>
  // CHECK-DAG: [[R1:%[a-z_0-9]+]] = "iree.tensor_to_memref"([[C]]) : (tensor<1x1xi32>) -> memref<1x1xi32>
  // CHECK-DAG: [[R2:%[a-z_0-9]+]] = "iree.tensor_to_memref"([[C1]]) : (tensor<i32>) -> memref<i32>
  // CHECK-DAG: [[R3:%[a-z_0-9]+]] = "iree.tensor_to_memref"([[C2]]) : (tensor<i32>) -> memref<i32>
  // CHECK-DAG: [[R4:%[a-z_0-9]+]] = "iree.constant"() {value = dense<1> : tensor<2xi64>} : () -> memref<2xi64>
  // CHECK-DAG: [[R5:%[a-z_0-9]+]] = "iree.constant"() {value = dense<1> : tensor<1xi64>} : () -> memref<1xi64>
  // CHECK-DAG: [[R6:%[a-z_0-9]+]] = "iree_hl_interp.reshape"([[R2]], [[R5]]) : (memref<i32>, memref<1xi64>) -> memref<1xi32>
  // CHECK-DAG: [[R7:%[a-z_0-9]+]] = "iree.constant"() {value = dense<1> : tensor<1xi64>} : () -> memref<1xi64>
  // CHECK-DAG: [[R8:%[a-z_0-9]+]] = "iree_hl_interp.reshape"([[R3]], [[R7]]) : (memref<i32>, memref<1xi64>) -> memref<1xi32>
  // CHECK-DAG: [[R9:%[a-z_0-9]+]] = "iree_hl_interp.concat"([[R6]], [[R8]]) {dimension = 0 : i32} : (memref<1xi32>, memref<1xi32>) -> memref<2xi32>
  // CHECK-DAG: [[R10:%[a-z_0-9]+]] = "iree.constant"() {value = dense<0> : tensor<2xi64>} : () -> memref<2xi64>
  // CHECK-DAG: [[R11:%[a-z_0-9]+]] = "iree_hl_interp.clone"([[R0]]) : (memref<2x4xi32>) -> memref<2x4xi32>
  // CHECK-NEXT: "iree_hl_interp.copy"([[R1]], [[R10]], [[R11]], [[R9]], [[R4]]) : (memref<1x1xi32>, memref<2xi64>, memref<2x4xi32>, memref<2xi32>, memref<2xi64>) -> ()
  // CHECK-NEXT: [[R12:%[a-z_0-9]+]] = "iree.memref_to_tensor"([[R11]]) : (memref<2x4xi32>) -> tensor<2x4xi32>
  %0 = "xla_hlo.dynamic-update-slice"(%cst_0, %cst, %cst_1, %cst_2) : (tensor<2x4xi32>, tensor<1x1xi32>, tensor<i32>, tensor<i32>) -> tensor<2x4xi32>

  // CHECK-NEXT: [[R13:%[a-z_0-9]+]] = "xla_hlo.add"([[C0]], [[R12]]) : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>
  %1 = "xla_hlo.add"(%cst_0, %0) : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>

  // CHECK-NEXT: "std.return"([[R13]]) : (tensor<2x4xi32>) -> ()
  "std.return"(%1) : (tensor<2x4xi32>) -> ()
}
