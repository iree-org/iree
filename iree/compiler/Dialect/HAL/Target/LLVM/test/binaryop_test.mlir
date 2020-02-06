// RUN: iree-opt -split-input-file -iree-hal-translate-executables -iree-hal-target-backends=llvm-ir %s | IreeFileCheck %s
flow.executable @simpleMath_ex_dispatch_0 {
  flow.dispatch.entry @simpleMath_rgn_dispatch_0 attributes {
      workload = dense<[4, 1, 1]> : vector<3xi32>
  }
  module {
    func @simpleMath_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}

// CHECK-LABEL: hal.executable @simpleMath_ex_dispatch_0 {
// CHECK-NEXT:   hal.executable.entry_point @simpleMath_rgn_dispatch_0 attributes {ordinal = 0 : i32, workgroup_size = dense<[32, 1, 1]> : vector<3xi32>}
// CHECK-NEXT:   hal.executable.binary attributes {
// CHECK-SAME:     data = dense
// CHECK-SAME:     format = 1280071245 : i32} {
// CHECK-NEXT:     module {
// CHECK-NEXT:       llvm.func @simpleMath_rgn_dispatch_0(%arg0: !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">, %arg1: !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">) attributes {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : vector<3xi32>, iree.executable.workload = dense<[4, 1, 1]> : vector<3xi32>} {
// CHECK-NEXT:          %0 = llvm.load %arg0 : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">
// CHECK-NEXT:          %1 = llvm.load %arg1 : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">
