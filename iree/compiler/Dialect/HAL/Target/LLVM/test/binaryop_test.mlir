// RUN: iree-opt -split-input-file -iree-hal-transformation-pipeline -iree-hal-target-backends=llvm-ir %s | IreeFileCheck %s
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

// CHECK-LABEL: hal.executable @simpleMath_ex_dispatch_0
// CHECK-DAG:   hal.executable.entry_point @simpleMath_rgn_dispatch_0
// CHECK-DAG:   hal.executable.binary attributes {
// CHECK-SAME:     data = dense
// CHECK-SAME:     format = 1280071245 : i32} {
// CHECK:      module {
// CHECK-NEXT:   llvm.func @simpleMath_rgn_dispatch_0(
// CHECK-SAME:   %{{[a-zA-Z0-9_]*}}: !llvm<"float*">, %{{[a-zA-Z0-9_]*}}: !llvm<"float*">, %{{[a-zA-Z0-9_]*}}: !llvm.i64, %{{[a-zA-Z0-9_]*}}: !llvm.i64, %{{[a-zA-Z0-9_]*}}: !llvm.i64,
// CHECK-SAME:   %{{[a-zA-Z0-9_]*}}: !llvm<"float*">, %{{[a-zA-Z0-9_]*}}: !llvm<"float*">, %{{[a-zA-Z0-9_]*}}: !llvm.i64, %{{[a-zA-Z0-9_]*}}: !llvm.i64, %{{[a-zA-Z0-9_]*}}: !llvm.i64)
// CHECK-SAME:   attributes {iree.executable.export} {
