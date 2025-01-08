// RUN: iree-opt --split-input-file --allow-unregistered-dialect \
// RUN:          --pass-pipeline="builtin.module(util.func(iree-global-opt-pack-storage))" \
// RUN:          %s | FileCheck %s


func.func @addition(%arg0 : tensor<4xi1>, %arg1 : tensor<4xi1>) -> tensor<4xi1> {
  %0 = arith.xori %arg0, %arg1 : tensor<4xi1>
  return %0 : tensor<4xi1>
}

