// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-check-ir-before-llvm-conversion))" %s --split-input-file | FileCheck %s

// CHECK-LABEL: func @alloca_complex(
module {
  func.func @alloca_complex(%arg0: index) {
    %0 = memref.alloca() : memref<128xcomplex<f32>>
    return
  }
}
