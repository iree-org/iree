// RUN: iree-compile --split-input-file --compile-mode=vm \
// RUN:   --iree-vm-bytecode-module-output-format=flatbuffer-text %s | FileCheck %s

// CHECK: "name": "constants"
vm.module @constants {
  vm.export @func
  vm.func @func() {
    vm.return
  }

  // CHECK: "rodata_segments": [{

  //      CHECK: "embedded_data": [
  // CHECK-NEXT:   1,
  // CHECK-NEXT:   2,
  // CHECK-NEXT:   3
  // CHECK-NEXT: ]
  vm.rodata private @dense_i8s dense<[1, 2, 3]> : tensor<3xi8>

  //      CHECK: "embedded_data": [
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   128,
  // CHECK-NEXT:   63,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   64,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   64,
  // CHECK-NEXT:   64
  // CHECK-NEXT: ]
  vm.rodata private @dense_float32s dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>

  //      CHECK: "embedded_data": [
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   128,
  // CHECK-NEXT:   63,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   128,
  // CHECK-NEXT:   63,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   128,
  // CHECK-NEXT:   63
  // CHECK-NEXT: ]
  vm.rodata private @splat_float32s dense<1.000000e+00> : tensor<3xf32>

  //      CHECK: "embedded_data": [
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   60,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   64,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   66
  // CHECK-NEXT: ]
  vm.rodata private @dense_float16s dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf16>

}
