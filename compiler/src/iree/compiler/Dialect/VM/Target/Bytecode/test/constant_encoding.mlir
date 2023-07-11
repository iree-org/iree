// RUN: iree-compile --split-input-file --compile-mode=vm \
// RUN:   --iree-vm-bytecode-module-output-format=flatbuffer-text %s | FileCheck %s

// CHECK: "name": "constants"
vm.module @constants {
  vm.export @func
  vm.func @func() {
    vm.return
  }

  // CHECK: "rodata_segments": [{

  // Tests that we densely pack i2 values. Note that the final element (3) is
  // padded out with zeros.
  //      CHECK: "embedded_data": [
  // CHECK-NEXT:   228,
  // CHECK-NEXT:   26,
  // CHECK-NEXT:   3
  // CHECK-NEXT: ]
  vm.rodata private @dense_i2 dense<[0, 1, 2, 3, 2, 2, 1, 0, 3]> : tensor<9xi2>

  // Tests that we densely pack i3 values and insert the wasted 2-bits of
  // padding in each byte. Smarter implementations would pack to 16- or 64-bit
  // physical storage to waste fewer bits.
  //      CHECK: "embedded_data": [
  // CHECK-NEXT:   8,
  // CHECK-NEXT:   26,
  // CHECK-NEXT:   44,
  // CHECK-NEXT:   62
  // CHECK-NEXT: ]
  vm.rodata private @dense_i3 dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi3>

  // Tests that we densely pack i4 values and handle partial values (14).
  //      CHECK: "embedded_data": [
  // CHECK-NEXT:   16,
  // CHECK-NEXT:   50,
  // CHECK-NEXT:   84,
  // CHECK-NEXT:   118,
  // CHECK-NEXT:   152,
  // CHECK-NEXT:   186,
  // CHECK-NEXT:   220,
  // CHECK-NEXT:   254,
  // CHECK-NEXT:   14
  // CHECK-NEXT: ]
  vm.rodata private @dense_i4 dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14]> : tensor<17xi4>

  //      CHECK: "embedded_data": [
  // CHECK-NEXT:   98,
  // CHECK-NEXT:   16,
  // CHECK-NEXT:   197,
  // CHECK-NEXT:   28
  // CHECK-NEXT: ]
  vm.rodata private @dense_i5 dense<[2, 3, 4, 5, 6, 7]> : tensor<6xi5>

  //      CHECK: "embedded_data": [
  // CHECK-NEXT:   1,
  // CHECK-NEXT:   2,
  // CHECK-NEXT:   3
  // CHECK-NEXT: ]
  vm.rodata private @dense_i8 dense<[1, 2, 3]> : tensor<3xi8>

  //      CHECK: "embedded_data": [
  // CHECK-NEXT:   1,
  // CHECK-NEXT:   4,
  // CHECK-NEXT:   12,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   4,
  // CHECK-NEXT:   10,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0
  // CHECK-NEXT: ]
  vm.rodata private @dense_i9 dense<[1, 2, 3, 4, 5]> : tensor<5xi9>

  //      CHECK: "embedded_data": [
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   60,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   64,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   66
  // CHECK-NEXT: ]
  vm.rodata private @dense_float16 dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf16>

  //      CHECK: "embedded_data": [
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   60,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   60,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   60
  // CHECK-NEXT: ]
  vm.rodata private @splat_float16 dense<1.000000e+00> : tensor<3xf16>

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
  vm.rodata private @dense_float32 dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>

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
  vm.rodata private @splat_float32 dense<1.000000e+00> : tensor<3xf32>

}
