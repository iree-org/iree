// RUN: iree-translate -split-input-file -iree-vm-ir-to-bytecode-module -iree-vm-bytecode-module-output-format=flatbuffer-text %s | IreeFileCheck %s

// CHECK: name: "constants"
vm.module @constants {
  vm.export @func
  vm.func @func() {
    vm.return
  }

  // CHECK: rodata_segments: [ {

  // CHECK: data: [ 1, 2, 3 ]
  vm.rodata @dense_i8s dense<[1, 2, 3]> : tensor<3xi8>

  // CHECK: data: [ 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64 ]
  vm.rodata @dense_float32s dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>

  // CHECK: data: [ 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63 ]
  vm.rodata @splat_float32s dense<1.000000e+00> : tensor<3xf32>
}
