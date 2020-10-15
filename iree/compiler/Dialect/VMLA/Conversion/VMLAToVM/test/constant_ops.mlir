// RUN: iree-opt -split-input-file -iree-convert-vmla-to-vm -cse %s | IreeFileCheck %s

// CHECK-LABEL: vm.func @denseConstant
func @denseConstant() -> !vmla.buffer {
  // CHECK-NEXT: [[RODATA:%.+]] = vm.rodata.inline : !vm.ref<!iree.byte_buffer> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32
  // CHECK-NEXT: = vm.call @vmla.buffer.const([[RODATA]]) : (!vm.ref<!iree.byte_buffer>) -> !vm.ref<!vmla.buffer>
  %0 = vmla.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32> -> !vmla.buffer
  return %0 : !vmla.buffer
}

// -----

// CHECK-LABEL: @splatConstant
func @splatConstant() -> !vmla.buffer {
  // CHECK-NEXT: [[RODATA:%.+]] = vm.rodata.inline : !vm.ref<!iree.byte_buffer> = dense<0.176776692> : tensor<1xf32>
  // CHECK-NEXT: [[SPLATTED:%.+]] = vm.call @vmla.buffer.const([[RODATA]]) : (!vm.ref<!iree.byte_buffer>) -> !vm.ref<!vmla.buffer>
  // CHECK-NEXT: [[LENGTH:%.+]] = vm.const.i32 2359296 : i32
  // CHECK-NEXT: [[RESULT:%.+]] = vm.call @vmla.buffer.alloc([[LENGTH]]) : (i32) -> !vm.ref<!vmla.buffer>
  // CHECK-NEXT: vm.call @vmla.buffer.fill([[SPLATTED]], [[RESULT]]) : (!vm.ref<!vmla.buffer>, !vm.ref<!vmla.buffer>) -> ()
  %0 = vmla.constant dense<0.176776692> : tensor<1x4x384x384xf32> -> !vmla.buffer
  // CHECK-NEXT: vm.return [[RESULT]] : !vm.ref<!vmla.buffer>
  return %0 : !vmla.buffer
}
