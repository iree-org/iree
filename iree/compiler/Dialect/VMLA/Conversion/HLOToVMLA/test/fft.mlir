// RUN: iree-opt -split-input-file -iree-vmla-pre-conversion-lowering -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

func private @fft(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) ->  (tensor<8xf32>, tensor<8xf32>) {
  // CHECK: [[RS:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[8]>
  // CHECK-NEXT: [[C32:%.+]] = constant 32 : index
  // CHECK-NEXT: [[OUTBUF1:%.+]] = vmla.buffer.alloc byte_length = [[C32]] : !vmla.buffer
  // CHECK-NEXT: [[OUTBUF2:%.+]] = vmla.buffer.alloc byte_length = [[C32]] : !vmla.buffer
  // CHECK-NEXT: vmla.fft %arg0([[RS]] : !shapex.ranked_shape<[8]>),  %arg1([[RS]] : !shapex.ranked_shape<[8]>), out [[OUTBUF1]], [[OUTBUF2]] : f32, f32
  %real, %imag = "vmla.fft.pseudo"(%arg0, %arg1) : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  return %real, %imag : tensor<8xf32>, tensor<8xf32>
}
