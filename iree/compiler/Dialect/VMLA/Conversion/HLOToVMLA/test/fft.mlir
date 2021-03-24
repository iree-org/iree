// RUN: iree-opt -split-input-file -iree-vmla-pre-conversion-lowering -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

func private @fft(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) ->  (tensor<8xf32>, tensor<8xf32>) {
  //  CHECK-DAG: [[RS:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[8]>
  //  CHECK-DAG: [[C32:%.+]] = constant 32 : index
  //      CHECK: [[OUTBUF1:%.+]] = vmla.buffer.alloc byte_length = [[C32]] : !vmla.buffer
  // CHECK-NEXT: [[OUTBUF2:%.+]] = vmla.buffer.alloc byte_length = [[C32]] : !vmla.buffer
  // CHECK-NEXT: vmla.fft %arg0([[RS]] : !shapex.ranked_shape<[8]>),  %arg1([[RS]] : !shapex.ranked_shape<[8]>), out [[OUTBUF1]], [[OUTBUF2]] : f32
  %real, %imag = "vmla.fft.pseudo"(%arg0, %arg1) : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  return %real, %imag : tensor<8xf32>, tensor<8xf32>
}

func private @ifft(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) ->  (tensor<8xf32>, tensor<8xf32>)  {
  //  CHECK-DAG: [[RS:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[8]>
  //  CHECK-DAG: [[C32:%.+]] = constant 32 : index
  //      CHECK: [[OUTBUF1:%.+]] = vmla.buffer.alloc byte_length = [[C32]] : !vmla.buffer
  // CHECK-NEXT: [[OUTBUF2:%.+]] = vmla.buffer.alloc byte_length = [[C32]] : !vmla.buffer
  // CHECK-NEXT: vmla.ifft %arg0([[RS]] : !shapex.ranked_shape<[8]>),  %arg1([[RS]] : !shapex.ranked_shape<[8]>), out [[OUTBUF1]], [[OUTBUF2]] : f32
  %real, %imag = "vmla.ifft.pseudo"(%arg0, %arg1) : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  return %real, %imag : tensor<8xf32>, tensor<8xf32>
}

func private @rfft(%arg0: tensor<8xf32>) ->  (tensor<5xf32>, tensor<5xf32>) {
  //  CHECK-DAG: [[RS:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[8]>
  //  CHECK-DAG: [[C20:%.+]] = constant 20 : index
  //      CHECK: [[OUTBUF1:%.+]] = vmla.buffer.alloc byte_length = [[C20]] : !vmla.buffer
  // CHECK-NEXT: [[OUTBUF2:%.+]] = vmla.buffer.alloc byte_length = [[C20]] : !vmla.buffer
  // CHECK-NEXT: vmla.rfft %arg0([[RS]] : !shapex.ranked_shape<[8]>), out [[OUTBUF1]], [[OUTBUF2]] : f32
  %real, %imag = "vmla.rfft.pseudo"(%arg0) : (tensor<8xf32>) -> (tensor<5xf32>, tensor<5xf32>)
  return %real, %imag : tensor<5xf32>, tensor<5xf32>
}

func private @irfft(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<8xf32> {
  //  CHECK-DAG: [[RS:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[5]>
  //  CHECK-DAG: [[C32:%.+]] = constant 32 : index
  // CHECK-NEXT: [[OUTBUF1:%.+]] = vmla.buffer.alloc byte_length = [[C32]] : !vmla.buffer
  // CHECK-NEXT: vmla.irfft %arg0([[RS]] : !shapex.ranked_shape<[5]>),  %arg1([[RS]] : !shapex.ranked_shape<[5]>), out [[OUTBUF1]] : f32
  %real = "vmla.irfft.pseudo"(%arg0, %arg1) : (tensor<5xf32>, tensor<5xf32>) -> (tensor<8xf32>)
  return %real : tensor<8xf32>
}

