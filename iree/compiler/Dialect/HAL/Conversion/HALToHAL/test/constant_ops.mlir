// RUN: iree-opt -split-input-file -iree-convert-to-hal -verify-diagnostics %s | IreeFileCheck %s

// CHECK-LABEL: func @constant_subspan
func @constant_subspan() {
  //  CHECK-DAG: [[BUFFER:%.+]] = hal.variable.load @pool_buffer : !hal.buffer
  //  CHECK-DAG: [[OFFSET:%.+]] = constant 123 : index
  //  CHECK-DAG: [[LENGTH:%.+]] = constant 16 : index
  // CHECK-NEXT: = hal.buffer.subspan [[BUFFER]], [[OFFSET]], [[LENGTH]] : !hal.buffer
  %cst0 = hal.constant.subspan @pool_buffer[#hal.byte_range<123, 16>] : tensor<4xf32>
  return
}
hal.variable @pool_buffer : !hal.buffer
