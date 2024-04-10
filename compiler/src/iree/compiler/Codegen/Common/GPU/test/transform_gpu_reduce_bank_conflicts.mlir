// RUN: iree-opt %s --iree-transform-dialect-interpreter --transform-dialect-drop-schedule | FileCheck %s

// CHECK-LABEL: func.func @pad_alloc
func.func @pad_alloc() {
// CHECK: %[[A:.*]] = memref.alloc() : memref<64x68xf16, #gpu.address_space<workgroup>>
  %alloc = memref.alloc() : memref<64x64xf16, #gpu.address_space<workgroup>>
// CHECK: %[[S1:.*]] = memref.subview %[[A]][0, 0] [64, 64] [1, 1] : memref<64x68xf16, #gpu.address_space<workgroup>> to memref<64x64xf16, strided<[68, 1]>, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f16
  %cst_1 = arith.constant dense<1.0> : vector<1x8xf16>

// CHECK: vector.transfer_read %[[S1]][%{{.*}}, %{{.*}}], %{{.*}} : memref<64x64xf16, strided<[68, 1]>, #gpu.address_space<workgroup>>, vector<8xf16>
  %2 = vector.transfer_read %alloc[%c0, %c0], %cst_0 : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<8xf16>
// CHECK: vector.transfer_write %{{.*}}, %[[S1]][%{{.*}}, %{{.*}}] : vector<1x8xf16>, memref<64x64xf16, strided<[68, 1]>, #gpu.address_space<workgroup>>
  vector.transfer_write %cst_1, %alloc[%c0, %c0] :
    vector<1x8xf16>, memref<64x64xf16, #gpu.address_space<workgroup>>
  return
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(
      %variant_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.reduce_shared_memory_bank_conflicts %0 { padding_size_bits = 64 } : (!transform.any_op) -> ()
    transform.yield
  }
} // module
