// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --iree-opt-experimental-early-data-tiling --compile-to=dispatch-creation | FileCheck %s
// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --iree-opt-experimental-early-data-tiling --compile-to=executable-targets | FileCheck %s --check-prefix=UKERNEL
// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-target-triple=x86_64-unknown-unknown-eabi-elf --iree-opt-experimental-early-data-tiling --compile-to=executable-targets | FileCheck %s --check-prefix=UKERNEL

// Fusing the epilogue and the unpack keeps the contraction on the mmt4d ukernel.
// UKERNEL-LABEL: llvm.func @dynamic_m_bias_dispatch_{{[0-9]+}}_mmt4d_
// UKERNEL:         llvm.call @iree_uk_mmt4d
// UKERNEL-LABEL: llvm.func @odd_relu_dispatch_{{[0-9]+}}_mmt4d_
// UKERNEL:         llvm.call @iree_uk_mmt4d
// UKERNEL-LABEL: llvm.func @batched_relu_dispatch_{{[0-9]+}}_batch_mmt4d_
// UKERNEL:         llvm.call @iree_uk_mmt4d
// UKERNEL-LABEL: llvm.func @extend_producer_dispatch_{{[0-9]+}}_mmt4d_
// UKERNEL:         llvm.call @iree_uk_mmt4d
// UKERNEL-LABEL: llvm.func @multi_m_relu_dispatch_{{[0-9]+}}_mmt4d_
// UKERNEL:         llvm.call @iree_uk_mmt4d

// CHECK-LABEL: util.func public @dynamic_m_bias
// CHECK: %[[MM:.*]] = linalg.mmt4d
// CHECK-SAME: -> tensor<8x?x8x8xf32>
// CHECK-NOT: flow.return
// CHECK: %[[EPILOGUE:.*]] = linalg.generic
// CHECK-SAME: ins(%[[MM]]
// CHECK: arith.addf
// CHECK-NOT: flow.return
// CHECK: %[[UNPACK:.*]] = linalg.unpack %[[EPILOGUE]]
// CHECK-SAME: tensor<8x?x8x8xf32> -> tensor<?x64xf32>
// CHECK-NEXT: iree_tensor_ext.dispatch.tensor.store %[[UNPACK]]
func.func @dynamic_m_bias(%lhs: tensor<?x128xf32>, %rhs: tensor<128x64xf32>, %bias: tensor<64xf32>) -> tensor<?x64xf32> {
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %m = tensor.dim %lhs, %c0 : tensor<?x128xf32>
  %empty = tensor.empty(%m) : tensor<?x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x64xf32>) -> tensor<?x64xf32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<?x128xf32>, tensor<128x64xf32>) outs(%init : tensor<?x64xf32>) -> tensor<?x64xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (n)>, affine_map<(m, n) -> (m, n)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%mm, %bias : tensor<?x64xf32>, tensor<64xf32>) outs(%empty : tensor<?x64xf32>) {
  ^bb0(%value: f32, %b: f32, %unused: f32):
    %sum = arith.addf %value, %b : f32
    linalg.yield %sum : f32
  } -> tensor<?x64xf32>
  return %result : tensor<?x64xf32>
}

// CHECK-LABEL: util.func public @odd_relu
// CHECK: %[[MM:.*]] = linalg.mmt4d
// CHECK-SAME: -> tensor<9x9x8x8xf32>
// CHECK-NOT: flow.return
// CHECK: %[[EPILOGUE:.*]] = linalg.generic
// CHECK-SAME: ins(%[[MM]]
// CHECK: arith.maximumf
// CHECK-NOT: flow.return
// CHECK: %[[UNPACK:.*]] = linalg.unpack %[[EPILOGUE]]
// CHECK-SAME: tensor<9x9x8x8xf32> -> tensor<65x67xf32>
// CHECK-NEXT: iree_tensor_ext.dispatch.tensor.store %[[UNPACK]]
func.func @odd_relu(%lhs: tensor<65x129xf32>, %rhs: tensor<129x67xf32>) -> tensor<65x67xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<65x67xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<65x67xf32>) -> tensor<65x67xf32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<65x129xf32>, tensor<129x67xf32>) outs(%init : tensor<65x67xf32>) -> tensor<65x67xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m, n)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%mm : tensor<65x67xf32>) outs(%empty : tensor<65x67xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %max = arith.maximumf %value, %zero : f32
    linalg.yield %max : f32
  } -> tensor<65x67xf32>
  return %result : tensor<65x67xf32>
}

// CHECK-LABEL: util.func public @batched_relu
// CHECK: %[[MM:.*]] = linalg.batch_mmt4d
// CHECK-SAME: -> tensor<2x8x8x8x8xf32>
// CHECK-NOT: flow.return
// CHECK: %[[EPILOGUE:.*]] = linalg.generic
// CHECK-SAME: ins(%[[MM]]
// CHECK: arith.maximumf
// CHECK-NOT: flow.return
// CHECK: %[[UNPACK:.*]] = linalg.unpack %[[EPILOGUE]]
// CHECK-SAME: tensor<2x8x8x8x8xf32> -> tensor<2x64x64xf32>
// CHECK-NEXT: iree_tensor_ext.dispatch.tensor.store %[[UNPACK]]
func.func @batched_relu(%lhs: tensor<2x64x128xf32>, %rhs: tensor<2x128x64xf32>) -> tensor<2x64x64xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x64x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>
  %mm = linalg.batch_matmul ins(%lhs, %rhs : tensor<2x64x128xf32>, tensor<2x128x64xf32>) outs(%init : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>, affine_map<(b, m, n) -> (b, m, n)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%mm : tensor<2x64x64xf32>) outs(%empty : tensor<2x64x64xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %max = arith.maximumf %value, %zero : f32
    linalg.yield %max : f32
  } -> tensor<2x64x64xf32>
  return %result : tensor<2x64x64xf32>
}

// The widening producer receives the packed input layout. The normal dispatch
// fusion policy keeps this producer separate; the contraction still owns its
// epilogue and unpack.
// CHECK-LABEL: util.func public @extend_producer
// CHECK: %[[WIDE:.*]] = linalg.generic
// CHECK-SAME: ins(%{{.*}} : tensor<8192xf16>)
// CHECK: arith.extf
// CHECK: arith.mulf
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[WIDE]]
// CHECK: flow.return
// CHECK: %[[MM:.*]] = linalg.mmt4d
// CHECK-NOT: flow.return
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[MM]]
// CHECK: arith.maximumf
// CHECK-NOT: flow.return
// CHECK: linalg.unpack
func.func @extend_producer(%lhs: tensor<64x128xf16>, %rhs: tensor<128x64xf32>) -> tensor<64x64xf32> {
  %zero = arith.constant 0.0 : f32
  %scale = arith.constant 1.5 : f32
  %wide_empty = tensor.empty() : tensor<64x128xf32>
  %wide = linalg.generic {
    indexing_maps = [affine_map<(m, k) -> (m, k)>, affine_map<(m, k) -> (m, k)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%lhs : tensor<64x128xf16>) outs(%wide_empty : tensor<64x128xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = arith.extf %value : f16 to f32
    %scaled = arith.mulf %cast, %scale : f32
    linalg.yield %scaled : f32
  } -> tensor<64x128xf32>
  %empty = tensor.empty() : tensor<64x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %mm = linalg.matmul ins(%wide, %rhs : tensor<64x128xf32>, tensor<128x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m, n)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%mm : tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %max = arith.maximumf %value, %zero : f32
    linalg.yield %max : f32
  } -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// Normalization adds an expand after the collapsed contraction. Sink it through
// the epilogue before encoding propagation, then restore the public ABI shape.
// CHECK-LABEL: util.func public @multi_m_relu
// CHECK: %[[MM:.*]] = linalg.mmt4d
// CHECK-SAME: -> tensor<1x1x4x8xf32>
// CHECK-NOT: flow.return
// CHECK: %[[RELU:.*]] = linalg.generic
// CHECK-SAME: ins(%[[MM]] : tensor<1x1x4x8xf32>)
// CHECK: arith.maximumf
// CHECK-NOT: flow.return
// CHECK: %[[UNPACK:.*]] = linalg.unpack %[[RELU]]
// CHECK-SAME: tensor<1x1x4x8xf32> -> tensor<6x4xf32>
// CHECK-NEXT: iree_tensor_ext.dispatch.tensor.store %[[UNPACK]]
// CHECK: flow.tensor.reshape
// CHECK-SAME: tensor<6x4xf32> -> tensor<2x3x4xf32>
func.func @multi_m_relu(%a: tensor<2x3x8xf32>, %b: tensor<8x4xf32>) -> tensor<2x3x4xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x3x4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %mm = linalg.generic {
    indexing_maps = [affine_map<(m0, m1, n, k) -> (m0, m1, k)>, affine_map<(m0, m1, n, k) -> (k, n)>, affine_map<(m0, m1, n, k) -> (m0, m1, n)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<2x3x8xf32>, tensor<8x4xf32>) outs(%init : tensor<2x3x4xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x3x4xf32>
  %relu = linalg.generic {
    indexing_maps = [affine_map<(a,b,c) -> (a,b,c)>, affine_map<(a,b,c) -> (a,b,c)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%mm : tensor<2x3x4xf32>) outs(%empty : tensor<2x3x4xf32>) {
  ^bb0(%v: f32, %unused: f32):
    %max = arith.maximumf %v, %zero : f32
    linalg.yield %max : f32
  } -> tensor<2x3x4xf32>
  return %relu : tensor<2x3x4xf32>
}
