func.func @attention(%query: tensor<192x1024x64xf32>, %key: tensor<192x1024x64xf32>, %value: tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32> {
  %0 = tensor.empty() : tensor<192x1024x64xf32>
  %1 = iree_linalg_ext.attention ins(%query, %key, %value : tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>) outs(%0 : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  return %1 : tensor<192x1024x64xf32>
}

// RUN: iree-opt %s --iree-hal-target-backends=cuda \
// RUN:     --iree-abi-transformation-pipeline \
// RUN:     --iree-flow-transformation-pipeline  \
// RUN:     --iree-stream-transformation-pipeline \
// RUN:     --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-codegen-llvmgpu-use-transform-dialect=%p/transform_dialect_codegen_attention_spec.mlir | \
// RUN: FileCheck %s --check-prefix=CHECK

// CHECK-DAG:  #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 128)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK:      func.func @_attention_dispatch_0() {
// CHECK-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:    %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<-1.000000e+30> : tensor<1x128xf32>
// CHECK-DAG:    %[[CST_1:.+]] = arith.constant dense<0.000000e+00> : vector<128x128xf32>
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[CST_2:.+]] = arith.constant dense<0.000000e+00> : tensor<1x128xf32>
// CHECK:        %[[D0:.+]] = bufferization.to_memref %[[CST_2]] : memref<1x128xf32>
// CHECK:        %[[D1:.+]] = bufferization.to_memref %[[CST_0]] : memref<1x128xf32>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf32>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<192x1024x64xf32>
// CHECK:        %[[D3:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf32>
// CHECK:        memref.assume_alignment %[[D3]], 64 : memref<192x1024x64xf32>
// CHECK:        %[[D4:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf32>
// CHECK:        memref.assume_alignment %[[D4]], 64 : memref<192x1024x64xf32>
// CHECK:        %[[D5:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<192x1024x64xf32>
// CHECK:        memref.assume_alignment %[[D5]], 64 : memref<192x1024x64xf32>
// CHECK:        %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK:        %[[WORKGROUP_ID_Y:.+]] = hal.interface.workgroup.id[1] : index
// CHECK-DAG:    %[[D6:.+]] = affine.apply #[[MAP]]()[%[[WORKGROUP_ID_Y]]]
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[D5]][%[[WORKGROUP_ID_X]], %[[D6]], 0] [1, 128, 64] [1, 1, 1]
// CHECK-SAME:     : memref<192x1024x64xf32> to memref<1x128x64xf32, strided<[65536, 64, 1], offset: ?>>
// CHECK:        %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32,
// CHECK-SAME:     #[[GPU:.+]].address_space<workgroup>>
// CHECK:        linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:     "parallel"]} ins(%[[D1]] : memref<1x128xf32>) outs(%[[ALLOC]] : memref<1x128xf32,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          linalg.yield %[[IN]] : f32
// CHECK:        }
// CHECK:        %[[ALLOC_3:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:     "parallel"]} ins(%[[D0]] : memref<1x128xf32>) outs(%[[ALLOC_3]] : memref<1x128xf32,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          linalg.yield %[[IN]] : f32
// CHECK:        }
// CHECK:        %[[D7:.+]] = vector.transfer_read %[[ALLOC]][%[[C0]], %[[C0]]], %[[CST]] {in_bounds = [true]} :
// CHECK-SAME:     memref<1x128xf32, #[[GPU]].address_space<workgroup>>, vector<128xf32>
// CHECK:        %[[D8:.+]] = vector.transfer_read %[[ALLOC_3]][%[[C0]], %[[C0]]], %[[CST]] {in_bounds = [true]} :
// CHECK-SAME:     memref<1x128xf32, #[[GPU]].address_space<workgroup>>, vector<128xf32>
// CHECK:        %[[D9:.+]] = vector.transfer_read %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]], %[[CST]] {in_bounds =
// CHECK-SAME:     [true, true]} : memref<1x128x64xf32, strided<[65536, 64, 1], offset: ?>>, vector<128x64xf32>
// CHECK:        %[[D10:.+]]:3 = scf.for %[[ARG0:.+]] = %[[C0]] to %[[C1024]] step %[[C128]]
// CHECK-SAME:     iter_args(%[[ARG1:.+]] = %[[D7]], %[[ARG2:.+]] = %[[D8]], %[[ARG3:.+]] = %[[D9]]) -> (vector<128xf32>,
// CHECK-SAME:     vector<128xf32>, vector<128x64xf32>) {
// CHECK:          %[[D11:.+]] = vector.transfer_read %[[D2]][%[[WORKGROUP_ID_X]], %[[D6]], %[[C0]]], %[[CST]]
// CHECK-SAME:       {in_bounds = [true, true]} : memref<192x1024x64xf32>, vector<128x64xf32>
// CHECK:          %[[D12:.+]] = vector.transfer_read %[[D3]][%[[WORKGROUP_ID_X]], %[[ARG0]], %[[C0]]], %[[CST]]
// CHECK-SAME:       {in_bounds = [true, true]} : memref<192x1024x64xf32>, vector<128x64xf32>
// CHECK:          %[[D13:.+]] = vector.contract {indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"], kind = #[[VECTOR:.+]].kind<add>}
// CHECK-SAME:       %[[D11]], %[[D12]], %[[CST_1]] : vector<128x64xf32>, vector<128x64xf32> into
// CHECK-SAME:       vector<128x128xf32>
// CHECK:          %[[D14:.+]] = vector.multi_reduction <maxf>, %[[D13]], %[[ARG1]] [1] : vector<128x128xf32> to
// CHECK-SAME:       vector<128xf32>
// CHECK:          %[[D15:.+]] = vector.broadcast %[[D14]] : vector<128xf32> to vector<128x128xf32>
// CHECK:          %[[D16:.+]] = vector.transpose %[[D15]], [1, 0] : vector<128x128xf32> to vector<128x128xf32>
// CHECK:          %[[D17:.+]] = arith.subf %[[D13]], %[[D16]] : vector<128x128xf32>
// CHECK:          %[[D18:.+]] = math.exp %[[D17]] : vector<128x128xf32>
// CHECK:          %[[D19:.+]] = arith.subf %[[ARG1]], %[[D14]] : vector<128xf32>
// CHECK:          %[[D20:.+]] = math.exp %[[D19]] : vector<128xf32>
// CHECK:          %[[D21:.+]] = arith.mulf %[[D20]], %[[ARG2]] : vector<128xf32>
// CHECK:          %[[D22:.+]] = vector.multi_reduction <add>, %[[D18]], %[[D21]] [1] : vector<128x128xf32> to
// CHECK-SAME:       vector<128xf32>
// CHECK:          %[[D23:.+]] = vector.broadcast %[[D22]] : vector<128xf32> to vector<128x128xf32>
// CHECK:          %[[D24:.+]] = vector.transpose %[[D23]], [1, 0] : vector<128x128xf32> to vector<128x128xf32>
// CHECK:          %[[D25:.+]] = arith.divf %[[D18]], %[[D24]] : vector<128x128xf32>
// CHECK:          %[[D26:.+]] = vector.broadcast %[[D21]] : vector<128xf32> to vector<64x128xf32>
// CHECK:          %[[D27:.+]] = vector.broadcast %[[D22]] : vector<128xf32> to vector<64x128xf32>
// CHECK:          %[[D28:.+]] = arith.divf %[[D26]], %[[D27]] : vector<64x128xf32>
// CHECK:          %[[D29:.+]] = vector.transpose %[[D28]], [1, 0] : vector<64x128xf32> to vector<128x64xf32>
// CHECK:          %[[D30:.+]] = arith.mulf %[[D29]], %[[ARG3]] : vector<128x64xf32>
// CHECK:          %[[D31:.+]] = vector.transfer_read %[[D4]][%[[WORKGROUP_ID_X]], %[[ARG0]], %[[C0]]], %[[CST]]
// CHECK-SAME:       {in_bounds = [true, true]} : memref<192x1024x64xf32>, vector<128x64xf32>
// CHECK:          %[[D32:.+]] = vector.contract {indexing_maps = [#[[MAP2]], #[[MAP5]], #[[MAP4]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"], kind = #[[VECTOR]].kind<add>}
// CHECK-SAME:       %[[D25]], %[[D31]], %[[D30]] : vector<128x128xf32>, vector<128x64xf32> into
// CHECK-SAME:       vector<128x64xf32>
// CHECK:          scf.yield %[[D14]], %[[D22]], %[[D32]] : vector<128xf32>, vector<128xf32>, vector<128x64xf32>
// CHECK:        }
// CHECK:        vector.transfer_write %[[D10]]#[[D2:.+]], %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]] {in_bounds =
// CHECK-SAME:     [true, true]} : vector<128x64xf32>, memref<1x128x64xf32, strided<[65536, 64, 1], offset: ?>>
// CHECK:        vector.transfer_write %[[D10]]#[[D1]], %[[ALLOC_3]][%[[C0]], %[[C0]]] {in_bounds = [true]} :
// CHECK-SAME:     vector<128xf32>, memref<1x128xf32, #[[GPU]].address_space<workgroup>>
// CHECK:        vector.transfer_write %[[D10]]#[[D0]], %[[ALLOC]][%[[C0]], %[[C0]]] {in_bounds = [true]} :
// CHECK-SAME:     vector<128xf32>, memref<1x128xf32, #[[GPU]].address_space<workgroup>>
// CHECK:        memref.dealloc %[[ALLOC]] : memref<1x128xf32, #[[GPU]].address_space<workgroup>>
// CHECK:        memref.dealloc %[[ALLOC_3]] : memref<1x128xf32, #[[GPU]].address_space<workgroup>>
// CHECK:        return
