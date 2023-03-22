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

// CHECK-DAG:#[[MAP:.+]] = affine_map<(d0) -> (d0 * 128)>
// CHECK-DAG:#[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:#[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:#[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG:#[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG:        func.func @_attention_dispatch_0() {
// CHECK-DAG:          %[[CST:.+]] = arith.constant dense<-1.000000e+30> : vector<1x128xf32>
// CHECK-DAG:          %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : vector<1x128xf32>
// CHECK-DAG:          %[[CST_1:.+]] = arith.constant dense<0.000000e+00> : vector<32x128xf32>
// CHECK-DAG:          %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:          %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:          %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:          %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:          %[[CST_2:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:              %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:           offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf32>
// CHECK:              memref.assume_alignment %[[D0]], 64 : memref<192x1024x64xf32>
// CHECK:              %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:           offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf32>
// CHECK:              memref.assume_alignment %[[D1]], 64 : memref<192x1024x64xf32>
// CHECK:              %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:           offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf32>
// CHECK:              memref.assume_alignment %[[D2]], 64 : memref<192x1024x64xf32>
// CHECK:              %[[D3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64)
// CHECK-SAME:           offset(%[[C0]]) : memref<192x1024x64xf32>
// CHECK:              memref.assume_alignment %[[D3]], 64 : memref<192x1024x64xf32>
// CHECK:              scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) in (192, 8) {
// CHECK-DAG:            %[[D4:.+]] = affine.apply #[[MAP]](%[[ARG1]])
// CHECK:                %[[SUBVIEW:.+]] = memref.subview %[[D3]][%[[ARG0]], %[[D4]], 0] [1, 128, 64] [1, 1, 1] :
// CHECK-SAME:             memref<192x1024x64xf32> to memref<1x128x64xf32, strided<[65536, 64, 1], offset: ?>>
// CHECK:                %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32,
// CHECK-SAME:             #[[GPU:.+]].address_space<workgroup>>
// CHECK:                %[[ALLOC_3:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32,
// CHECK-SAME:             #[[GPU]].address_space<workgroup>>
// CHECK:                vector.transfer_write %[[CST]], %[[ALLOC_3]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} :
// CHECK-SAME:             vector<1x128xf32>, memref<1x128xf32, #[[GPU]].address_space<workgroup>>
// CHECK:                vector.transfer_write %[[CST_0]], %[[ALLOC]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} :
// CHECK-SAME:             vector<1x128xf32>, memref<1x128xf32, #[[GPU]].address_space<workgroup>>
// CHECK:                scf.for %[[ARG2:.+]] = %[[C0]] to %[[C1024]] step %[[C128]] {
// CHECK:                  scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C128]] step %[[C32]] {
// CHECK:                    %[[D5:.+]] = arith.addi %[[ARG3]], %[[D4]] : index
// CHECK:                    %[[D6:.+]] = vector.transfer_read %[[D0]][%[[ARG0]], %[[D5]], %[[C0]]], %[[CST_2]]
// CHECK-SAME:                 {in_bounds = [true, true]} : memref<192x1024x64xf32>, vector<32x64xf32>
// CHECK:                    %[[D7:.+]] = vector.transfer_read %[[D1]][%[[ARG0]], %[[ARG2]], %[[C0]]], %[[CST_2]]
// CHECK-SAME:                 {in_bounds = [true, true]} : memref<192x1024x64xf32>, vector<128x64xf32>
// CHECK:                    %[[D8:.+]] = vector.contract {indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]],
// CHECK-SAME:                 iterator_types = ["parallel", "parallel", "reduction"], kind = #[[VECTOR:.+]].kind<add>}
// CHECK-SAME:                 %[[D6]], %[[D7]], %[[CST_1]] : vector<32x64xf32>, vector<128x64xf32> into
// CHECK-SAME:                 vector<32x128xf32>
// CHECK:                    %[[D9:.+]] = vector.transfer_read %[[ALLOC_3]][%[[C0]], %[[ARG3]]], %[[CST_2]] {in_bounds =
// CHECK-SAME:                 [true]} : memref<1x128xf32, #[[GPU]].address_space<workgroup>>, vector<32xf32>
// CHECK:                    %[[D10:.+]] = vector.multi_reduction <maxf>, %[[D8]], %[[D9]] [1] : vector<32x128xf32> to
// CHECK-SAME:                 vector<32xf32>
// CHECK:                    %[[D11:.+]] = vector.broadcast %[[D10]] : vector<32xf32> to vector<128x32xf32>
// CHECK:                    %[[D12:.+]] = vector.transpose %[[D11]], [1, 0] : vector<128x32xf32> to vector<32x128xf32>
// CHECK:                    %[[D13:.+]] = arith.subf %[[D8]], %[[D12]] : vector<32x128xf32>
// CHECK:                    %[[D14:.+]] = math.exp %[[D13]] : vector<32x128xf32>
// CHECK:                    %[[D15:.+]] = vector.transfer_read %[[ALLOC]][%[[C0]], %[[ARG3]]], %[[CST_2]] {in_bounds =
// CHECK-SAME:                 [true]} : memref<1x128xf32, #[[GPU]].address_space<workgroup>>, vector<32xf32>
// CHECK:                    %[[D16:.+]] = arith.subf %[[D9]], %[[D10]] : vector<32xf32>
// CHECK:                    %[[D17:.+]] = math.exp %[[D16]] : vector<32xf32>
// CHECK:                    %[[D18:.+]] = arith.mulf %[[D17]], %[[D15]] : vector<32xf32>
// CHECK:                    %[[D19:.+]] = vector.multi_reduction <add>, %[[D14]], %[[D18]] [1] : vector<32x128xf32> to
// CHECK-SAME:                 vector<32xf32>
// CHECK:                    %[[D20:.+]] = vector.broadcast %[[D19]] : vector<32xf32> to vector<128x32xf32>
// CHECK:                    %[[D21:.+]] = vector.transpose %[[D20]], [1, 0] : vector<128x32xf32> to vector<32x128xf32>
// CHECK:                    %[[D22:.+]] = arith.divf %[[D14]], %[[D21]] : vector<32x128xf32>
// CHECK:                    %[[D23:.+]] = vector.transfer_read %[[SUBVIEW]][%[[C0]], %[[ARG3]], %[[C0]]], %[[CST_2]]
// CHECK-SAME:                 {in_bounds = [true, true]} : memref<1x128x64xf32, strided<[65536, 64, 1], offset: ?>>,
// CHECK-SAME:                 vector<32x64xf32>
// CHECK:                    %[[D24:.+]] = vector.broadcast %[[D18]] : vector<32xf32> to vector<64x32xf32>
// CHECK:                    %[[D25:.+]] = vector.broadcast %[[D19]] : vector<32xf32> to vector<64x32xf32>
// CHECK:                    %[[D26:.+]] = arith.divf %[[D24]], %[[D25]] : vector<64x32xf32>
// CHECK:                    %[[D27:.+]] = vector.transpose %[[D26]], [1, 0] : vector<64x32xf32> to vector<32x64xf32>
// CHECK:                    %[[D28:.+]] = arith.mulf %[[D27]], %[[D23]] : vector<32x64xf32>
// CHECK:                    %[[D29:.+]] = vector.transfer_read %[[D2]][%[[ARG0]], %[[ARG2]], %[[C0]]], %[[CST_2]]
// CHECK-SAME:                 {in_bounds = [true, true]} : memref<192x1024x64xf32>, vector<128x64xf32>
// CHECK:                    %[[D30:.+]] = vector.contract {indexing_maps = [#[[MAP1]], #[[MAP4]], #[[MAP3]]],
// CHECK-SAME:                 iterator_types = ["parallel", "parallel", "reduction"], kind = #[[VECTOR]].kind<add>}
// CHECK-SAME:                 %[[D22]], %[[D29]], %[[D28]] : vector<32x128xf32>, vector<128x64xf32> into
// CHECK-SAME:                 vector<32x64xf32>
// CHECK:                    vector.transfer_write %[[D30]], %[[SUBVIEW]][%[[ARG3]], %[[C0]], %[[C0]]] {in_bounds =
// CHECK-SAME:                 [true, true]} : vector<32x64xf32>, memref<1x128x64xf32, strided<[65536, 64, 1], offset:
// CHECK-SAME:                 ?>>
// CHECK:                    vector.transfer_write %[[D10]], %[[ALLOC_3]][%[[C0]], %[[ARG3]]] {in_bounds = [true]} :
// CHECK-SAME:                 vector<32xf32>, memref<1x128xf32, #[[GPU]].address_space<workgroup>>
// CHECK:                    vector.transfer_write %[[D19]], %[[ALLOC]][%[[C0]], %[[ARG3]]] {in_bounds = [true]} :
// CHECK-SAME:                 vector<32xf32>, memref<1x128xf32, #[[GPU]].address_space<workgroup>>
// CHECK:                  }
// CHECK:                }
// CHECK:                memref.dealloc %[[ALLOC]] : memref<1x128xf32, #[[GPU]].address_space<workgroup>>
// CHECK:                memref.dealloc %[[ALLOC_3]] : memref<1x128xf32, #[[GPU]].address_space<workgroup>>
// CHECK:              } {mapping = [#[[GPU]].block<x>, #[[GPU]].block<y>]}
// CHECK:              return
// CHECK:            }
// CHECK:          }
// CHECK:        }
// CHECK:      }
