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
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[MAP7:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK:            func.func @_attention_dispatch_0() {
// CHECK-DAG:          %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<32x128xf32>
// CHECK-DAG:          %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:          %[[CST_0:.+]] = arith.constant -1.000000e+30 : f32
// CHECK-DAG:          %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:          %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:          %[[CST_1:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:          %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:          %[[C1:.+]] = arith.constant 1 : index
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
// CHECK:              %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK:              %[[WORKGROUP_ID_Y:.+]] = hal.interface.workgroup.id[1] : index
// CHECK-DAG:          %[[D4:.+]] = affine.apply #[[MAP]]()[%[[WORKGROUP_ID_Y]]]
// CHECK:              %[[SUBVIEW:.+]] = memref.subview %[[D3]][%[[WORKGROUP_ID_X]], %[[D4]], 0] [1, 128, 64] [1, 1, 1]
// CHECK-SAME:           : memref<192x1024x64xf32> to memref<1x128x64xf32, strided<[65536, 64, 1], offset: ?>>
// CHECK:              %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32,
// CHECK-SAME:           #[[GPU:.+]].address_space<workgroup>>
// CHECK:              %[[ALLOC_2:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32,
// CHECK-SAME:           #[[GPU]].address_space<workgroup>>
// CHECK-DAG:          %[[D5:.+]] = gpu.thread_id  x
// CHECK:              %[[D6:.+]] = arith.divui %[[D5]], %[[C32]] : index
// CHECK:              %[[D7:.+]] = arith.cmpi ult, %[[D6]], %[[C1]] : index
// CHECK:              scf.if %[[D7]] {
// CHECK-DAG:              %[[D10:.+]] = affine.apply #[[MAP1]]()[%[[D6]]]
// CHECK:                %[[SUBVIEW_7:.+]] = memref.subview %[[ALLOC_2]][%[[D10]], 0] [1, 128] [1, 1] :
// CHECK-SAME:             memref<1x128xf32, #[[GPU]].address_space<workgroup>> to memref<1x128xf32, strided<[128, 1],
// CHECK-SAME:             offset: ?>, #[[GPU]].address_space<workgroup>>
// CHECK:                linalg.fill ins(%[[CST_0]] : f32) outs(%[[SUBVIEW_7]] : memref<1x128xf32, strided<[128, 1],
// CHECK-SAME:             offset: ?>, #[[GPU]].address_space<workgroup>>)
// CHECK:              }
// CHECK:              gpu.barrier
// CHECK:              scf.if %[[D7]] {
// CHECK-DAG:              %[[D10]] = affine.apply #[[MAP1]]()[%[[D6]]]
// CHECK:                %[[SUBVIEW_7]] = memref.subview %[[ALLOC]][%[[D10]], 0] [1, 128] [1, 1] : memref<1x128xf32,
// CHECK-SAME:             #[[GPU]].address_space<workgroup>> to memref<1x128xf32, strided<[128, 1], offset: ?>,
// CHECK-SAME:             #[[GPU]].address_space<workgroup>>
// CHECK:                linalg.fill ins(%[[CST_1]] : f32) outs(%[[SUBVIEW_7]] : memref<1x128xf32, strided<[128, 1],
// CHECK-SAME:             offset: ?>, #[[GPU]].address_space<workgroup>>)
// CHECK:              }
// CHECK:              gpu.barrier
// CHECK:              %[[SUBVIEW_3:.+]] = memref.subview %[[ALLOC_2]][0, 0] [1, 128] [1, 1] : memref<1x128xf32,
// CHECK-SAME:           #[[GPU]].address_space<workgroup>> to memref<128xf32, strided<[1]>,
// CHECK-SAME:           #[[GPU]].address_space<workgroup>>
// CHECK:              %[[SUBVIEW_4:.+]] = memref.subview %[[ALLOC]][0, 0] [1, 128] [1, 1] : memref<1x128xf32,
// CHECK-SAME:           #[[GPU]].address_space<workgroup>> to memref<128xf32, strided<[1]>,
// CHECK-SAME:           #[[GPU]].address_space<workgroup>>
// CHECK-DAG:            %[[D8:.+]] = affine.apply #[[MAP1]]()[%[[D6]]]
// CHECK:              %[[D9:.+]] = arith.addi %[[D8]], %[[D4]] : index
// CHECK:              %[[SUBVIEW_5:.+]] = memref.subview %[[SUBVIEW_4]][%[[D8]]] [32] [1] : memref<128xf32,
// CHECK-SAME:           strided<[1]>, #[[GPU]].address_space<workgroup>> to memref<32xf32, strided<[1], offset: ?>,
// CHECK-SAME:           #[[GPU]].address_space<workgroup>>
// CHECK:              %[[SUBVIEW_6:.+]] = memref.subview %[[SUBVIEW]][0, 0, 0] [1, 128, 64] [1, 1, 1] :
// CHECK-SAME:           memref<1x128x64xf32, strided<[65536, 64, 1], offset: ?>> to memref<128x64xf32, strided<[64, 1],
// CHECK-SAME:           offset: ?>>
// CHECK:              scf.for %[[ARG0:.+]] = %[[C0]] to %[[C1024]] step %[[C128]] {
// CHECK:                %[[ALLOC_7:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128xf32,
// CHECK-SAME:             #[[GPU]].address_space<workgroup>>
// CHECK:                linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]]], iterator_types = ["parallel"]}
// CHECK-SAME:             ins(%[[SUBVIEW_3]] : memref<128xf32, strided<[1]>, #[[GPU]].address_space<workgroup>>)
// CHECK-SAME:             outs(%[[ALLOC_7]] : memref<128xf32, #[[GPU]].address_space<workgroup>>) {
// CHECK:                ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                  linalg.yield %[[IN]] : f32
// CHECK:                }
// CHECK:                %[[ALLOC_8:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32,
// CHECK-SAME:             #[[GPU]].address_space<workgroup>>
// CHECK:                %[[SUBVIEW_9:.+]] = memref.subview %[[ALLOC_8]][%[[D8]], 0] [32, 128] [1, 1] :
// CHECK-SAME:             memref<128x128xf32, #[[GPU]].address_space<workgroup>> to memref<32x128xf32, strided<[128,
// CHECK-SAME:             1], offset: ?>, #[[GPU]].address_space<workgroup>>
// CHECK:                %[[D10]] = vector.transfer_read %[[D0]][%[[WORKGROUP_ID_X]], %[[D9]], %[[C0]]], %[[CST_1]]
// CHECK-SAME:             {in_bounds = [true, true]} : memref<192x1024x64xf32>, vector<32x64xf32>
// CHECK:                %[[D11:.+]] = vector.transfer_read %[[D1]][%[[WORKGROUP_ID_X]], %[[ARG0]], %[[C0]]], %[[CST_1]]
// CHECK-SAME:             {in_bounds = [true, true]} : memref<192x1024x64xf32>, vector<128x64xf32>
// CHECK:                %[[D12:.+]] = vector.contract {indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP5]]],
// CHECK-SAME:             iterator_types = ["parallel", "parallel", "reduction"], kind = #[[VECTOR:.+]].kind<add>}
// CHECK-SAME:             %[[D10]], %[[D11]], %[[CST]] : vector<32x64xf32>, vector<128x64xf32> into vector<32x128xf32>
// CHECK:                vector.transfer_write %[[D12]], %[[SUBVIEW_9]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} :
// CHECK-SAME:             vector<32x128xf32>, memref<32x128xf32, strided<[128, 1], offset: ?>,
// CHECK-SAME:             #[[GPU]].address_space<workgroup>>
// CHECK:                gpu.barrier
// CHECK:                %[[SUBVIEW_10:.+]] = memref.subview %[[ALLOC_7]][%[[D8]]] [32] [1] : memref<128xf32,
// CHECK-SAME:             #[[GPU]].address_space<workgroup>> to memref<32xf32, strided<[1], offset: ?>,
// CHECK-SAME:             #[[GPU]].address_space<workgroup>>
// CHECK:                %[[D13:.+]] = vector.transfer_read %[[ALLOC_8]][%[[D8]], %[[C0]]], %[[CST_1]] {in_bounds =
// CHECK-SAME:             [true, true]} : memref<128x128xf32, #[[GPU]].address_space<workgroup>>, vector<32x128xf32>
// CHECK:                %[[D14:.+]] = vector.transfer_read %[[ALLOC_7]][%[[D8]]], %[[CST_1]] {in_bounds = [true]} :
// CHECK-SAME:             memref<128xf32, #[[GPU]].address_space<workgroup>>, vector<32xf32>
// CHECK:                %[[D15:.+]] = vector.multi_reduction <maxf>, %[[D13]], %[[D14]] [1] : vector<32x128xf32> to
// CHECK-SAME:             vector<32xf32>
// CHECK:                vector.transfer_write %[[D15]], %[[SUBVIEW_10]][%[[C0]]] {in_bounds = [true]} : vector<32xf32>,
// CHECK-SAME:             memref<32xf32, strided<[1], offset: ?>, #[[GPU]].address_space<workgroup>>
// CHECK:                gpu.barrier
// CHECK:                %[[D16:.+]] = vector.transfer_read %[[ALLOC_7]][%[[D8]]], %[[CST_1]] {in_bounds = [true]} :
// CHECK-SAME:             memref<128xf32, #[[GPU]].address_space<workgroup>>, vector<32xf32>
// CHECK:                %[[D17:.+]] = vector.broadcast %[[D16]] : vector<32xf32> to vector<128x32xf32>
// CHECK:                %[[D18:.+]] = vector.transpose %[[D17]], [1, 0] : vector<128x32xf32> to vector<32x128xf32>
// CHECK:                %[[D19:.+]] = vector.transfer_read %[[ALLOC_8]][%[[D8]], %[[C0]]], %[[CST_1]] {in_bounds =
// CHECK-SAME:             [true, true]} : memref<128x128xf32, #[[GPU]].address_space<workgroup>>, vector<32x128xf32>
// CHECK:                %[[D20:.+]] = arith.subf %[[D19]], %[[D18]] : vector<32x128xf32>
// CHECK:                %[[D21:.+]] = math.exp %[[D20]] : vector<32x128xf32>
// CHECK:                vector.transfer_write %[[D21]], %[[SUBVIEW_9]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} :
// CHECK-SAME:             vector<32x128xf32>, memref<32x128xf32, strided<[128, 1], offset: ?>,
// CHECK-SAME:             #[[GPU]].address_space<workgroup>>
// CHECK:                gpu.barrier
// CHECK:                %[[D22:.+]] = vector.transfer_read %[[ALLOC_2]][%[[C0]], %[[D8]]], %[[CST_1]] {in_bounds =
// CHECK-SAME:             [true]} : memref<1x128xf32, #[[GPU]].address_space<workgroup>>, vector<32xf32>
// CHECK:                %[[D23:.+]] = vector.transfer_read %[[ALLOC_7]][%[[D8]]], %[[CST_1]] {in_bounds = [true]} :
// CHECK-SAME:             memref<128xf32, #[[GPU]].address_space<workgroup>>, vector<32xf32>
// CHECK:                %[[D24:.+]] = vector.transfer_read %[[SUBVIEW_4]][%[[D8]]], %[[CST_1]] {in_bounds = [true]} :
// CHECK-SAME:             memref<128xf32, strided<[1]>, #[[GPU]].address_space<workgroup>>, vector<32xf32>
// CHECK:                %[[D25:.+]] = arith.subf %[[D22]], %[[D23]] : vector<32xf32>
// CHECK:                %[[D26:.+]] = math.exp %[[D25]] : vector<32xf32>
// CHECK:                %[[D27:.+]] = arith.mulf %[[D26]], %[[D24]] : vector<32xf32>
// CHECK:                vector.transfer_write %[[D27]], %[[SUBVIEW_5]][%[[C0]]] {in_bounds = [true]} : vector<32xf32>,
// CHECK-SAME:             memref<32xf32, strided<[1], offset: ?>, #[[GPU]].address_space<workgroup>>
// CHECK:                gpu.barrier
// CHECK:                %[[ALLOC_11:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128xf32,
// CHECK-SAME:             #[[GPU]].address_space<workgroup>>
// CHECK:                linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]]], iterator_types = ["parallel"]}
// CHECK-SAME:             ins(%[[SUBVIEW_4]] : memref<128xf32, strided<[1]>, #[[GPU]].address_space<workgroup>>)
// CHECK-SAME:             outs(%[[ALLOC_11]] : memref<128xf32, #[[GPU]].address_space<workgroup>>) {
// CHECK:                ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                  linalg.yield %[[IN]] : f32
// CHECK:                }
// CHECK:                %[[SUBVIEW_12:.+]] = memref.subview %[[ALLOC_11]][%[[D8]]] [32] [1] : memref<128xf32,
// CHECK-SAME:             #[[GPU]].address_space<workgroup>> to memref<32xf32, strided<[1], offset: ?>,
// CHECK-SAME:             #[[GPU]].address_space<workgroup>>
// CHECK:                %[[D28:.+]] = vector.transfer_read %[[ALLOC_8]][%[[D8]], %[[C0]]], %[[CST_1]] {in_bounds =
// CHECK-SAME:             [true, true]} : memref<128x128xf32, #[[GPU]].address_space<workgroup>>, vector<32x128xf32>
// CHECK:                %[[D29:.+]] = vector.transfer_read %[[ALLOC_11]][%[[D8]]], %[[CST_1]] {in_bounds = [true]} :
// CHECK-SAME:             memref<128xf32, #[[GPU]].address_space<workgroup>>, vector<32xf32>
// CHECK:                %[[D30:.+]] = vector.multi_reduction <add>, %[[D28]], %[[D29]] [1] : vector<32x128xf32> to
// CHECK-SAME:             vector<32xf32>
// CHECK:                vector.transfer_write %[[D30]], %[[SUBVIEW_12]][%[[C0]]] {in_bounds = [true]} : vector<32xf32>,
// CHECK-SAME:             memref<32xf32, strided<[1], offset: ?>, #[[GPU]].address_space<workgroup>>
// CHECK:                gpu.barrier
// CHECK:                %[[D31:.+]] = vector.transfer_read %[[ALLOC_11]][%[[D8]]], %[[CST_1]] {in_bounds = [true]} :
// CHECK-SAME:             memref<128xf32, #[[GPU]].address_space<workgroup>>, vector<32xf32>
// CHECK:                %[[D32:.+]] = vector.broadcast %[[D31]] : vector<32xf32> to vector<128x32xf32>
// CHECK:                %[[D33:.+]] = vector.transpose %[[D32]], [1, 0] : vector<128x32xf32> to vector<32x128xf32>
// CHECK:                %[[D34:.+]] = vector.transfer_read %[[ALLOC_8]][%[[D8]], %[[C0]]], %[[CST_1]] {in_bounds =
// CHECK-SAME:             [true, true]} : memref<128x128xf32, #[[GPU]].address_space<workgroup>>, vector<32x128xf32>
// CHECK:                %[[D35:.+]] = arith.divf %[[D34]], %[[D33]] : vector<32x128xf32>
// CHECK:                vector.transfer_write %[[D35]], %[[SUBVIEW_9]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} :
// CHECK-SAME:             vector<32x128xf32>, memref<32x128xf32, strided<[128, 1], offset: ?>,
// CHECK-SAME:             #[[GPU]].address_space<workgroup>>
// CHECK:                gpu.barrier
// CHECK:                %[[ALLOC_13:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32,
// CHECK-SAME:             #[[GPU]].address_space<workgroup>>
// CHECK:                linalg.generic {indexing_maps = [#[[MAP6]], #[[MAP6]]], iterator_types = ["parallel",
// CHECK-SAME:             "parallel"]} ins(%[[SUBVIEW_6]] : memref<128x64xf32, strided<[64, 1], offset: ?>>)
// CHECK-SAME:             outs(%[[ALLOC_13]] : memref<128x64xf32, #[[GPU]].address_space<workgroup>>) {
// CHECK:                ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                  linalg.yield %[[IN]] : f32
// CHECK:                }
// CHECK:                %[[SUBVIEW_14:.+]] = memref.subview %[[ALLOC_13]][%[[D8]], 0] [32, 64] [1, 1] :
// CHECK-SAME:             memref<128x64xf32, #[[GPU]].address_space<workgroup>> to memref<32x64xf32, strided<[64, 1],
// CHECK-SAME:             offset: ?>, #[[GPU]].address_space<workgroup>>
// CHECK:                %[[D36:.+]] = vector.transfer_read %[[SUBVIEW]][%[[C0]], %[[D8]], %[[C0]]], %[[CST_1]]
// CHECK-SAME:             {in_bounds = [true, true]} : memref<1x128x64xf32, strided<[65536, 64, 1], offset: ?>>,
// CHECK-SAME:             vector<32x64xf32>
// CHECK:                %[[D37:.+]] = vector.transfer_read %[[SUBVIEW_4]][%[[D8]]], %[[CST_1]] {in_bounds = [true]} :
// CHECK-SAME:             memref<128xf32, strided<[1]>, #[[GPU]].address_space<workgroup>>, vector<32xf32>
// CHECK:                %[[D38:.+]] = vector.broadcast %[[D37]] : vector<32xf32> to vector<64x32xf32>
// CHECK:                %[[D39:.+]] = vector.transfer_read %[[ALLOC_11]][%[[D8]]], %[[CST_1]] {in_bounds = [true]} :
// CHECK-SAME:             memref<128xf32, #[[GPU]].address_space<workgroup>>, vector<32xf32>
// CHECK:                %[[D40:.+]] = vector.broadcast %[[D39]] : vector<32xf32> to vector<64x32xf32>
// CHECK:                %[[D41:.+]] = arith.divf %[[D38]], %[[D40]] : vector<64x32xf32>
// CHECK:                %[[D42:.+]] = vector.transpose %[[D41]], [1, 0] : vector<64x32xf32> to vector<32x64xf32>
// CHECK:                %[[D43:.+]] = arith.mulf %[[D42]], %[[D36]] : vector<32x64xf32>
// CHECK:                %[[D44:.+]] = vector.transfer_read %[[ALLOC_8]][%[[D8]], %[[C0]]], %[[CST_1]] {in_bounds =
// CHECK-SAME:             [true, true]} : memref<128x128xf32, #[[GPU]].address_space<workgroup>>, vector<32x128xf32>
// CHECK:                %[[D45:.+]] = vector.transfer_read %[[D2]][%[[WORKGROUP_ID_X]], %[[ARG0]], %[[C0]]], %[[CST_1]]
// CHECK-SAME:             {in_bounds = [true, true]} : memref<192x1024x64xf32>, vector<128x64xf32>
// CHECK:                %[[D46:.+]] = vector.contract {indexing_maps = [#[[MAP3]], #[[MAP7]], #[[MAP5]]],
// CHECK-SAME:             iterator_types = ["parallel", "parallel", "reduction"], kind = #[[VECTOR]].kind<add>}
// CHECK-SAME:             %[[D44]], %[[D45]], %[[D43]] : vector<32x128xf32>, vector<128x64xf32> into vector<32x64xf32>
// CHECK:                vector.transfer_write %[[D46]], %[[SUBVIEW_14]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} :
// CHECK-SAME:             vector<32x64xf32>, memref<32x64xf32, strided<[64, 1], offset: ?>,
// CHECK-SAME:             #[[GPU]].address_space<workgroup>>
// CHECK:                gpu.barrier
// CHECK:                linalg.generic {indexing_maps = [#[[MAP6]], #[[MAP6]]], iterator_types = ["parallel",
// CHECK-SAME:             "parallel"]} ins(%[[ALLOC_13]] : memref<128x64xf32, #[[GPU]].address_space<workgroup>>)
// CHECK-SAME:             outs(%[[SUBVIEW_6]] : memref<128x64xf32, strided<[64, 1], offset: ?>>) {
// CHECK:                ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                  linalg.yield %[[IN]] : f32
// CHECK:                }
// CHECK:                linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]]], iterator_types = ["parallel"]}
// CHECK-SAME:             ins(%[[ALLOC_7]] : memref<128xf32, #[[GPU]].address_space<workgroup>>) outs(%[[SUBVIEW_3]] :
// CHECK-SAME:             memref<128xf32, strided<[1]>, #[[GPU]].address_space<workgroup>>) {
// CHECK:                ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                  linalg.yield %[[IN]] : f32
// CHECK:                }
// CHECK:                linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]]], iterator_types = ["parallel"]}
// CHECK-SAME:             ins(%[[ALLOC_11]] : memref<128xf32, #[[GPU]].address_space<workgroup>>) outs(%[[SUBVIEW_4]] :
// CHECK-SAME:             memref<128xf32, strided<[1]>, #[[GPU]].address_space<workgroup>>) {
// CHECK:                ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                  linalg.yield %[[IN]] : f32
// CHECK:                }
// CHECK:                memref.dealloc %[[ALLOC_7]] : memref<128xf32, #[[GPU]].address_space<workgroup>>
// CHECK:                memref.dealloc %[[ALLOC_8]] : memref<128x128xf32, #[[GPU]].address_space<workgroup>>
// CHECK:                memref.dealloc %[[ALLOC_11]] : memref<128xf32, #[[GPU]].address_space<workgroup>>
// CHECK:                memref.dealloc %[[ALLOC_13]] : memref<128x64xf32, #[[GPU]].address_space<workgroup>>
// CHECK:              }
// CHECK:              memref.dealloc %[[ALLOC]] : memref<1x128xf32, #[[GPU]].address_space<workgroup>>
// CHECK:              memref.dealloc %[[ALLOC_2]] : memref<1x128xf32, #[[GPU]].address_space<workgroup>>
// CHECK:              return
// CHECK:            }
// CHECK:          }
// CHECK:        }
// CHECK:      }
