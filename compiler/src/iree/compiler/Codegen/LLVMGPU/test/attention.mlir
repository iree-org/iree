// RUN: iree-opt %s --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-transform-dialect-interpreter)))' \
// RUN:   --iree-codegen-transform-dialect-library=%p/attention_transform_spec.mlir| \
// RUN: FileCheck --check-prefix=CHECK %s

hal.executable @_attention_dispatch_0 {
  hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>) {
    hal.executable.export public @_attention_dispatch_0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @_attention_dispatch_0() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<192x1024x64xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [192, 1024, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>> -> tensor<192x1024x64xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [192, 1024, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>> -> tensor<192x1024x64xf16>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [192, 1024, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>> -> tensor<192x1024x64xf16>
        %7 = tensor.empty() : tensor<192x1024x64xf16>
        %8 = iree_linalg_ext.attention ins(%4, %5, %6 : tensor<192x1024x64xf16>, tensor<192x1024x64xf16>, tensor<192x1024x64xf16>) outs(%7 : tensor<192x1024x64xf16>) -> tensor<192x1024x64xf16>
        flow.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [192, 1024, 64], strides = [1, 1, 1] : tensor<192x1024x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<192x1024x64xf16>>
        return
      }
    }
  }
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 128)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<()[s0, s1, s2] -> (s2 * 32 + ((s0 + s1 * 4) floordiv 32) * 32)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK:      func.func @_attention_dispatch_0() {
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<32x64xf32>
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<-1.000000e+30> : vector<32xf32>
// CHECK-DAG:    %[[CST_1:.+]] = arith.constant dense<0.000000e+00> : vector<32xf32>
// CHECK-DAG:    %[[CST_2:.+]] = arith.constant dense<0.000000e+00> : vector<32x128xf32>
// CHECK-DAG:    %[[CST_3:.+]] = arith.constant dense<1.000000e+00> : vector<64x32xf32>
// CHECK-DAG:    %[[CST_4:.+]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:    %[[CST_5:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        %[[D3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        memref.assume_alignment %[[D3]], 64 : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK:        %[[WORKGROUP_ID_Y:.+]] = hal.interface.workgroup.id[1] : index
// CHECK-DAG:    %[[D4:.+]] = affine.apply #[[MAP]]()[%[[WORKGROUP_ID_Y]]]
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[D0]][%[[WORKGROUP_ID_X]], %[[D4]], 0] [1, 128, 64] [1, 1, 1] :
// CHECK-SAME:     memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>> to memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
// CHECK:        %[[SUBVIEW_6:.+]] = memref.subview %[[D3]][%[[WORKGROUP_ID_X]], %[[D4]], 0] [1, 128, 64] [1, 1, 1] :
// CHECK-SAME:     memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>> to memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
// CHECK:        %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU:.+]].address_space<workgroup>>
// CHECK:        gpu.barrier
// CHECK:        linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel",
// CHECK-SAME:     "parallel"]} ins(%[[SUBVIEW]] : memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
// CHECK-SAME:     outs(%[[ALLOC]] : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>>) {
// CHECK:        ^bb0(%[[IN:.+]]: f16, %[[OUT:.+]]: f16):
// CHECK:          linalg.yield %[[IN]] : f16
// CHECK:        }
// CHECK:        gpu.barrier
// CHECK:        %[[ALLOC_7:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        gpu.barrier
// CHECK:        linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel",
// CHECK-SAME:     "parallel"]} ins(%[[SUBVIEW_6]] : memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
// CHECK-SAME:     outs(%[[ALLOC_7]] : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>>) {
// CHECK:        ^bb0(%[[IN:.+]]: f16, %[[OUT:.+]]: f16):
// CHECK:          linalg.yield %[[IN]] : f16
// CHECK:        }
// CHECK:        gpu.barrier
// CHECK-DAG:    %[[D5:.+]] = gpu.thread_id  x
// CHECK-DAG:    %[[D6:.+]] = gpu.thread_id  y
// CHECK-DAG:    %[[D7:.+]] = gpu.thread_id  z
// CHECK-DAG:    %[[D8:.+]] = affine.apply #[[MAP2]]()[%[[D5]], %[[D6]], %[[D7]]]
// CHECK:        gpu.barrier
// CHECK:        gpu.barrier
// CHECK:        gpu.barrier
// CHECK:        %[[D9:.+]] = vector.transfer_read %[[ALLOC]][%[[C0]], %[[D8]], %[[C0]]], %[[CST_4]] {in_bounds = [true,
// CHECK-SAME:     true]} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>>, vector<32x64xf16>
// CHECK:        %[[D10:.+]] = arith.extf %[[D9]] : vector<32x64xf16> to vector<32x64xf32>
// CHECK:        %[[D11:.+]]:3 = scf.for %[[ARG0:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C128]]
// CHECK-SAME:     iter_args(%[[ARG1:[a-zA-Z0-9_]+]] = %[[CST_0]], %[[ARG2:[a-zA-Z0-9_]+]] = %[[CST_1]],
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]] = %[[CST]]) -> (vector<32xf32>, vector<32xf32>, vector<32x64xf32>) {
// CHECK:          %[[SUBVIEW_8:.+]] = memref.subview %[[D1]][%[[WORKGROUP_ID_X]], %[[ARG0]], 0] [1, 128, 64] [1, 1, 1]
// CHECK-SAME:       : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
// CHECK:          %[[SUBVIEW_9:.+]] = memref.subview %[[D2]][%[[WORKGROUP_ID_X]], %[[ARG0]], 0] [1, 128, 64] [1, 1, 1]
// CHECK-SAME:       : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
// CHECK:          %[[ALLOC_10:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x64xf16,
// CHECK-SAME:       #[[GPU]].address_space<workgroup>>
// CHECK:          gpu.barrier
// CHECK:          linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP3]]], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:       ins(%[[SUBVIEW_8]] : memref<128x64xf16, strided<[64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%[[ALLOC_10]] :
// CHECK-SAME:       memref<128x64xf16, #[[GPU]].address_space<workgroup>>) {
// CHECK:          ^bb0(%[[IN:.+]]: f16, %[[OUT:.+]]: f16):
// CHECK:            linalg.yield %[[IN]] : f16
// CHECK:          }
// CHECK:          gpu.barrier
// CHECK:          %[[ALLOC_11:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x64xf16,
// CHECK-SAME:       #[[GPU]].address_space<workgroup>>
// CHECK:          gpu.barrier
// CHECK:          linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP3]]], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:       ins(%[[SUBVIEW_9]] : memref<128x64xf16, strided<[64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%[[ALLOC_11]] :
// CHECK-SAME:       memref<128x64xf16, #[[GPU]].address_space<workgroup>>) {
// CHECK:          ^bb0(%[[IN:.+]]: f16, %[[OUT:.+]]: f16):
// CHECK:            linalg.yield %[[IN]] : f16
// CHECK:          }
// CHECK:          gpu.barrier
// CHECK:          %[[D13:.+]] = vector.transfer_read %[[ALLOC_10]][%[[C0]], %[[C0]]], %[[CST_4]] {in_bounds = [true,
// CHECK-SAME:       true]} : memref<128x64xf16, #[[GPU]].address_space<workgroup>>, vector<128x64xf16>
// CHECK:          %[[D14:.+]] = arith.extf %[[D13]] : vector<128x64xf16> to vector<128x64xf32>
// CHECK:          %[[D15:.+]] = vector.contract {indexing_maps = [#[[MAP4]], #[[MAP5]], #[[MAP6]]], iterator_types =
// CHECK-SAME:       ["parallel", "parallel", "reduction"], kind = #[[VECTOR:.+]].kind<add>} %[[D10]], %[[D14]],
// CHECK-SAME:       %[[CST_2]] : vector<32x64xf32>, vector<128x64xf32> into vector<32x128xf32>
// CHECK:          %[[D16:.+]] = vector.multi_reduction <maximumf>, %[[D15]], %[[ARG1]] [1] : vector<32x128xf32> to
// CHECK-SAME:       vector<32xf32>
// CHECK:          %[[D17:.+]] = vector.broadcast %[[D16]] : vector<32xf32> to vector<128x32xf32>
// CHECK:          %[[D18:.+]] = vector.transpose %[[D17]], [1, 0] : vector<128x32xf32> to vector<32x128xf32>
// CHECK:          %[[D19:.+]] = arith.subf %[[D15]], %[[D18]] : vector<32x128xf32>
// CHECK:          %[[D20:.+]] = math.exp2 %[[D19]] : vector<32x128xf32>
// CHECK:          %[[D21:.+]] = arith.subf %[[ARG1]], %[[D16]] : vector<32xf32>
// CHECK:          %[[D22:.+]] = math.exp2 %[[D21]] : vector<32xf32>
// CHECK:          %[[D23:.+]] = arith.mulf %[[D22]], %[[ARG2]] : vector<32xf32>
// CHECK:          %[[D24:.+]] = vector.multi_reduction <add>, %[[D20]], %[[D23]] [1] : vector<32x128xf32> to
// CHECK-SAME:       vector<32xf32>
// CHECK:          %[[D29:.+]] = arith.truncf %[[D20]] : vector<32x128xf32> to vector<32x128xf16>
// CHECK:          %[[D31:.+]] = vector.broadcast %[[D22]] : vector<32xf32> to vector<64x32xf32>
// CHECK:          %[[D33:.+]] = vector.transpose %[[D31]], [1, 0] : vector<64x32xf32> to vector<32x64xf32>
// CHECK:          %[[D34:.+]] = arith.mulf %[[D33]], %[[ARG3]] : vector<32x64xf32>
// CHECK:          %[[D35:.+]] = vector.transfer_read %[[ALLOC_11]][%[[C0]], %[[C0]]], %[[CST_4]] {in_bounds = [true,
// CHECK-SAME:       true]} : memref<128x64xf16, #[[GPU]].address_space<workgroup>>, vector<128x64xf16>
// CHECK:          %[[D36:.+]] = arith.extf %[[D29]] : vector<32x128xf16> to vector<32x128xf32>
// CHECK:          %[[D37:.+]] = arith.extf %[[D35]] : vector<128x64xf16> to vector<128x64xf32>
// CHECK:          %[[D38:.+]] = vector.transpose %[[D37]], [1, 0] : vector<128x64xf32> to vector<64x128xf32>
// CHECK:          %[[D39:.+]] = vector.contract {indexing_maps = [#[[MAP4]], #[[MAP5]], #[[MAP6]]], iterator_types =
// CHECK-SAME:       ["parallel", "parallel", "reduction"], kind = #[[VECTOR]].kind<add>} %[[D36]], %[[D38]], %[[D34]] :
// CHECK-SAME:       vector<32x128xf32>, vector<64x128xf32> into vector<32x64xf32>
// CHECK:          gpu.barrier
// CHECK:          scf.yield %[[D16]], %[[D24]], %[[D39]] : vector<32xf32>, vector<32xf32>, vector<32x64xf32>
// CHECK:        }
// CHECK:        %[[DSCALE1:.+]] = vector.broadcast %[[D11]]#1 : vector<32xf32> to vector<64x32xf32>
// CHECK:        %[[DSCALE2:.+]] = arith.divf %[[CST_3]], %[[DSCALE1]] : vector<64x32xf32>
// CHECK:        %[[DSCALE3:.+]] = vector.transpose %[[DSCALE2]], [1, 0] : vector<64x32xf32> to vector<32x64xf32>
// CHECK:        %[[DSCALE4:.+]] = arith.mulf %[[DSCALE3]], %[[D11]]#2 : vector<32x64xf32>
// CHECK:        %[[D12:.+]] = arith.truncf %[[DSCALE4]] : vector<32x64xf32> to vector<32x64xf16>
// CHECK:        vector.transfer_write %[[D12]], %[[ALLOC_7]][%[[C0]], %[[D8]], %[[C0]]] {in_bounds = [true, true]} :
// CHECK-SAME:     vector<32x64xf16>, memref<1x128x64xf16, #[[GPU]].address_space<workgroup>>
// CHECK:        gpu.barrier
// CHECK:        gpu.barrier
// CHECK:        linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel",
// CHECK-SAME:     "parallel"]} ins(%[[ALLOC_7]] : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>>)
// CHECK-SAME:     outs(%[[SUBVIEW_6]] : memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) {
// CHECK:        ^bb0(%[[IN:.+]]: f16, %[[OUT:.+]]: f16):
// CHECK:          linalg.yield %[[IN]] : f16
// CHECK:        }
// CHECK:        gpu.barrier
