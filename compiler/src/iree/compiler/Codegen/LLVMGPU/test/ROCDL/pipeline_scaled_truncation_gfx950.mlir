// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-configuration-pipeline), iree-codegen-linalg-to-rocdl-pipeline)))" %s | FileCheck %s

// This pipeline-level test exists because there are fragile pattern matches
// needed to generate efficient calls to scaling truncation
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>

#pipeline_layout = #hal.pipeline.layout<constants = 1,
  bindings = [
    #hal.pipeline.binding<storage_buffer>,
    #hal.pipeline.binding<storage_buffer>,
    #hal.pipeline.binding<storage_buffer>
  ]>
hal.executable @fp4_dynamic_quantt {
  hal.executable.variant @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @fp4_dynamic_quant layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @fp4_dynamic_quant() {
        %c0 = arith.constant 0 : index
        %cst_0.25 = arith.constant 2.500000e-01 : f32
        %cst_neg_inf = arith.constant 0xff800000 : f32
        %len_i32 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %len = arith.index_castui %len_i32 : i32 to index
        %input.bind = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x32xf32>>{%len}
        %trunc.bind = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf4E2M1FN>>{%len}
        %scales.bind = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%len}

        %input = iree_tensor_ext.dispatch.tensor.load %input.bind, offsets = [0, 0], sizes = [%len, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x32xf32>>{%len} -> tensor<?x32xf32>
        %abs.max.empty = tensor.empty(%len) : tensor<?xf32>
        %abs.max.init = linalg.fill ins(%cst_neg_inf : f32) outs(%abs.max.empty : tensor<?xf32>) -> tensor<?xf32>
        %abs.max = linalg.generic {indexing_maps = [#map, #map1],
                                   iterator_types = ["parallel", "reduction"]}
                                   ins(%input : tensor<?x32xf32>)
                                   outs(%abs.max.init : tensor<?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %abs = math.absf %in : f32
          %max = arith.maximumf %abs, %out : f32
          linalg.yield %max : f32
        } -> tensor<?xf32>
        %scales.init = tensor.empty(%len) : tensor<?xi8>
        %scales = linalg.generic {indexing_maps = [#map2, #map2],
                                   iterator_types = ["parallel"]}
                                   ins(%abs.max : tensor<?xf32>)
                                   outs(%scales.init : tensor<?xi8>) {
        ^bb0(%in0: f32, %out: i8):
          %normalized = arith.mulf %in0, %cst_0.25 : f32
          %only.exp = arith.truncf %normalized : f32 to f8E8M0FNU
          %scale.byte = arith.bitcast %only.exp : f8E8M0FNU to i8
          linalg.yield %scale.byte : i8
        } -> tensor<?xi8>
        %trunc.empty = tensor.empty(%len) : tensor<?x32xf4E2M1FN>
        %trunc = linalg.generic {indexing_maps = [#map, #map1, #map],
                                   iterator_types = ["parallel", "parallel"]}
                                   ins(%input, %abs.max : tensor<?x32xf32>, tensor<?xf32>)
                                   outs(%trunc.empty : tensor<?x32xf4E2M1FN>) {
        ^bb0(%in1: f32, %scale: f32, %out: f4E2M1FN):
          %normalized2 = arith.mulf %scale, %cst_0.25 : f32
          %scaling.trunc = arith.scaling_truncf %in1, %normalized2 : f32, f32 to f4E2M1FN
          linalg.yield %scaling.trunc : f4E2M1FN
        } -> tensor<?x32xf4E2M1FN>

        iree_tensor_ext.dispatch.tensor.store %scales, %scales.bind, offsets = [0], sizes = [%len], strides = [1] : tensor<?xi8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%len}
        iree_tensor_ext.dispatch.tensor.store %trunc, %trunc.bind, offsets = [0, 0], sizes = [%len, 32], strides = [1, 1] : tensor<?x32xf4E2M1FN> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf4E2M1FN>>{%len}
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable public @fp4_dynamic_quant
// CHECK: hal.executable.variant public @rocm
// (Note to editors: this mainly shouldn't be 1)
// CHECK: workgroup_size = [64 : index
// CHECK: llvm.intr.vector.reduce.fmax
// CHECK-COUNT-4: rocdl.cvt.scalef32.pk.fp4.f32 {{.*}} -> %{{.*}}[3] : i32
// CHECK-NOT: rocdl.cvt.scalef32.pk.fp4.f32
// CHECK: return
