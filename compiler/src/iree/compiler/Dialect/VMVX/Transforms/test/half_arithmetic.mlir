// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-configuration-preprocessing-pipeline,builtin.module(iree-vmvx-configuration-pipeline,iree-vmvx-transformation-pipeline))))' %s | FileCheck %s

// Half arithmetic must round before storing the original two-byte values.
#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#pipeline_layout_static = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @add_half_odd {
  hal.executable.variant public @vmvx_bytecode_fb target(#executable_target_vmvx_bytecode_fb) {
    hal.executable.export public @add_half_odd ordinal(0) layout(#pipeline_layout_static)
        count(%device: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @add_half_odd() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan layout(#pipeline_layout_static) binding(0)
            alignment(64) offset(%c0) flags(ReadOnly)
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3xf16>>
        %rhs = hal.interface.binding.subspan layout(#pipeline_layout_static) binding(1)
            alignment(64) offset(%c0) flags(ReadOnly)
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3xf16>>
        %out = hal.interface.binding.subspan layout(#pipeline_layout_static) binding(2)
            alignment(64) offset(%c0)
            : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3xf16>>
        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [0], sizes = [3], strides = [1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3xf16>> -> tensor<3xf16>
        %rhs_t = iree_tensor_ext.dispatch.tensor.load %rhs, offsets = [0], sizes = [3], strides = [1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3xf16>> -> tensor<3xf16>
        %empty = tensor.empty() : tensor<3xf16>
        %sum = linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]}
          ins(%lhs_t, %rhs_t : tensor<3xf16>, tensor<3xf16>) outs(%empty : tensor<3xf16>) {
        ^bb0(%a: f16, %b: f16, %o: f16):
          %s = arith.addf %a, %b : f16
          linalg.yield %s : f16
        } -> tensor<3xf16>
        iree_tensor_ext.dispatch.tensor.store %sum, %out, offsets = [0], sizes = [3], strides = [1]
            : tensor<3xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3xf16>>
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable private @add_half_odd
// CHECK: func.func @add_half_odd
// CHECK: arith.bitcast {{.+}} : f16 to i16
// CHECK: arith.addf {{.+}} : f32
// CHECK: arith.trunci {{.+}} : i32 to i16
// CHECK: arith.bitcast {{.+}} : i16 to f16
// CHECK-NOT: arith.addf {{.+}} : f16
