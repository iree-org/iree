// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

hal.executable @conv2d_nchw_fchw {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @conv2d_nchw_fchw ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @conv2d_nchw_fchw() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x16x130x130xf16>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x16x3x3xf16>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x32x128x128xf16>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 16, 130, 130], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x16x130x130xf16>> -> tensor<2x16x130x130xf16>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [32, 16, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x16x3x3xf16>> -> tensor<32x16x3x3xf16>
      %5 = tensor.empty() : tensor<2x32x128x128xf16>
      %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x32x128x128xf16>) -> tensor<2x32x128x128xf16>
      %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x16x130x130xf16>, tensor<32x16x3x3xf16>) outs(%6 : tensor<2x32x128x128xf16>) -> tensor<2x32x128x128xf16>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 32, 128, 128], strides = [1, 1, 1, 1] : tensor<2x32x128x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x32x128x128xf16>>
      return
    }
  }
}
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %img2col_tensor, %transformed = transform.iree.convert_conv2d_to_img2col_and_adjust_workgroup_count_region %0 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  }
}

// CHECK:    #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK:    hal.executable.export public @conv2d_nchw_fchw
// CHECK:      ^bb0(%[[ARG0:.+]]: !hal.device, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index, %[[ARG4:.+]]: index, %[[ARG5:.+]]: index, %[[ARG6:.+]]: index, %[[ARG7:.+]]: index):
// CHECK:        %[[COLLAPSE:.+]] = affine.apply #[[MAP]]()[%[[ARG3]], %[[ARG4]]]
// CHECK:        %[[X:.+]], %[[Y:.+]], %[[Z:.+]] = flow.dispatch.workgroup_count_from_dag_root %[[ARG1]], %[[ARG2]], %[[COLLAPSE]], %[[ARG5]], %[[ARG6]], %[[ARG7]]
// CHECK:        hal.return %[[X]], %[[Y]], %[[Z]] : index, index, index

// -----

hal.executable @conv2d_nhwc_hwcf {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @conv2d_nhwc_hwcf ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @conv2d_nhwc_hwcf() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x130x130x16xf16>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x16x32xf16>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x128x128x32xf16>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 130, 130, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x130x130x16xf16>> -> tensor<2x130x130x16xf16>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 16, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x16x32xf16>> -> tensor<3x3x16x32xf16>
      %5 = tensor.empty() : tensor<2x128x128x32xf16>
      %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x128x128x32xf16>) -> tensor<2x128x128x32xf16>
      %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x130x130x16xf16>, tensor<3x3x16x32xf16>) outs(%6 : tensor<2x128x128x32xf16>) -> tensor<2x128x128x32xf16>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 32], strides = [1, 1, 1, 1] : tensor<2x128x128x32xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x128x128x32xf16>>
      return
    }
  }
}
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %img2col_tensor, %transformed = transform.iree.convert_conv2d_to_img2col_and_adjust_workgroup_count_region %0 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  }
}

// CHECK:    #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK:    hal.executable.export public @conv2d_nhwc_hwcf
// CHECK:      ^bb0(%[[ARG0:.+]]: !hal.device, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index, %[[ARG4:.+]]: index, %[[ARG5:.+]]: index, %[[ARG6:.+]]: index, %[[ARG7:.+]]: index):
// CHECK:        %[[COLLAPSE:.+]] = affine.apply #[[MAP]]()[%[[ARG2]], %[[ARG3]]]
// CHECK:        %[[X:.+]], %[[Y:.+]], %[[Z:.+]] = flow.dispatch.workgroup_count_from_dag_root %[[ARG1]], %[[COLLAPSE]], %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]]
// CHECK:        hal.return %[[X]], %[[Y]], %[[Z]] : index, index, index
