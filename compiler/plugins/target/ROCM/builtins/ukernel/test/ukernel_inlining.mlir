// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-compile --iree-hal-target-device=hip --iree-rocm-target=gfx942 \
// RUN:   --iree-rocm-enable-ukernels=all \
// RUN:   --iree-hal-dump-executable-intermediates-to=%t %s -o /dev/null
// RUN: cat %t/*.linked.ll | FileCheck %s --check-prefix=LINKED
// RUN: cat %t/*.optimized.ll | FileCheck %s --check-prefix=OPT

// Regression test for the linked-in bitcode getting inlined into the kernel.
//
// Both the AMDGPU ukernels and the ROCm device library (ocml/ockl) are
// compiled by clang, which leaves per-function "target-cpu"/"target-features"
// attributes behind. The kernels IREE generates carry no such attributes, and
// TargetTransformInfoImplBase::areInlineCompatible - which AMDGPU's TTI falls
// through to - compares them as literal attribute values rather than as
// resolved subtargets. So a callee that carries them is never considered
// inline-compatible, even though it resolves to exactly the same subtarget via
// the target machine, and even when marked `alwaysinline`.

// First check that the ukernel and the device library are really being linked
// in, so that this cannot silently pass by way of them never being selected.
// LINKED: call {{.*}}@iree_uk_amdgpu_argmax_f32i64(
// LINKED: call {{.*}}@__ockl_

// Then check that neither survived as a call, i.e. both were inlined.
// OPT: define {{.*}}@argmax_1d_f32i64_dispatch_0
// OPT-NOT: call {{.*}}@iree_uk_amdgpu_
// OPT-NOT: call {{.*}}@__ockl_

func.func @argmax_1d_f32i64(%arg0: tensor<1x?xf32>) -> tensor<1x1xi64> {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<1x?xf32>)
      outs(%3, %1 : tensor<1xf32>, tensor<1xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<1xf32>, tensor<1xi64>)
  %expanded = tensor.expand_shape %4#1 [[0, 1]] output_shape [1, 1]
      : tensor<1xi64> into tensor<1x1xi64>
  return %expanded : tensor<1x1xi64>
}
