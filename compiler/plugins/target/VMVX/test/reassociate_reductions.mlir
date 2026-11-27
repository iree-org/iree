// RUN: iree-compile %s --iree-hal-target-backends=vmvx --compile-to=executable-configurations -o %t.default
// RUN: FileCheck %s --check-prefix=DEFAULT --input-file=%t.default
// RUN: iree-compile %s --iree-hal-target-backends=vmvx --iree-vmvx-reassociate-fp-reductions --compile-to=executable-configurations -o %t.enabled
// RUN: FileCheck %s --check-prefix=ENABLED --input-file=%t.enabled
// RUN: iree-compile %t.enabled --compile-from=executable-configurations --compile-to=executable-targets -o %t.resumed
// RUN: FileCheck %s --check-prefix=RESUMED --input-file=%t.resumed

// DEFAULT-NOT: reassociate_fp_reductions
// DEFAULT: ukernels = "none"
// DEFAULT-NOT: reassociate_fp_reductions
// ENABLED: reassociate_fp_reductions = true
// ENABLED: ins(%{{.*}} : tensor<?x576xf32>)
// RESUMED: reassociate_fp_reductions = true
// The resumed compiler must recover the policy without a command-line flag.
// The two-stage reduction requires storage for 9 partial sums per row.
// RESUMED: vm.const.i{{32|64}} 9

func.func @sum(%in: tensor<?x576xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%in : tensor<?x576xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %square = arith.mulf %x, %x : f32
    %sum = arith.addf %acc, %square : f32
    linalg.yield %sum : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
