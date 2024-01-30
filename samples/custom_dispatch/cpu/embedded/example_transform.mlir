// RUN: iree-compile %s \
// RUN:     --iree-hal-executable-object-search-path=$IREE_BINARY_DIR \
// RUN:     --iree-preprocessing-transform-spec-filename=%p/example_transform_spec.mlir | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --module=- \
// RUN:     --function=mixed_invocation \
// RUN:     --input=5xf32=7 \
// RUN:     --input=5xf32=4 \
// RUN:     --input=10xf32=-4 \
// RUN:     --input=10xf32=3 | \
// RUN: FileCheck %s

// The configuration used for executable compilation.
// This lets the compiler and runtime know the format and requirements of the
// executable binaries produced and multiple variants with differing formats
// and compilation options (architectures, etc) can be embedded for runtime
// selection.
#x86_64_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 32 : index,
  target_triple = "x86_64-none-elf"
}>

// The target devices that the program will run on. We can compile and run with
// multiple targets, but this example is maintaining an implicit requirement
// that the custom kernel being spliced in is supported by the target device,
// hence we only support llvm-cpu here.
#cpu_target = #hal.device.target<"llvm-cpu", {
  executable_targets = [
    #x86_64_target
  ]
}>

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module @example attributes {hal.device.targets = [#cpu_target]} {

  // CHECK-LABEL: EXEC @mixed_invocation
  func.func @mixed_invocation(%lhs: tensor<?xf32>,
                              %rhs: tensor<?xf32>,
                              %lhs_static: tensor<10xf32>,
                              %rhs_static: tensor<10xf32>) -> (tensor<?xf32>, tensor<10xf32>) {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %lhs, %c0 : tensor<?xf32>
    %empty = tensor.empty(%dim) : tensor<?xf32>
    %max = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                            affine_map<(d0) -> (d0)>,
                                            affine_map<(d0) -> (d0)>],
                           iterator_types = ["parallel"]}
                           ins(%lhs, %rhs : tensor<?xf32>, tensor<?xf32>)
                           outs(%empty : tensor<?xf32>) {
    ^bb0(%in: f32, %in0: f32, %out: f32):
      %m = arith.mulf %in, %in0 : f32
      linalg.yield %m : f32
    } -> tensor<?xf32>
    %abs = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                            affine_map<(d0) -> (d0)>],
                           iterator_types = ["parallel"]}
                           ins(%max : tensor<?xf32>)
                           outs(%empty : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %a = math.absf %in : f32
      linalg.yield %a : f32
    } -> tensor<?xf32>
    %neg = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                            affine_map<(d0) -> (d0)>],
                           iterator_types = ["parallel"]}
                           ins(%abs : tensor<?xf32>)
                           outs(%empty : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %n = arith.negf %in : f32
      linalg.yield %n : f32
    } -> tensor<?xf32>

    %empty_static = tensor.empty() : tensor<10xf32>
    %max_static = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                            affine_map<(d0) -> (d0)>,
                                            affine_map<(d0) -> (d0)>],
                           iterator_types = ["parallel"]}
                           ins(%lhs_static, %rhs_static : tensor<10xf32>, tensor<10xf32>)
                           outs(%empty_static : tensor<10xf32>) {
    ^bb0(%in: f32, %in0: f32, %out: f32):
      %m = arith.mulf %in, %in0 : f32
      linalg.yield %m : f32
    } -> tensor<10xf32>
    %abs_static = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                            affine_map<(d0) -> (d0)>],
                           iterator_types = ["parallel"]}
                           ins(%max_static : tensor<10xf32>)
                           outs(%empty_static : tensor<10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %a = math.absf %in : f32
      linalg.yield %a : f32
    } -> tensor<10xf32>
    %neg_static = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                            affine_map<(d0) -> (d0)>],
                           iterator_types = ["parallel"]}
                           ins(%abs_static : tensor<10xf32>)
                           outs(%empty_static : tensor<10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %n = arith.negf %in : f32
      linalg.yield %n : f32
    } -> tensor<10xf32>

    // Add 1 to show that it actually runs the custom kernel.
    // CHECK: 5xf32=-27 -27 -27 -27 -27
    // CHECK: 10xf32=-11 -11 -11 -11 -11 -11 -11 -11 -11 -11
    return %neg, %neg_static : tensor<?xf32>, tensor<10xf32>
  }
}  // module
