### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###

iree_add_all_subdirs()

iree_microbenchmark_suite(
  NAME
   "microbenchmark_llvm-cpu"
  SRCS
    "dynamic_shape_vectorization.mlir"
    "linalg_mmt4d.mlir"
    "linalg_transpose.mlir"
    "stablehlo_conv.mlir"
    "stablehlo_dot.mlir"
    "stablehlo_dot_general.mlir"
    "stablehlo_fft_abs.mlir"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-cpu-features=host"
)

# Only target cuda benchmarks if the backend is enabled.
if(IREE_TARGET_BACKEND_CUDA)
  iree_microbenchmark_suite(
    NAME
    "microbenchmark_cuda"
    SRCS
      "linalg_transpose.mlir"
    FLAGS
      "--iree-hal-target-backends=cuda"
      "--iree-input-type=stablehlo"
  )
endif()
