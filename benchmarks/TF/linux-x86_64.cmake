set(LINUX_X86_64_CASCADELAKE_CPU_COMPILATION_FLAGS
  "--iree-input-type=mhlo"
  "--iree-llvm-target-cpu=cascadelake"
  "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
)

# CPU, Dylib-Sync, x86_64, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "linux-x86_64"

  MODULES
    "${MINILM_I32_MODULE}"

  BENCHMARK_MODES
    "full-inference,default-flags"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-x86_64-CascadeLake"
  COMPILATION_FLAGS
    ${LINUX_X86_64_CASCADELAKE_CPU_COMPILATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-dylib-sync"
  DRIVER
    "local-sync"
)
