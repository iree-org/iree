# Use CMake to build the sample until bazel can support lld.
# TODO(llvm-bazel/#228): re-enable bazel.

iree_add_all_subdirs()

if(${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT})

iree_cc_binary(
  NAME
    simple_embedding_dylib
  SRCS
    "device_dylib.c"
    "simple_embedding.c"
  DEPS
    ::simple_embedding_test_bytecode_module_dylib_arm_64_c
    ::simple_embedding_test_bytecode_module_dylib_riscv_64_c
    ::simple_embedding_test_bytecode_module_dylib_x86_64_c
    iree::base
    iree::hal
    iree::hal::local
    iree::hal::local::loaders::embedded_library_loader
    iree::hal::local::task_driver
    iree::modules::hal
    iree::task::api
    iree::vm
    iree::vm::bytecode_module
)

iree_cc_binary(
  NAME
    simple_embedding_local_sync
  SRCS
    "device_local_sync.c"
    "simple_embedding.c"
  DEPS
    ::simple_embedding_test_bytecode_module_dylib_arm_64_c
    ::simple_embedding_test_bytecode_module_dylib_riscv_64_c
    ::simple_embedding_test_bytecode_module_dylib_x86_64_c
    iree::base
    iree::hal
    iree::hal::local
    iree::hal::local::loaders::embedded_library_loader
    iree::hal::local::sync_driver
    iree::modules::hal
    iree::vm
    iree::vm::bytecode_module
)

iree_bytecode_module(
  NAME
    simple_embedding_test_bytecode_module_dylib_x86_64
  SRC
    "simple_embedding_test.mlir"
  C_IDENTIFIER
    "iree_samples_simple_embedding_test_module_dylib_x86_64"
  FLAGS
    "-iree-input-type=mhlo"
    "-iree-mlir-to-vm-bytecode-module"
    "-iree-hal-target-backends=dylib-llvm-aot"
    "-iree-llvm-target-triple=x86_64-pc-linux-elf"
    "-iree-llvm-link-embedded=true"
    "-iree-llvm-debug-symbols=false"
  PUBLIC
)

iree_bytecode_module(
  NAME
    simple_embedding_test_bytecode_module_dylib_riscv_64
  SRC
    "simple_embedding_test.mlir"
  C_IDENTIFIER
    "iree_samples_simple_embedding_test_module_dylib_riscv_64"
  FLAGS
    "-iree-input-type=mhlo"
    "-iree-mlir-to-vm-bytecode-module"
    "-iree-hal-target-backends=dylib-llvm-aot"
    "-iree-llvm-target-triple=riscv64-pc-linux-elf"
    "-iree-llvm-target-cpu=generic-rv64"
    "-iree-llvm-target-cpu-features=+m,+a,+f,+d,+c"
    "-iree-llvm-target-abi=lp64d"
    "-iree-llvm-link-embedded=true"
    "-iree-llvm-debug-symbols=false"
  PUBLIC
)

iree_bytecode_module(
  NAME
    simple_embedding_test_bytecode_module_dylib_arm_64
  SRC
    "simple_embedding_test.mlir"
  C_IDENTIFIER
    "iree_samples_simple_embedding_test_module_dylib_arm_64"
  FLAGS
    "-iree-input-type=mhlo"
    "-iree-mlir-to-vm-bytecode-module"
    "-iree-hal-target-backends=dylib-llvm-aot"
    "-iree-llvm-target-triple=aarch64-pc-linux-elf"
    "-iree-llvm-link-embedded=true"
    "-iree-llvm-debug-symbols=false"
  PUBLIC
)

iree_run_binary_test(
  NAME
    "simple_embedding_dylib_test"
  TEST_BINARY
    ::simple_embedding_dylib
)

iree_run_binary_test(
  NAME
    "simple_embedding_local_sync_test"
  TEST_BINARY
    ::simple_embedding_local_sync
)

endif()

if(${IREE_TARGET_BACKEND_VULKAN-SPIRV})

iree_cc_binary(
  NAME
    simple_embedding_vulkan
  SRCS
    "device_vulkan.c"
    "simple_embedding.c"
  DEPS
    ::simple_embedding_test_bytecode_module_vulkan_c
    iree::base
    iree::hal
    iree::hal::vulkan::registration
    iree::modules::hal
    iree::vm
    iree::vm::bytecode_module
)

iree_bytecode_module(
  NAME
    simple_embedding_test_bytecode_module_vulkan
  SRC
    "simple_embedding_test.mlir"
  C_IDENTIFIER
    "iree_samples_simple_embedding_test_module_vulkan"
  FLAGS
    "-iree-input-type=mhlo"
    "-iree-mlir-to-vm-bytecode-module"
    "-iree-hal-target-backends=vulkan-spirv"
    "-iree-llvm-debug-symbols=false"
  PUBLIC
)

iree_run_binary_test(
  NAME
    "simple_embedding_vulkan_test"
  TEST_BINARY
    ::simple_embedding_vulkan
)

endif()

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
