<!-- markdownlint-disable -->
$ iree-run-module --list_drivers

# ============================================================================
# Available HAL drivers
# ============================================================================
# Use --list_devices={driver name} to enumerate available devices.

        cuda: NVIDIA CUDA HAL driver (via dylib)
         hip: HIP HAL driver (via dylib)
  local-sync: Local execution using a lightweight inline synchronous queue
  local-task: Local execution using the IREE multithreading task system
      vulkan: Vulkan 1.x (dynamic)
