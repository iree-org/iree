# "Simple Embedding" sample

This sample shows how to run a simple pointwise array multiplication bytecode
module on various HAL device targets with the minimum runtime overhead. Some of
these devices are compatible with bare-metal system without threading or file IO
support.

## Background

The main bytecode testing tool
[iree-run-module](../../tools/iree-run-module-main.cc)
requires a proper operating system support to set up the runtime environment to
execute an IREE bytecode module. For embedded systems, the support such as file
system or multi-thread asynchronous control may not be available. This sample
demonstrates how to setup the simplest framework to load and run the IREE
bytecode with various target backends.

## Build instructions

### CMake (native and cross compilation)

Set up the CMake configuration with `-DIREE_BUILD_SAMPLES=ON` (default on)

Then run
```sh
cmake --build <build dir> --target samples/simple_embedding/all
```

### Bazel (host only)

```sh
bazel build samples/simple_embedding:all
```

The resulting executables are listed as `simple_embedding_<HAL devices>`.

## Code structure

The sample consists of three parts:

### simple_embedding_test.mlir

The simple pointwise array multiplication op with the entry function called
`simple_mul`, two <4xf32> inputs, and one <4xf32> output. The ML bytecode
modules are automatically generated during the build time with the target HAL
device configurations from the host compiler `iree-compile`.

### simple_embedding.c

The main function of the sample has the following steps:

1. Create a VM instance
2. Create a HAL module based on the target device (see the next section)
3. Load the bytecode module of the ML workload
4. Associate the HAL module with the bytecode module in the VM context
5. Prepare the function entry point and inputs
6. Invoke function
7. Retrieve function output

### device_*.c

The HAL device for different target backends. Devices are created using a
specific executable loader and device constructor. For example,
[device_embedded_sync.c](./device_embedded_sync.c) creates a "sync" device with
the embedded ELF loader:

```c
iree_hal_sync_device_params_t params;
iree_hal_sync_device_params_initialize(&params);
iree_hal_executable_loader_t* loader = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_embedded_elf_loader_create(
      /*plugin_manager=*/NULL, iree_allocator_system(),
      &loader));

iree_string_view_t identifier = iree_make_cstring_view("local-sync");

iree_status_t status =
    iree_hal_sync_device_create(identifier, &params, /*loader_count=*/1,
                                &loader, iree_allocator_system(), device);
```

Whereas for [device_embedded.c](./device_embedded.c), the "sync device" is
replaced with the multithreaded "task device", which uses a "task executor":

```c
...
iree_task_executor_t* executor = NULL;
iree_host_size_t executor_count = 0;
iree_status_t status =
    iree_task_executors_create_from_flags(iree_allocator_system(),
                                          1, &executor, &executor_count);
IREE_ASSERT_EQ(count, 1, "NUMA unsupported");

iree_string_view_t identifier = iree_make_cstring_view("local-task");
if (iree_status_is_ok(status)) {
  // Create the device.
  status = iree_hal_task_device_create(identifier, &params,
                                       /*queue_count=*/1, &executor,
                                       /*loader_count=*/1, &loader,
                                       iree_allocator_system(), device);
```
An example that utilizes a higher-level driver registry is in
[device_vulkan.c](./device_vulkan.c)

#### Load device-specific bytecode module

To avoid the file IO, the bytecode module is converted into a data stream
(`module_data`) that's embedded in the executable. The same strategy can be
applied to build applications for the embedded systems without a proper file IO.

## Generic platform support

Some of the devices in this sample support a generic platform (or the
machine mode without an operating system). For example, `device_vmvx_sync`
should support any architecture that IREE supports, and `device_embedded_sync`
should support any architecture that supports `llvm-cpu` codegen target
backend (may need to add the bytecode module data if it is not already in
[device_embedded_sync.c](./device_embedded_sync.c)).
