// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Kernel argument struct layouts for AMDGPU dispatch. Covers both the explicit
// kernel arguments (iree_hal_amdgpu_device_kernel_args_t) used by IREE's
// compiled kernel dispatch and the implicit kernel arguments
// (iree_amdgpu_kernel_implicit_args_t) defined by the LLVM AMDGPU backend for
// OpenCL/HIP compatibility.

#ifndef IREE_HAL_DRIVERS_AMDGPU_ABI_KERNEL_ARGS_H_
#define IREE_HAL_DRIVERS_AMDGPU_ABI_KERNEL_ARGS_H_

#include "iree/hal/drivers/amdgpu/abi/common.h"

typedef struct iree_hsa_queue_t iree_hsa_queue_t;

//===----------------------------------------------------------------------===//
// Kernel Arguments
//===----------------------------------------------------------------------===//

// Explicit kernel arguments for IREE dispatches where the kernel function
// signature is known at executable load time. These are populated from the
// code object metadata and cached for the lifetime of the executable so that
// dispatches can be issued without re-querying the symbol properties.
// This must match what the kernel was compiled to support.
typedef struct iree_hal_amdgpu_device_kernel_args_t {
  // Opaque handle to the kernel object to execute.
  uint64_t kernel_object;
  // Dispatch setup parameters. Used to configure kernel dispatch parameters
  // such as the number of dimensions in the grid. The parameters are
  // described by hsa_kernel_dispatch_packet_setup_t.
  uint16_t setup;
  // XYZ dimensions of work-group, in work-items. Must be greater than 0.
  // If the grid has fewer than 3 dimensions the unused must be 1.
  uint16_t workgroup_size[3];
  // Size in bytes of private memory allocation request (per work-item).
  uint32_t private_segment_size;
  // Size in bytes of group memory allocation request (per work-group). Must
  // not be less than the sum of the group memory used by the kernel (and the
  // functions it calls directly or indirectly) and the dynamically allocated
  // group segment variables.
  uint32_t group_segment_size;
  // Size of kernarg segment memory that is required to hold the values of the
  // kernel arguments, in bytes. Must be a multiple of 16.
  uint16_t kernarg_size;
  // Alignment (in bytes) of the buffer used to pass arguments to the kernel,
  // which is the maximum of 16 and the maximum alignment of any of the kernel
  // arguments.
  uint16_t kernarg_alignment;
  // Total number of 4-byte constants used by the dispatch (if a HAL dispatch).
  uint16_t constant_count;
  // Total number of bindings used by the dispatch (if a HAL dispatch).
  uint16_t binding_count;
  // Reserved for future hot kernel metadata. Must be zero.
  uint32_t reserved;
} iree_hal_amdgpu_device_kernel_args_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_device_kernel_args_t) <= 64,
    "keep hot kernel arg structure in as few cache lines as possible; every "
    "dispatch issued must access this information and it is likely uncached");

// Implicit kernel arguments passed to OpenCL/HIP kernels that use them.
// Not all kernels require this and the metadata needs to be checked to detect
// its use (or if the total kernargs size is > what we think it should be).
// Layout-wise explicit args always start at offset 0 and implicit args follow
// those with 8-byte alignment.
//
// The metadata will contain exact fields and offsets and most driver code will
// carefully walk to detect, align, pad, and write each field:
// OpenCL/HIP: (`amd::KernelParameterDescriptor`...)
// https://github.com/ROCm/clr/blob/5da72f9d524420c43fe3eee44b11ac875d884e0f/rocclr/device/rocm/rocvirtual.cpp#L3197
//
// This complex construction was required once upon a time. The LLVM code
// producing the kernargs layout and metadata handles these cases much more
// simply by only ever truncating the implicit args at the last used field:
// https://github.com/llvm/llvm-project/blob/7f1b465c6ae476e59dc90652d58fc648932d23b1/llvm/lib/Target/AMDGPU/AMDGPUHSAMetadataStreamer.cpp#L389
//
// Then at some point in time someone was like "meh, who cares about optimizing"
// and decided to include all of them always 🤦:
// https://github.com/llvm/llvm-project/blob/7f1b465c6ae476e59dc90652d58fc648932d23b1/llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp#L299
//
// What this means in practice is that if any implicit arg is used then all will
// be included and declared in the metadata even if only one is actually read by
// the kernel -- there's no way for us to know. In the ideal case none of them
// are read and the kernel function gets the `amdgpu-no-implicitarg-ptr` attr
// so that all of them can be skipped. Otherwise we reserve the 256 bytes and
// just splat them all in. This at least keeps our code simple relative to all
// the implementations that enumerate the metadata and write args one at a time.
// We really should try to force `amdgpu-no-implicitarg-ptr` when we generate
// code, though.
//
// For our bare-metal C runtime device code we have total freedom and don't use
// any OpenCL/HIP-related things that would emit the implicit args.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_amdgpu_kernel_implicit_args_t {
  // Grid dispatch workgroup count.
  // Some languages, such as OpenCL, support a last workgroup in each
  // dimension being partial. This count only includes the non-partial
  // workgroup count. This is not the same as the value in the AQL dispatch
  // packet, which has the grid size in workitems.
  //
  // Represented in metadata as:
  //   hidden_block_count_x
  //   hidden_block_count_y
  //   hidden_block_count_z
  uint32_t block_count[3];  // + 0/4/8

  // Grid dispatch workgroup size.
  // This size only applies to the non-partial workgroups. This is the same
  // value as the AQL dispatch packet workgroup size.
  //
  // Represented in metadata as:
  //   hidden_group_size_x
  //   hidden_group_size_y
  //   hidden_group_size_z
  uint16_t group_size[3];  // + 12/14/16

  // Grid dispatch work group size of the partial work group, if it exists.
  // Any dimension that does not exist must be 0. Only used in OpenCL and can
  // be 0.
  //
  // Represented in metadata as:
  //   hidden_remainder_x
  //   hidden_remainder_y
  //   hidden_remainder_z
  uint16_t remainder[3];  // + 18/20/22

  uint64_t reserved0;  // + 24 hidden_tool_correlation_id
  uint64_t reserved1;  // + 32

  // OpenCL grid dispatch global offset.
  // Always 0 in HIP but still required as the device library functions for
  // grid locations is shared with OpenCL and unconditionally factors it in.
  //
  // Hardcoded to 0 in HIP:
  // https://github.com/ROCm/clr/blob/a2550e0a9ecaa8f371cb14d08904c51874c37cbe/hipamd/src/hip_module.cpp#L348
  //
  // Represented in metadata as:
  //   hidden_global_offset_x
  //   hidden_global_offset_y
  //   hidden_global_offset_z
  uint64_t global_offset[3];  // + 40/48/56

  // Grid dispatch dimensionality. This is the same value as the AQL
  // dispatch packet dimensionality. Must be a value between 1 and 3.
  //
  // Hardcoded to 3 in HIP:
  // https://github.com/ROCm/clr/blob/a2550e0a9ecaa8f371cb14d08904c51874c37cbe/hipamd/src/hip_module.cpp#L349
  //
  // Represented in metadata as:
  //   hidden_grid_dims
  uint16_t grid_dims;  // + 64

  // Fixed-size buffer for `-mprintf-kind=buffered` support.
  // By default LLVM uses `hostcall` but that's a mess and we avoid it.
  // `__printf_alloc` in the device library is used to grab this pointer, the
  // header DWORDs are manipulated, and the contents are written to the buffer.
  //
  // struct {
  //   atomic_uint32_t offset;
  //   uint32_t size;
  //   uint8_t data[size];
  // } printf_buffer_t;
  //
  // One of many disappointing parts of this scheme is that constant string
  // values are interned, MD5 hashed, and stored *externally* in the amdhsa data
  // blob. In order to print with any constant format string this data blob
  // needs to be parsed, retained, and referenced every time a printf packet is
  // processed. It would have been significantly better to embed the table in
  // the ELF as a global constant instead as then we could reference it on both
  // host and device and not need to parse the amdhsa blob.
  //
  // The contents of the data buffer are best defined by the janky parser code:
  // https://github.com/ROCm/clr/blob/a2550e0a9ecaa8f371cb14d08904c51874c37cbe/rocclr/device/rocm/rocprintf.cpp#L454
  // Each printf consists of a control DWORD followed by 8-byte aligned
  // contents. Effectively:
  // struct {
  //   uint32_t is_stderr : 1;       // else stdout
  //   uint32_t constant : 1;        // constant format string code path
  //   uint32_t size_in_bytes : 30;  // (including this header)
  //   uint64_t data[size_in_bytes / 8];
  // } printf_packet_t;
  //
  // To construct the full format data buffer if constant == 1:
  //  data[0] contains the lower 64-bits of the MD5 hash of the string followed
  //  by size_in_bytes-12 arguments. The data buffer needs to be expanded into
  //  an 8-byte aligned NUL-terminated string with the corresponding hash
  //  followed by the arguments verbatim. Once reconstituted the subsequent
  //  logic is the same.
  //
  // The data buffer is an 8-byte aligned NUL-terminated string followed by
  // the argument data. E.g. `hi! %s` would be encoded as `hi! %s` 0x00 0x??
  // (with the last byte being padding to an 8-byte boundary). The reference
  // code for formatting the string lives in the CLR:
  // https://github.com/ROCm/clr/blob/a2550e0a9ecaa8f371cb14d08904c51874c37cbe/rocclr/device/devhcprintf.cpp#L168
  // Note that the documentation is incorrect about there being a version prefix
  // and it expects the first uint64_t to contain the format string bytes.
  //
  // Note that in another disappointing display of rube-goldbergian development
  // this implementation for some reason uses uint64_t for its data elements
  // but never aligns it - meaning that consumer code must use unaligned loads
  // in order to read the data. The CLR just copies it out each time. One could
  // think that was for streaming (release the buffer contents early back to
  // dispatches) but since they fully halt the world and synchronize after every
  // dispatch containing a print none of that matters and it's just poor
  // engineering.
  //
  // The compiler emits strings in the delimited form of
  // `"0:0:<format_string_hash>,<actual_format_string>"`. Note that the first
  // two values should always be 0 and are delimited by `:` while the MD5 hash
  // is delimited from the format string itself by `,`. There's some special
  // handling in the CLR for `:` being in the format string because whoever
  // wrote it did a find from the end instead of a prefix consume - there's
  // special handling of \72 (`:`) and other weird things that I'm not sure is
  // needed. Example from LLVM: `"0:0:8addc4c0362218ac,Hello World!:\n"`.
  //
  // The hash is the lower 64 bits of the MD5 hash in hex but we don't care as
  // it's just a semi-unique value we use to lookup the string formats. On load
  // we sort and do a binary search instead of creating an std::map for every
  // single print invocation like the CLR does. Just... wow.
  //
  // Handling the contents is also overtly complicated and poorly documented:
  // https://github.com/ROCm/clr/blob/a2550e0a9ecaa8f371cb14d08904c51874c37cbe/rocclr/device/devhcprintf.cpp#L168
  //
  // See:
  // https://github.com/ROCm/llvm-project/commit/631c965483e03355cdc1dba578e787b259c4d79d
  // https://github.com/ROCm/llvm-project/blob/997363823fcc5ccc7b0cc572aad05ba08714bf5f/amd/device-libs/ockl/src/cprintf.cl#L17
  // https://github.com/ROCm/clr/blob/a2550e0a9ecaa8f371cb14d08904c51874c37cbe/rocclr/device/rocm/rocprintf.cpp#L393
  //
  // Note that having a printf in a kernel causes the kernel to dispatch
  // synchronously :facepalm:. We can't do the same and would need to emit
  // flush packets (or something) into the control queue. What a mess.
  // https://github.com/ROCm/clr/blob/a2550e0a9ecaa8f371cb14d08904c51874c37cbe/rocclr/device/rocm/rocvirtual.cpp#L3644
  // https://github.com/ROCm/clr/blob/a2550e0a9ecaa8f371cb14d08904c51874c37cbe/rocclr/device/rocm/rocprintf.cpp#L428-L429
  //
  // Represented in metadata as:
  //   hidden_printf_buffer
  void* printf_buffer;  // + 72

  // Used for ASAN, printf, and more modern device memory allocations.
  // It's bizarre and only "documented" in code and I really hope we don't have
  // to touch it. Note that due to some LLVM bug sometimes this will be included
  // in the offset table for a kernel even if it is not used (the
  // `amdgpu-no-hostcall-ptr` attribute is set). At this point I'm quite sure no
  // one has ever actually inspected the files produced by the LLVM backend.
  //
  // Represented in metadata as:
  //   hidden_hostcall_buffer
  void* hostcall_buffer;  // + 80

  // Multi-grid support was deprecated in ROCM 5.x and should never appear in
  // any program we generate ourselves or care about running.
  //
  // Represented in metadata as:
  //   hidden_multigrid_sync_arg
  uint64_t deprecated_multigrid_sync_arg;

  // Device memory heap pointer for device malloc/free.
  // We don't support kernels using this as it requires too much goo for little
  // payoff. The kernels we run shouldn't be malloc/freeing internally. If they
  // do we will need to implement the heap API via hostcalls and other silly
  // things that add a tremendous amount of complexity.
  //
  // See:
  // https://github.com/ROCm/llvm-project/blob/97753eeaa4c79c2db2dcd9f37b7989596a8d4f15/amd/device-libs/ockl/src/dm.cl#L192
  //
  // Represented in metadata as:
  //   hidden_heap_v1
  uint64_t unused_heap_v1;

  // AQL queue handles are only used by OpenCL device-side enqueue and we do not
  // support that. We could, probably, by passing in our execution queue but
  // since HIP has never supported it the use case doesn't exist. If we wanted
  // to support device-enqueue we'd do it in a structured fashion instead of
  // letting kernels splat right into the AQL queue.
  //
  // See:
  // https://github.com/ROCm/llvm-project/blob/97753eeaa4c79c2db2dcd9f37b7989596a8d4f15/amd/device-libs/opencl/src/devenq/enqueue.cl#L310
  //
  // Represented in metadata as:
  //   hidden_default_queue
  uint64_t unused_default_queue;

  // Completion actions were (I believe) an attempt at dynamic parallelism and
  // HIP has never supported them. Device-side enqueue in OpenCL uses this but
  // we don't support those kernels.
  //
  // See:
  // https://github.com/ROCm/llvm-project/blob/97753eeaa4c79c2db2dcd9f37b7989596a8d4f15/amd/device-libs/opencl/src/devenq/enqueue.cl#L311
  //
  // Represented in metadata as:
  //   hidden_completion_action
  uint64_t unused_completion_action;

  // The value of the sharedMemBytes parameter to the dispatch indicating how
  // much dynamic shared memory was reserved for the kernel. This may be larger
  // than the requested amount. The total group_segment_size for a dispatch is
  // the static LDS requirement of the kernel plus this value.
  //
  // Represented in metadata as:
  //   hidden_dynamic_lds_size
  uint32_t dynamic_lds_size;

  uint8_t reserved[68];

  // Only used by GFX8, which we don't support.
  //
  // Represented in metadata as:
  //   hidden_private_base
  uint32_t deprecated_private_base;

  // Only used by GFX8, which we don't support.
  //
  // Represented in metadata as:
  //   hidden_shared_base
  uint32_t deprecated_shared_base;

  // AQL queue the dispatch is running on.
  // Only used by pre-GFX9 devices, which we don't support.
  //
  // Represented in metadata as:
  //   hidden_queue_ptr;
  iree_hsa_queue_t* deprecated_queue_ptr;
} iree_amdgpu_kernel_implicit_args_t;

#define IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE               \
  (IREE_AMDGPU_OFFSETOF(iree_amdgpu_kernel_implicit_args_t, \
                        dynamic_lds_size) +                 \
   sizeof(((iree_amdgpu_kernel_implicit_args_t*)NULL)->dynamic_lds_size))

#endif  // IREE_HAL_DRIVERS_AMDGPU_ABI_KERNEL_ARGS_H_
