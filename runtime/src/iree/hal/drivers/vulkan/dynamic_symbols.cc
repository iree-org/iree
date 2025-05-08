// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/dynamic_symbols.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/dynamic_symbol_tables.h"

namespace iree {
namespace hal {
namespace vulkan {

// Read-only table of function pointer information designed to be in .rdata.
// To reduce binary size this structure is packed (knowing that we won't have
// gigabytes of function pointers :).
struct FunctionPtrInfo {
  // Name of the function (like 'vkSomeFunction').
  const char* function_name;
  // 1 if the function pointer can be resolved via vkGetDeviceProcAddr.
  uint32_t is_device : 1;
  // 1 if the function is required and the loader should bail if not found.
  uint32_t is_required : 1;
  // TODO(benvanik): remove from table by manually walking sizeof(uintptr_t).
  // An offset in bytes from the base of &syms to where the PFN_vkSomeFunction
  // member is located.
  uint32_t member_offset : 30;
} IREE_ATTRIBUTE_PACKED;

namespace {

#define REQUIRED_PFN_FUNCTION_PTR(function_name, is_device) \
  {#function_name, is_device, 1, offsetof(DynamicSymbols, function_name)},
#define OPTIONAL_PFN_FUNCTION_PTR(function_name, is_device) \
  {#function_name, is_device, 0, offsetof(DynamicSymbols, function_name)},
#define EXCLUDED_PFN_FUNCTION_PTR(function_name, is_device)
#define INS_PFN_FUNCTION_PTR(requirement, function_name) \
  requirement##_PFN_FUNCTION_PTR(function_name, 0)
#define DEV_PFN_FUNCTION_PTR(requirement, function_name) \
  requirement##_PFN_FUNCTION_PTR(function_name, 1)

// Defines the table of mandatory FunctionPtrInfos resolved prior to instance
// creation. These are safe to call with no instance parameter and should be
// exported by all loaders/ICDs.
static constexpr const FunctionPtrInfo kInstancelessFunctionPtrInfos[] = {
    IREE_VULKAN_DYNAMIC_SYMBOL_INSTANCELESS_TABLE(INS_PFN_FUNCTION_PTR)};

// Defines the table of FunctionPtrInfos for dynamic loading that must wait
// until an instance has been created to be resolved.
static constexpr const FunctionPtrInfo kDynamicFunctionPtrInfos[] = {
    IREE_VULKAN_DYNAMIC_SYMBOL_INSTANCE_DEVICE_TABLES(INS_PFN_FUNCTION_PTR,
                                                      DEV_PFN_FUNCTION_PTR)};

static const char* kVulkanLoaderSearchNames[] = {
#if defined(IREE_PLATFORM_ANDROID)
    "libvulkan.so",
#elif defined(IREE_PLATFORM_IOS) || defined(IREE_PLATFORM_MACOS)
    "libvulkan.dylib",
#elif defined(IREE_PLATFORM_WINDOWS)
    "vulkan-1.dll",
#else
    "libvulkan.so.1",
#endif  // IREE_PLATFORM_ANDROID
};

iree_status_t ResolveFunctions(
    DynamicSymbols* syms, const DynamicSymbols::GetProcAddrFn& get_proc_addr) {
  // Resolve the method the shared object uses to resolve other functions.
  // Some libraries will export all symbols while others will only export this
  // single function.
  syms->vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      get_proc_addr("vkGetInstanceProcAddr"));

#if defined(IREE_PLATFORM_ANDROID)
  // Since Android 8 Oreo, Android re-architected the OS framework with project
  // Treble. Framework libraries and vendor libraries have a more strict and
  // clear separation. Their dependencies are carefully scrutinized and only
  // selected cases are allowed. This is enforced with linker namespaces.
  //
  // /data/local/tmp is the preferred directory for automating native binary
  // tests built using NDK toolchain. They should be allowed to access libraries
  // like libvulkan.so for their functionality. However, there was an issue
  // with fully treblized Android 10 where /data/local/tmp did not have access
  // to the linker namespaces needed by libvulkan.so. This is fixed via
  // https://android.googlesource.com/platform/system/linkerconfig/+/296da5b1eb88a3527ee76352c2d987f82f3252eb
  //
  // But as typically in the Android system, it takes a long time to see the
  // fix getting propagated, if ever. A known workaround is to symlink the
  // vendor Vulkan implementation under /vendor/lib[64]/hw/vulkan.*.so as
  // libvulkan.so under /data/local/tmp and use LD_LIBRARY_PATH=/data/local/tmp
  // when invoking the test binaries. This effectively bypasses the Android
  // Vulkan loader. This is fine for ARM Mali GPUs, whose driver exposes
  // the symbol `vkGetInstanceProcAddr`. But for Qualcomm Adreno GPUs,
  // the Vulkan implementation library does not directly expose the symbol.
  // Instead it's hidden as `qglinternal::vkGetInstanceProcAddr`. So try to
  // see whether we can get this symbol. This is a reasonable workaround
  // as otherwise it means we need to wrap. every. single. binary. test.
  // as. a. full-blown. Android. app.
  if (!syms->vkGetInstanceProcAddr) {
    syms->vkGetInstanceProcAddr =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(get_proc_addr(
            // C++ mangled name for "qglinternal::vkGetInstanceProcAddr"
            "_ZN11qglinternal21vkGetInstanceProcAddrEP12VkInstance_TPKc"));
  }
#endif  // IREE_PLATFORM_ANDROID

  if (!syms->vkGetInstanceProcAddr) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "required method vkGetInstanceProcAddr not found in provided Vulkan "
        "library (did you pick the wrong file?)");
  }

  // Resolve the mandatory functions that we need to create instances.
  // If the provided |get_proc_addr| cannot resolve these then it's not a loader
  // or ICD we want to use, anyway.
  for (int i = 0; i < IREE_ARRAYSIZE(kInstancelessFunctionPtrInfos); ++i) {
    const auto& function_ptr = kInstancelessFunctionPtrInfos[i];
    auto* member_ptr = reinterpret_cast<PFN_vkVoidFunction*>(
        reinterpret_cast<uint8_t*>(syms) + function_ptr.member_offset);
    *member_ptr =
        syms->vkGetInstanceProcAddr(VK_NULL_HANDLE, function_ptr.function_name);
    if (*member_ptr == nullptr) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "mandatory Vulkan function %s not available; invalid loader/ICD?",
          function_ptr.function_name);
    }
  }

  return iree_ok_status();
}

}  // namespace

// static
iree_status_t DynamicSymbols::Create(const GetProcAddrFn& get_proc_addr,
                                     ref_ptr<DynamicSymbols>* out_syms) {
  IREE_TRACE_SCOPE_NAMED("DynamicSymbols::Create");

  auto syms = make_ref<DynamicSymbols>();
  IREE_RETURN_IF_ERROR(ResolveFunctions(syms.get(), get_proc_addr));
  syms->FixupExtensionFunctions();

  *out_syms = std::move(syms);
  return iree_ok_status();
}

// static
iree_status_t DynamicSymbols::CreateFromSystemLoader(
    iree_allocator_t host_allocator, ref_ptr<DynamicSymbols>* out_syms) {
  IREE_TRACE_SCOPE_NAMED("DynamicSymbols::CreateFromSystemLoader");

  iree_dynamic_library_t* loader_library = NULL;
  iree_status_t status = iree_dynamic_library_load_from_files(
      IREE_ARRAYSIZE(kVulkanLoaderSearchNames), kVulkanLoaderSearchNames,
      IREE_DYNAMIC_LIBRARY_FLAG_NONE, host_allocator, &loader_library);
  if (iree_status_is_not_found(status)) {
    iree_status_ignore(status);
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "Vulkan runtime library not available; ensure installed and on path");
  } else if (!iree_status_is_ok(status)) {
    return status;
  }

  auto syms = make_ref<DynamicSymbols>();
  syms->loader_library_ = loader_library;

  IREE_RETURN_IF_ERROR(
      ResolveFunctions(syms.get(), [loader_library](const char* function_name) {
        PFN_vkVoidFunction fn = NULL;
        iree_status_t status = iree_dynamic_library_lookup_symbol(
            loader_library, function_name, (void**)&fn);
        if (!iree_status_is_ok(status)) {
          IREE_IGNORE_ERROR(status);
          return (PFN_vkVoidFunction)NULL;
        }
        return fn;
      }));
  syms->FixupExtensionFunctions();

  *out_syms = std::move(syms);
  return iree_ok_status();
}

iree_status_t DynamicSymbols::LoadFromInstance(VkInstance instance) {
  IREE_TRACE_SCOPE_NAMED("DynamicSymbols::LoadFromInstance");
  return LoadFromDevice(instance, VK_NULL_HANDLE);
}

iree_status_t DynamicSymbols::LoadFromDevice(VkInstance instance,
                                             VkDevice device) {
  IREE_TRACE_SCOPE_NAMED("DynamicSymbols::LoadFromDevice");

  if (!instance) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "instance must have been created and a default "
                            "instance proc lookup function is required");
  }

  // Setup the lookup methods first. The rest of the syms uses these to
  // resolve function pointers.
  this->vkGetDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
      this->vkGetInstanceProcAddr(instance, "vkGetDeviceProcAddr"));
  if (!this->vkGetDeviceProcAddr) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "required Vulkan function vkGetDeviceProcAddr not "
                            "available; invalid driver handle?");
  }

  // Load the rest of the functions.
  for (int i = 0; i < IREE_ARRAYSIZE(kDynamicFunctionPtrInfos); ++i) {
    const auto& function_ptr = kDynamicFunctionPtrInfos[i];
    auto* member_ptr = reinterpret_cast<PFN_vkVoidFunction*>(
        reinterpret_cast<uint8_t*>(this) + function_ptr.member_offset);
    if (function_ptr.is_device && device) {
      *member_ptr =
          this->vkGetDeviceProcAddr(device, function_ptr.function_name);
    } else {
      *member_ptr =
          this->vkGetInstanceProcAddr(instance, function_ptr.function_name);
    }
    if (*member_ptr == nullptr && function_ptr.is_required) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "required Vulkan function %s not available",
                              function_ptr.function_name);
    }
  }

  FixupExtensionFunctions();

  return iree_ok_status();
}

DynamicSymbols::DynamicSymbols() = default;

DynamicSymbols::~DynamicSymbols() {
  if (loader_library_) {
    iree_dynamic_library_release(loader_library_);
  }
}

void DynamicSymbols::FixupExtensionFunctions() {
  this->vkGetSemaphoreCounterValue = this->vkGetSemaphoreCounterValue
                                         ? this->vkGetSemaphoreCounterValue
                                         : this->vkGetSemaphoreCounterValueKHR;
  this->vkWaitSemaphores = this->vkWaitSemaphores ? this->vkWaitSemaphores
                                                  : this->vkWaitSemaphoresKHR;
  this->vkSignalSemaphore = this->vkSignalSemaphore
                                ? this->vkSignalSemaphore
                                : this->vkSignalSemaphoreKHR;
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
