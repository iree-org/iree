// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_HAL_VULKAN_DYNAMIC_SYMBOLS_H_
#define IREE_HAL_VULKAN_DYNAMIC_SYMBOLS_H_

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <memory>

#include "iree/base/dynamic_library.h"
#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/hal/vulkan/dynamic_symbol_tables.h"

namespace iree {
namespace hal {
namespace vulkan {

struct FunctionPtrInfo;

// Dynamic Vulkan function loader for use with vulkan.hpp.
// This loader is a subset of the DispatchLoaderDynamic implementation that only
// loads functions we are interested in (a compute-specific subset) and avoids
// extensions we will never use.
//
// This exposes all Vulkan methods as function pointer members. Optional
// methods will be nullptr if not present. Excluded methods will be omitted.
//
// DynamicSymbols instances are designed to be passed to vulkan.hpp methods as
// the last argument, though they may also be called directly.
// **Always make sure to pass the loader to vulkan.hpp methods!**
//
// Loading is performed by walking a table of required and optional functions
// (defined in dynamic_symbol_tables.h) and populating the member function
// pointers exposed on this struct when available. For example, if the
// vkSomeFunction method is marked in the table as OPTIONAL the loader will
// attempt to lookup the function and if successful set the
// DynamicSymbols::vkSomeFunction pointer to the resolved address. If the
// function is not found then it will be set to nullptr so users can check for
// function availability.
//
// Documentation:
// https://github.com/KhronosGroup/Vulkan-Hpp#extensions--per-device-function-pointers
//
// Usage:
//  IREE_ASSIGN_OR_RETURN(auto syms, DynamicSymbols::CreateFromSystemLoader());
//  VkInstance instance = VK_NULL_HANDLE;
//  syms->vkCreateInstance(..., &instance);
//  IREE_RETURN_IF_ERROR(syms->LoadFromInstance(instance));
struct DynamicSymbols : public RefObject<DynamicSymbols> {
  using GetProcAddrFn =
      std::function<PFN_vkVoidFunction(const char* function_name)>;

  DynamicSymbols();
  ~DynamicSymbols();

  // Creates the dynamic symbol table using the given |get_proc_addr| to resolve
  // the vkCreateInstance function.
  //
  // After the instance is created the caller must use LoadFromInstance (or
  // LoadFromDevice) to load the remaining symbols.
  static StatusOr<ref_ptr<DynamicSymbols>> Create(
      const GetProcAddrFn& get_proc_addr);

  // Loads all required and optional Vulkan functions from the Vulkan loader.
  // This will look for a Vulkan loader on the system (like libvulkan.so) and
  // dlsym the functions from that.
  //
  // The loaded function pointers will point to thunks in the ICD. This may
  // enable additional debug checking and more readable stack traces (as
  // errors come from within the ICD, where we have symbols).
  static StatusOr<ref_ptr<DynamicSymbols>> CreateFromSystemLoader();

  // Loads all required and optional Vulkan functions from the given instance.
  //
  // The loaded function pointers will point to thunks in the ICD. This may
  // enable additional debug checking and more readable stack traces (as
  // errors come from within the ICD, where we have symbols).
  Status LoadFromInstance(VkInstance instance);

  // Loads all required and optional Vulkan functions from the given device,
  // falling back to the instance when required.
  //
  // This attempts to directly query the methods from the device, bypassing any
  // ICD or shim layers. These methods will generally have less overhead at
  // runtime as they need not jump through the various trampolines.
  Status LoadFromDevice(VkInstance instance, VkDevice device);

  // Define members for each function pointer.
  // See dynamic_symbol_tables.h for the full list of methods.
  //
  // Each required and optional function in the loader tables will expand to
  // the following member, such as for example 'vkSomeFunction':
  //   PFN_vkSomeFunction vkSomeFunction;
#define REQUIRED_PFN(function_name) PFN_##function_name function_name = nullptr
#define OPTIONAL_PFN(function_name) PFN_##function_name function_name = nullptr
#define EXCLUDED_PFN(function_name)
#define PFN_MEMBER(requirement, function_name) requirement##_PFN(function_name);
  REQUIRED_PFN(vkGetInstanceProcAddr);
  REQUIRED_PFN(vkGetDeviceProcAddr);
  IREE_VULKAN_DYNAMIC_SYMBOL_TABLES(PFN_MEMBER, PFN_MEMBER);
#undef REQUIRED_PFN
#undef OPTIONAL_PFN
#undef EXCLUDED_PFN
#undef PFN_MEMBER

 private:
  void FixupExtensionFunctions();

  // Optional Vulkan Loader dynamic library.
  std::unique_ptr<DynamicLibrary> loader_library_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_DYNAMIC_SYMBOLS_H_
