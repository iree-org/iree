// Copyright 2020 Google LLC
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

#include "iree/hal/llvmjit/registration/driver_module.h"

#include <inttypes.h>

#include "iree/hal/llvmjit/llvmjit_driver.h"
#include "llvm/Support/TargetSelect.h"

#define IREE_HAL_LLVMJIT_DRIVER_ID 0x4C4C564Du  // LLVM

static iree_status_t iree_hal_llvmjit_driver_factory_enumerate(
    void* self, const iree_hal_driver_info_t** out_driver_infos,
    iree_host_size_t* out_driver_info_count) {
  static const iree_hal_driver_info_t driver_infos[1] = {{
      /*driver_id=*/IREE_HAL_LLVMJIT_DRIVER_ID,
      /*driver_name=*/iree_make_cstring_view("llvm"),
      /*full_name=*/iree_make_cstring_view("LLVM Bitcode JIT (deprecated)"),
  }};
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_llvmjit_driver_factory_try_create(
    void* self, iree_hal_driver_id_t driver_id, iree_allocator_t allocator,
    iree_hal_driver_t** out_driver) {
  if (driver_id != IREE_HAL_LLVMJIT_DRIVER_ID) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver with ID %016" PRIu64
                            " is provided by this factory",
                            driver_id);
  }
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto* driver = new iree::hal::llvmjit::LLVMJITDriver();
  *out_driver = reinterpret_cast<iree_hal_driver_t*>(driver);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_llvmjit_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      /*self=*/NULL,
      iree_hal_llvmjit_driver_factory_enumerate,
      iree_hal_llvmjit_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
