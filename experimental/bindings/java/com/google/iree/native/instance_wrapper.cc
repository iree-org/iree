// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/bindings/java/com/google/iree/native/instance_wrapper.h"

#include <mutex>

#include "iree/base/internal/flags.h"
#include "iree/hal/vmla/registration/driver_module.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/modules/strings/strings_module.h"
#include "iree/modules/tensorlist/native_module.h"

namespace iree {
namespace java {

namespace {

void SetupVm() {
  // TODO(jennik): Pass flags through from java and us iree_flags_parse.
  // This checked version will abort()/exit() and that's... not great.
  char binname[] = "libiree.so";
  char* argv[] = {binname};
  char** aargv = argv;
  int argc = 1;
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &aargv);

  // TODO(jennik): register all available drivers
  IREE_CHECK_OK(iree_hal_vmla_driver_module_register(
      iree_hal_driver_registry_default()));
  IREE_CHECK_OK(iree_vm_register_builtin_types());
  IREE_CHECK_OK(iree_hal_module_register_types());
  IREE_CHECK_OK(iree_tensorlist_module_register_types());
  IREE_CHECK_OK(iree_strings_module_register_types());
}

}  // namespace

Status InstanceWrapper::Create() {
  static std::once_flag setup_vm_once;
  std::call_once(setup_vm_once, [] { SetupVm(); });

  return iree_vm_instance_create(iree_allocator_system(), &instance_);
}

iree_vm_instance_t* InstanceWrapper::instance() const { return instance_; }

InstanceWrapper::~InstanceWrapper() { iree_vm_instance_release(instance_); }

}  // namespace java
}  // namespace iree
