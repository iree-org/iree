// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./binding.h"
#include "./hal.h"
#include "./invoke.h"
#include "./io.h"
#include "./loop.h"
#include "./numpy_interop.h"
#include "./py_module.h"
#include "./status_utils.h"
#include "./vm.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/drivers/init.h"

namespace iree {
namespace python {

NB_MODULE(_runtime, m) {
  numpy::InitializeNumPyInterop();
  IREE_TRACE_APP_ENTER();

  IREE_CHECK_OK(iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default()));

  m.doc() = "IREE Binding Backend Helpers";
  SetupHalBindings(m);
  SetupInvokeBindings(m);
  SetupIoBindings(m);
  SetupLoopBindings(m);
  SetupPyModuleBindings(m);
  SetupVmBindings(m);

  m.def("parse_flags", [](py::args py_flags) {
    std::vector<std::string> alloced_flags;
    alloced_flags.push_back("python");
    for (py::handle py_flag : py_flags) {
      alloced_flags.push_back(py::cast<std::string>(py_flag));
    }

    // Must build pointer vector after filling so pointers are stable.
    std::vector<char *> flag_ptrs;
    for (auto &alloced_flag : alloced_flags) {
      flag_ptrs.push_back(const_cast<char *>(alloced_flag.c_str()));
    }

    char **argv = &flag_ptrs[0];
    int argc = flag_ptrs.size();
    CheckApiStatus(iree_flags_parse(IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                                    &argc, &argv),
                   "Error parsing flags");
  });

  m.def("disable_leak_checker", []() { py::set_leak_warnings(false); });
}

}  // namespace python
}  // namespace iree
