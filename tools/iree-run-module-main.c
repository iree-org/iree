// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/replay/help.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/run_module.h"
#include "iree/vm/api.h"

IREE_FLAG(bool, agents_md, false,
          "Prints an agent-oriented Markdown guide for HAL replay capture and "
          "tooling workflows and exits.");

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Parse command line flags.
  iree_flags_set_usage("iree-run-module", iree_hal_replay_capture_usage_text());
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (FLAG_agents_md) {
    iree_hal_replay_print_agent_markdown(stdout);
    fflush(stdout);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(EXIT_SUCCESS);
    return EXIT_SUCCESS;
  }

  // Hosting applications can provide their own allocators to pool resources or
  // track allocation statistics related to IREE code.
  iree_allocator_t host_allocator = iree_allocator_system();
  // Hosting applications should reuse instances across multiple contexts that
  // have similar composition (similar types/modules/etc). Most applications can
  // get by with a single shared instance.
  iree_vm_instance_t* instance = NULL;
  iree_status_t status =
      iree_tooling_create_instance(host_allocator, &instance);

  // Utility to run the module with the command line flags. This particular
  // method is only useful in these IREE tools that want consistent flags -
  // a real application will need to do what this is doing with its own setup
  // and I/O handling.
  int exit_code = EXIT_SUCCESS;
  if (iree_status_is_ok(status)) {
    status = iree_tooling_run_module_from_flags(instance, host_allocator,
                                                &exit_code);
  }

  iree_vm_instance_release(instance);

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
