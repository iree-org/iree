// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/run_module.h"
#include "iree/vm/api.h"

IREE_FLAG(bool, agents_md, false,
          "Prints AGENTS.md guidance for iree-run-module replay capture and "
          "exits.");

static const char kIreeRunModuleUsage[] =
    "Runs a compiled IREE module.\n"
    "\n"
    "Replay capture wraps the resolved HAL device group after normal "
    "--device=\n"
    "selection. Use --device_replay_output=path.ireereplay to record the HAL\n"
    "work issued by this run. All devices in the group share one recorder, so\n"
    "multi-device HAL calls are emitted into one host-call-ordered stream.\n"
    "Device-visible ordering is still the captured semaphore, event,\n"
    "command-buffer, and barrier graph; replay does not infer FIFO ordering\n"
    "from host record order.\n"
    "\n"
    "Replay capture flags:\n"
    "  --device_replay_output=path.ireereplay\n"
    "      Writes a HAL replay stream. The recorder is closed after module\n"
    "      execution and teardown so the file header contains the final "
    "length.\n"
    "  --device_replay_file_policy=reference|capture-ranges|capture-all|fail\n"
    "      Controls imported fd-backed HAL files such as parameter archives.\n"
    "      reference records path and validation metadata. capture-ranges\n"
    "      embeds only bytes read by queue_read operations. capture-all "
    "embeds\n"
    "      every byte. fail rejects fd-backed files.\n"
    "  --device_replay_file_validation=identity|digest\n"
    "      Validation for referenced fd-backed files. identity is the default\n"
    "      and avoids file content scans. digest reads every byte during "
    "capture\n"
    "      and replay.\n"
    "  --agents_md\n"
    "      Prints AGENTS.md guidance specific to iree-run-module capture. Use\n"
    "      `iree-run-replay --agents_md` for the full replay tool playbook.\n"
    "\n"
    "Example:\n"
    "  iree-run-module --device=local-sync --module=model.vmfb \\\n"
    "      --function=main --input=@inputs.txt \\\n"
    "      --device_replay_output=/tmp/model.ireereplay \\\n"
    "      --device_replay_file_policy=reference\n";

static void iree_run_module_print_agent_markdown(FILE* file) {
  fputs(
      "# iree-run-module Replay Capture\n"
      "\n"
      "`iree-run-module` can write a `.ireereplay` file while running a VMFB.\n"
      "Keep the normal module, function, input, parameter, and device flags, "
      "and\n"
      "add `--device_replay_output=/path/to/model.ireereplay`.\n"
      "\n"
      "Use `--device_replay_file_policy=reference` for large stable parameter\n"
      "archives so captures record path and identity metadata instead of "
      "copying\n"
      "the entire file. Use `capture-ranges` for hermetic correctness "
      "replays\n"
      "that only need bytes observed through HAL reads. Use `capture-all` "
      "only\n"
      "for small external files, and use `fail` when any fd-backed external "
      "file\n"
      "must make capture fail loudly.\n"
      "\n"
      "When replay capture is enabled, this tool emits standard replay scopes\n"
      "named `init`, `execute`, and `deinit`. Benchmark those phases later "
      "with\n"
      "`iree-benchmark-replay --replay_scope=execute` while still executing "
      "the\n"
      "complete captured stream.\n"
      "\n"
      "For replay execution, executable substitution, file remapping, dump "
      "JSONL,\n"
      "and the shared replay failure contract, pipe `iree-run-replay "
      "--agents_md`\n"
      "into your AGENTS.md.\n",
      file);
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Parse command line flags.
  iree_flags_set_usage("iree-run-module", kIreeRunModuleUsage);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (FLAG_agents_md) {
    iree_run_module_print_agent_markdown(stdout);
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
