// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Self-launching coordinated test harness for multi-process tests.
//
// A single test binary re-executes itself in different roles. The launcher
// process spawns children with --iree_test_role=<name> flags, waits for
// readiness signals, and collects exit codes. Role functions receive the
// shared temp directory path and remaining argc/argv.
//
// Two usage patterns:
//
//   Gtest integration (primary):
//     Link coordinated_test_main instead of gtest_main. Register your config
//     with IREE_COORDINATED_TEST_REGISTER(config). Write TEST() bodies that
//     call iree_coordinated_test_run(). The main() handles child dispatch
//     before gtest init, so children never run RUN_ALL_TESTS().
//
//   Standalone (no gtest):
//     Call iree_coordinated_test_main() from your own main().

#ifndef IREE_TESTING_COORDINATED_TEST_H_
#define IREE_TESTING_COORDINATED_TEST_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Entry function for a single role. Receives remaining argc/argv (harness
// flags stripped) and the shared temp directory path. Returns 0 on success,
// nonzero on failure. The exit code is collected by the launcher.
typedef int (*iree_test_role_entry_fn_t)(int argc, char** argv,
                                         const char* temp_directory);

// Describes a single role in a coordinated test.
typedef struct iree_test_role_t {
  // Role name. Matches --iree_test_role=<name>. Must be unique within a
  // config.
  const char* name;

  // Entry function called when this process is assigned this role.
  iree_test_role_entry_fn_t entry;

  // If true, the launcher waits for this role to call
  // iree_coordinated_test_signal_ready() before spawning the next role.
  // Use for server roles that must bind a socket before clients connect.
  bool signals_ready;
} iree_test_role_t;

// Configuration for a coordinated test.
typedef struct iree_coordinated_test_config_t {
  // Roles to spawn, in launch order. The launcher spawns each role as a
  // child process. If a role has signals_ready=true, the launcher waits for
  // its ready signal before spawning the next role.
  const iree_test_role_t* roles;
  iree_host_size_t role_count;

  // Overall timeout in milliseconds for all children to complete. If any
  // child exceeds the timeout, all remaining children are killed.
  // 0 = default (30000ms).
  int64_t timeout_ms;
} iree_coordinated_test_config_t;

// Checks if --iree_test_role is present in argv, indicating this process is
// a child. If so, dispatches to the matching role entry function and returns
// its exit code (>= 0). If not a child process, returns -1.
//
// If |config| is NULL, uses the globally registered config (see
// iree_coordinated_test_register_config). If no config is available and this
// is a child process, aborts.
//
// Strips --iree_test_role and --iree_test_temp_dir from argc/argv before
// calling the role entry function.
int iree_coordinated_test_dispatch_if_child(
    int argc, char** argv, const iree_coordinated_test_config_t* config);

// Runs as launcher: discovers the self executable path, creates a temp
// directory, spawns children in role order (waiting for ready signals as
// configured), waits for all children to exit, cleans up, and reports
// results.
//
// Returns 0 if all children exit 0, else returns 1.
// Call from a gtest TEST body or from main().
int iree_coordinated_test_run(int argc, char** argv,
                              const iree_coordinated_test_config_t* config);

// Convenience: dispatch_if_child + run. For standalone binaries that don't
// need gtest in the launcher process.
int iree_coordinated_test_main(int argc, char** argv,
                               const iree_coordinated_test_config_t* config);

// Registers a config for use by dispatch_if_child when called with
// config=NULL. Typically called via IREE_COORDINATED_TEST_REGISTER at file
// scope. Only one config may be registered per binary.
void iree_coordinated_test_register_config(
    const iree_coordinated_test_config_t* config);

// Returns the argc/argv saved by coordinated_test_main.cc for use in TEST
// bodies. Only valid when using the coordinated_test_main library.
int iree_coordinated_test_argc(void);
char** iree_coordinated_test_argv(void);

// Signals to the launcher that this role is ready. Creates a sentinel file
// in the temp directory. Must be called exactly once per role that has
// signals_ready=true.
void iree_coordinated_test_signal_ready(const char* temp_directory);

#ifdef __cplusplus
}  // extern "C"

// Registers a coordinated test config at static initialization time.
// Use at file scope in the test .cc file:
//   IREE_COORDINATED_TEST_REGISTER(kConfig);
#define IREE_COORDINATED_TEST_REGISTER(config)          \
  static struct iree_coordinated_test_registrar_t {     \
    iree_coordinated_test_registrar_t() {               \
      iree_coordinated_test_register_config(&(config)); \
    }                                                   \
  } iree_coordinated_test_registrar_instance_##__LINE__

#endif  // __cplusplus

#endif  // IREE_TESTING_COORDINATED_TEST_H_
