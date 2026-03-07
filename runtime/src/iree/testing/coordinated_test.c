// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/coordinated_test.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(IREE_PLATFORM_WINDOWS)

#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#else  // POSIX

#include <dirent.h>
#include <errno.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#if defined(IREE_PLATFORM_APPLE)
#include <mach-o/dyld.h>
#endif  // IREE_PLATFORM_APPLE

// Required by posix_spawn on some platforms.
extern char** environ;

#endif  // IREE_PLATFORM_WINDOWS

//===----------------------------------------------------------------------===//
// Global config registration
//===----------------------------------------------------------------------===//

static const iree_coordinated_test_config_t*
    iree_coordinated_test_global_config;

void iree_coordinated_test_register_config(
    const iree_coordinated_test_config_t* config) {
  iree_coordinated_test_global_config = config;
}

static const iree_coordinated_test_config_t*
iree_coordinated_test_resolve_config(
    const iree_coordinated_test_config_t* config) {
  if (config) return config;
  if (iree_coordinated_test_global_config)
    return iree_coordinated_test_global_config;
  fprintf(stderr,
          "FATAL: iree_coordinated_test_dispatch_if_child called with "
          "config=NULL and no globally registered config.\n"
          "Use IREE_COORDINATED_TEST_REGISTER(config) at file scope.\n");
  abort();
  return NULL;
}

//===----------------------------------------------------------------------===//
// argc/argv accessors (set by coordinated_test_main.cc)
//===----------------------------------------------------------------------===//

static int iree_coordinated_test_saved_argc = 0;
static char** iree_coordinated_test_saved_argv = NULL;

int iree_coordinated_test_argc(void) {
  return iree_coordinated_test_saved_argc;
}

char** iree_coordinated_test_argv(void) {
  return iree_coordinated_test_saved_argv;
}

// Called by coordinated_test_main.cc to save argc/argv for TEST bodies.
void iree_coordinated_test_set_args(int argc, char** argv) {
  iree_coordinated_test_saved_argc = argc;
  iree_coordinated_test_saved_argv = argv;
}

//===----------------------------------------------------------------------===//
// Flag scanning
//===----------------------------------------------------------------------===//

// Scans argv for --iree_test_role=<value> and --iree_test_temp_dir=<value>.
// If found, sets *out_role / *out_temp_dir and strips the flags from argv
// (shifts remaining elements, decrements *argc_ptr).
static void iree_coordinated_test_scan_flags(int* argc_ptr, char** argv,
                                             const char** out_role,
                                             const char** out_temp_dir) {
  *out_role = NULL;
  *out_temp_dir = NULL;
  int argc = *argc_ptr;
  int write_index = 1;  // Preserve argv[0].
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--iree_test_role=", 17) == 0) {
      *out_role = argv[i] + 17;
    } else if (strncmp(argv[i], "--iree_test_temp_dir=", 21) == 0) {
      *out_temp_dir = argv[i] + 21;
    } else {
      argv[write_index++] = argv[i];
    }
  }
  *argc_ptr = write_index;
  argv[write_index] = NULL;
}

//===----------------------------------------------------------------------===//
// Executable path discovery
//===----------------------------------------------------------------------===//

// Returns the path to the current executable. |argv0| is used as a fallback
// when /proc/self/exe is unavailable or points to an interpreter. The caller
// must free the returned string with free(). Returns NULL on failure.
static char* iree_coordinated_test_get_self_path(const char* argv0) {
  IREE_TRACE_ZONE_BEGIN(z0);
  char* result = NULL;

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
  // readlink on /proc/self/exe with grow-and-retry.
  iree_host_size_t buffer_size = 256;
  while (buffer_size <= 65536) {
    char* buffer = (char*)malloc(buffer_size);
    if (!buffer) break;
    ssize_t length = readlink("/proc/self/exe", buffer, buffer_size);
    if (length < 0) {
      free(buffer);
      break;
    }
    if ((iree_host_size_t)length < buffer_size) {
      buffer[length] = '\0';
      result = buffer;
      break;
    }
    free(buffer);
    buffer_size *= 2;
  }

  // Under binfmt_misc interpreters (QEMU user-mode, Wine, FEX-Emu, etc.),
  // /proc/self/exe resolves to the interpreter binary rather than the test
  // executable. Detect this by comparing against argv[0]: if they resolve to
  // different files, we're running under an interpreter and argv[0] is the
  // correct path. With binfmt_misc registered, the kernel transparently
  // invokes the interpreter when the binary is exec'd, so posix_spawn of
  // the original binary works correctly.
  if (result && argv0) {
    char* argv0_real = realpath(argv0, NULL);
    if (argv0_real && strcmp(result, argv0_real) != 0) {
      free(result);
      result = argv0_real;
    } else {
      free(argv0_real);
    }
  }

#elif defined(IREE_PLATFORM_APPLE)
  uint32_t buffer_size = 0;
  _NSGetExecutablePath(NULL, &buffer_size);  // Get required size.
  char* raw_path = (char*)malloc(buffer_size);
  if (raw_path) {
    if (_NSGetExecutablePath(raw_path, &buffer_size) == 0) {
      // Resolve symlinks and relative components.
      result = realpath(raw_path, NULL);
    }
    free(raw_path);
  }

#elif defined(IREE_PLATFORM_WINDOWS)
  DWORD buffer_size = 256;
  while (buffer_size <= 65536) {
    char* buffer = (char*)malloc(buffer_size);
    if (!buffer) break;
    DWORD length = GetModuleFileNameA(NULL, buffer, buffer_size);
    if (length == 0) {
      free(buffer);
      break;
    }
    if (length < buffer_size) {
      result = buffer;
      break;
    }
    free(buffer);
    buffer_size *= 2;
  }

#else
  fprintf(stderr,
          "iree_coordinated_test: unsupported platform for executable path "
          "discovery\n");
#endif

  if (result) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, result);
  }
  IREE_TRACE_ZONE_END(z0);
  return result;
}

//===----------------------------------------------------------------------===//
// Process management
//===----------------------------------------------------------------------===//

typedef struct iree_test_process_t {
#if defined(IREE_PLATFORM_WINDOWS)
  HANDLE handle;
#else
  pid_t pid;
#endif
} iree_test_process_t;

// Spawns a child process running the given argv (NULL-terminated).
// The child inherits the parent's stdout/stderr.
static bool iree_coordinated_test_process_spawn(const char* const* argv,
                                                iree_test_process_t* out) {
  IREE_TRACE_ZONE_BEGIN(z0);
  bool success = false;

#if defined(IREE_PLATFORM_WINDOWS)
  // Build command line string from argv with proper quoting.
  char command_line[32768];
  iree_host_size_t position = 0;
  for (int i = 0; argv[i]; ++i) {
    if (i > 0) command_line[position++] = ' ';
    bool needs_quotes = strchr(argv[i], ' ') != NULL;
    if (needs_quotes) command_line[position++] = '"';
    iree_host_size_t arg_length = strlen(argv[i]);
    if (position + arg_length + 4 > sizeof(command_line)) {
      IREE_TRACE_ZONE_END(z0);
      return false;
    }
    memcpy(command_line + position, argv[i], arg_length);
    position += arg_length;
    if (needs_quotes) command_line[position++] = '"';
  }
  command_line[position] = '\0';

  STARTUPINFOA startup_info;
  memset(&startup_info, 0, sizeof(startup_info));
  startup_info.cb = sizeof(startup_info);
  PROCESS_INFORMATION process_info;
  memset(&process_info, 0, sizeof(process_info));

  if (CreateProcessA(NULL, command_line, NULL, NULL, TRUE, 0, NULL, NULL,
                     &startup_info, &process_info)) {
    CloseHandle(process_info.hThread);
    out->handle = process_info.hProcess;
    success = true;
  }

#else  // POSIX
  pid_t pid = 0;
  // posix_spawn expects a mutable argv (the signature is char*const*, not
  // const char*const*). We cast away const here — posix_spawn does not
  // modify the argv strings.
  int result = posix_spawn(&pid, argv[0], NULL, NULL, (char**)argv, environ);
  if (result == 0) {
    out->pid = pid;
    success = true;
  }
#endif

  IREE_TRACE_ZONE_END(z0);
  return success;
}

// Waits for a process to exit within |timeout_ms|. On success, sets
// |out_exit_code| and returns true. On timeout, returns false.
static bool iree_coordinated_test_process_wait(iree_test_process_t* process,
                                               int64_t timeout_ms,
                                               int* out_exit_code) {
  IREE_TRACE_ZONE_BEGIN(z0);
  bool completed = false;

#if defined(IREE_PLATFORM_WINDOWS)
  DWORD wait_result = WaitForSingleObject(
      process->handle, (DWORD)(timeout_ms > 0 ? timeout_ms : INFINITE));
  if (wait_result != WAIT_TIMEOUT) {
    DWORD exit_code = 1;
    GetExitCodeProcess(process->handle, &exit_code);
    *out_exit_code = (int)exit_code;
    completed = true;
  }

#else  // POSIX
  // Poll with 10ms intervals.
  int64_t remaining_ms = timeout_ms > 0 ? timeout_ms : 30000;
  while (remaining_ms > 0) {
    int status = 0;
    pid_t result = waitpid(process->pid, &status, WNOHANG);
    if (result > 0) {
      if (WIFEXITED(status)) {
        *out_exit_code = WEXITSTATUS(status);
      } else if (WIFSIGNALED(status)) {
        *out_exit_code = 128 + WTERMSIG(status);
      } else {
        *out_exit_code = 1;
      }
      completed = true;
      break;
    }
    if (result < 0 && errno != EINTR) {
      *out_exit_code = 1;
      completed = true;  // Process gone or error.
      break;
    }
    struct timespec sleep_time = {0, 10 * 1000 * 1000};  // 10ms.
    nanosleep(&sleep_time, NULL);
    remaining_ms -= 10;
  }
#endif

  IREE_TRACE_ZONE_END(z0);
  return completed;
}

// Forcibly terminates a process and reaps it.
static void iree_coordinated_test_process_kill(iree_test_process_t* process) {
#if defined(IREE_PLATFORM_WINDOWS)
  TerminateProcess(process->handle, 1);
  WaitForSingleObject(process->handle, 5000);
#else
  kill(process->pid, SIGKILL);
  int status = 0;
  waitpid(process->pid, &status, 0);
#endif
}

// Closes handles associated with a process.
static void iree_coordinated_test_process_close(iree_test_process_t* process) {
#if defined(IREE_PLATFORM_WINDOWS)
  if (process->handle) {
    CloseHandle(process->handle);
    process->handle = NULL;
  }
#else
  (void)process;
#endif
}

//===----------------------------------------------------------------------===//
// Temp directory management
//===----------------------------------------------------------------------===//

// Returns the best parent directory for temp files. Checks TEST_TMPDIR
// (Bazel), TMPDIR (POSIX convention), TEMP (Windows convention), then
// falls back to platform default.
static const char* iree_coordinated_test_get_temp_parent(void) {
  const char* dir = getenv("TEST_TMPDIR");
  if (dir && dir[0]) return dir;
  dir = getenv("TMPDIR");
  if (dir && dir[0]) return dir;
#if defined(IREE_PLATFORM_WINDOWS)
  dir = getenv("TEMP");
  if (dir && dir[0]) return dir;
  return "C:\\Temp";
#else
  return "/tmp";
#endif
}

// Creates a unique temp directory. Returns a malloc'd path or NULL on failure.
static char* iree_coordinated_test_make_temp_dir(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  char* result = NULL;

#if defined(IREE_PLATFORM_WINDOWS)
  char parent[MAX_PATH];
  GetTempPathA(MAX_PATH, parent);
  static volatile long counter = 0;
  DWORD pid = GetCurrentProcessId();
  long count = InterlockedIncrement(&counter);
  char path[MAX_PATH];
  snprintf(path, sizeof(path), "%siree_ct_%lu_%ld", parent, (unsigned long)pid,
           count);
  if (CreateDirectoryA(path, NULL)) {
    result = _strdup(path);
  }

#else  // POSIX
  const char* parent = iree_coordinated_test_get_temp_parent();

  // Check that the resulting path will be short enough for Unix domain
  // sockets (107 bytes for sun_path + NUL). The longest socket path we
  // produce is "<temp_dir>/carrier.sock" (13 chars + NUL).
  // If TEST_TMPDIR is too long, fall back to /tmp.
  char template_path[256];
  int written = snprintf(template_path, sizeof(template_path),
                         "%s/iree_ct_XXXXXX", parent);
  if (written < 0 || (iree_host_size_t)written >= sizeof(template_path) ||
      written + 14 > 107) {
    // Path too long for sun_path — fall back to /tmp.
    snprintf(template_path, sizeof(template_path), "/tmp/iree_ct_XXXXXX");
  }
  if (mkdtemp(template_path)) {
    result = strdup(template_path);
  }
#endif

  if (result) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, result);
  }
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Best-effort recursive removal of a directory.
static void iree_coordinated_test_remove_temp_dir(const char* path) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path);

#if defined(IREE_PLATFORM_WINDOWS)
  char search_path[MAX_PATH];
  snprintf(search_path, sizeof(search_path), "%s\\*", path);
  WIN32_FIND_DATAA find_data;
  HANDLE find_handle = FindFirstFileA(search_path, &find_data);
  if (find_handle != INVALID_HANDLE_VALUE) {
    do {
      if (strcmp(find_data.cFileName, ".") == 0 ||
          strcmp(find_data.cFileName, "..") == 0)
        continue;
      char child_path[MAX_PATH];
      snprintf(child_path, sizeof(child_path), "%s\\%s", path,
               find_data.cFileName);
      if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        iree_coordinated_test_remove_temp_dir(child_path);
      } else {
        DeleteFileA(child_path);
      }
    } while (FindNextFileA(find_handle, &find_data));
    FindClose(find_handle);
    RemoveDirectoryA(path);
  }

#else  // POSIX
  DIR* dir = opendir(path);
  if (dir) {
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
      if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
        continue;
      char child_path[512];
      snprintf(child_path, sizeof(child_path), "%s/%s", path, entry->d_name);
      struct stat st;
      if (stat(child_path, &st) == 0 && S_ISDIR(st.st_mode)) {
        iree_coordinated_test_remove_temp_dir(child_path);
      } else {
        unlink(child_path);
      }
    }
    closedir(dir);
    rmdir(path);
  }
#endif

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Ready file protocol
//===----------------------------------------------------------------------===//

void iree_coordinated_test_signal_ready(const char* temp_directory) {
  IREE_TRACE_ZONE_BEGIN(z0);
  char ready_path[512];
  snprintf(ready_path, sizeof(ready_path), "%s/.ready", temp_directory);
  FILE* file = fopen(ready_path, "w");
  if (file) fclose(file);
  IREE_TRACE_ZONE_END(z0);
}

// Polls for the ready file to appear. Returns true when found, false on
// timeout.
static bool iree_coordinated_test_wait_ready(const char* temp_directory,
                                             int64_t timeout_ms) {
  IREE_TRACE_ZONE_BEGIN(z0);
  bool found = false;
  char ready_path[512];
  snprintf(ready_path, sizeof(ready_path), "%s/.ready", temp_directory);
  int64_t remaining_ms = timeout_ms;
  while (remaining_ms > 0) {
#if defined(IREE_PLATFORM_WINDOWS)
    DWORD attributes = GetFileAttributesA(ready_path);
    if (attributes != INVALID_FILE_ATTRIBUTES) {
      DeleteFileA(ready_path);  // Clean up for reuse.
      found = true;
      break;
    }
    Sleep(1);
#else
    struct stat st;
    if (stat(ready_path, &st) == 0) {
      unlink(ready_path);  // Clean up for reuse.
      found = true;
      break;
    }
    struct timespec sleep_time = {0, 1 * 1000 * 1000};  // 1ms.
    nanosleep(&sleep_time, NULL);
#endif
    remaining_ms -= 1;
  }
  IREE_TRACE_ZONE_END(z0);
  return found;
}

//===----------------------------------------------------------------------===//
// Child dispatch
//===----------------------------------------------------------------------===//

int iree_coordinated_test_dispatch_if_child(
    int argc, char** argv, const iree_coordinated_test_config_t* config) {
  const char* role_name = NULL;
  const char* temp_directory = NULL;
  iree_coordinated_test_scan_flags(&argc, argv, &role_name, &temp_directory);
  if (!role_name) return -1;  // Not a child.

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, role_name);

  config = iree_coordinated_test_resolve_config(config);
  int result = 1;
  for (iree_host_size_t i = 0; i < config->role_count; ++i) {
    if (strcmp(config->roles[i].name, role_name) == 0) {
      result = config->roles[i].entry(argc, argv, temp_directory);
      IREE_TRACE_ZONE_END(z0);
      return result;
    }
  }
  fprintf(stderr, "FATAL: unknown role '%s'\n", role_name);
  IREE_TRACE_ZONE_END(z0);
  return 1;
}

//===----------------------------------------------------------------------===//
// Launcher
//===----------------------------------------------------------------------===//

// Maximum number of roles that can be spawned in a single coordinated test.
#define IREE_COORDINATED_TEST_MAX_ROLES 16

int iree_coordinated_test_run(int argc, char** argv,
                              const iree_coordinated_test_config_t* config) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!config || config->role_count == 0 ||
      config->role_count > IREE_COORDINATED_TEST_MAX_ROLES) {
    fprintf(stderr,
            "coordinated_test: invalid config (role_count=%" PRIhsz ")\n",
            config ? config->role_count : (iree_host_size_t)0);
    IREE_TRACE_ZONE_END(z0);
    return 1;
  }

  int64_t timeout_ms = config->timeout_ms > 0 ? config->timeout_ms : 30000;

  // Discover our own executable path.
  char* self_path = iree_coordinated_test_get_self_path(argv[0]);
  if (!self_path) {
    fprintf(stderr, "coordinated_test: failed to discover executable path\n");
    IREE_TRACE_ZONE_END(z0);
    return 1;
  }

  // Create unique temp directory.
  char* temp_directory = iree_coordinated_test_make_temp_dir();
  if (!temp_directory) {
    fprintf(stderr, "coordinated_test: failed to create temp directory\n");
    free(self_path);
    IREE_TRACE_ZONE_END(z0);
    return 1;
  }

  fprintf(stderr, "coordinated_test: temp_dir=%s\n", temp_directory);

  // Build per-role flag strings.
  char role_flag[256];
  char temp_dir_flag[512];
  snprintf(temp_dir_flag, sizeof(temp_dir_flag), "--iree_test_temp_dir=%s",
           temp_directory);

  // Spawn children. argv layout: [self_path, original_args..., --role, --temp,
  // NULL].
  iree_test_process_t processes[IREE_COORDINATED_TEST_MAX_ROLES];
  memset(processes, 0, sizeof(processes));
  bool spawned[IREE_COORDINATED_TEST_MAX_ROLES];
  memset(spawned, 0, sizeof(spawned));
  int exit_codes[IREE_COORDINATED_TEST_MAX_ROLES];
  memset(exit_codes, 0, sizeof(exit_codes));
  bool completed[IREE_COORDINATED_TEST_MAX_ROLES];
  memset(completed, 0, sizeof(completed));

  // Build the base argv: [self_path, original_args..., <placeholder>,
  // <placeholder>, NULL]. We'll fill in the role/temp flags per-role.
  int base_argc = argc + 2;  // original + 2 new flags
  const char** child_argv =
      (const char**)malloc((base_argc + 1) * sizeof(const char*));
  if (!child_argv) {
    free(self_path);
    iree_coordinated_test_remove_temp_dir(temp_directory);
    free(temp_directory);
    IREE_TRACE_ZONE_END(z0);
    return 1;
  }
  child_argv[0] = self_path;
  for (int i = 1; i < argc; ++i) {
    child_argv[i] = argv[i];
  }
  // Slots for role flag and temp_dir flag.
  int role_flag_index = argc;
  int temp_dir_flag_index = argc + 1;
  child_argv[temp_dir_flag_index] = temp_dir_flag;
  child_argv[base_argc] = NULL;

  int result = 0;
  int64_t remaining_ms = timeout_ms;

  for (iree_host_size_t i = 0; i < config->role_count; ++i) {
    snprintf(role_flag, sizeof(role_flag), "--iree_test_role=%s",
             config->roles[i].name);
    child_argv[role_flag_index] = role_flag;

    fprintf(stderr, "coordinated_test: spawning role '%s'\n",
            config->roles[i].name);
    if (!iree_coordinated_test_process_spawn(child_argv, &processes[i])) {
      fprintf(stderr, "coordinated_test: FAILED to spawn role '%s'\n",
              config->roles[i].name);
      result = 1;
      break;
    }
    spawned[i] = true;

    // Wait for ready signal if required.
    if (config->roles[i].signals_ready) {
      if (!iree_coordinated_test_wait_ready(temp_directory, remaining_ms)) {
        fprintf(stderr,
                "coordinated_test: TIMEOUT waiting for role '%s' to signal "
                "ready\n",
                config->roles[i].name);
        result = 1;
        break;
      }
    }
  }

  // Wait for all spawned children to exit.
  if (result == 0) {
    for (iree_host_size_t i = 0; i < config->role_count; ++i) {
      if (!spawned[i]) continue;
      if (iree_coordinated_test_process_wait(&processes[i], remaining_ms,
                                             &exit_codes[i])) {
        completed[i] = true;
      }
    }
  }

  // Kill any that didn't complete.
  for (iree_host_size_t i = 0; i < config->role_count; ++i) {
    if (spawned[i] && !completed[i]) {
      fprintf(stderr, "coordinated_test: killing role '%s' (timeout)\n",
              config->roles[i].name);
      iree_coordinated_test_process_kill(&processes[i]);
      exit_codes[i] = -1;
      result = 1;
    }
    if (spawned[i]) {
      iree_coordinated_test_process_close(&processes[i]);
    }
  }

  // Report results.
  fprintf(stderr, "coordinated_test: results:\n");
  for (iree_host_size_t i = 0; i < config->role_count; ++i) {
    if (!spawned[i]) {
      fprintf(stderr, "  %-12s NOT SPAWNED\n", config->roles[i].name);
      continue;
    }
    if (!completed[i]) {
      fprintf(stderr, "  %-12s TIMEOUT\n", config->roles[i].name);
      continue;
    }
    if (exit_codes[i] != 0) {
      fprintf(stderr, "  %-12s FAILED (exit code %d)\n", config->roles[i].name,
              exit_codes[i]);
      result = 1;
    } else {
      fprintf(stderr, "  %-12s PASSED\n", config->roles[i].name);
    }
  }

  // Cleanup.
  free(child_argv);
  iree_coordinated_test_remove_temp_dir(temp_directory);
  free(temp_directory);
  free(self_path);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

//===----------------------------------------------------------------------===//
// Convenience entry point
//===----------------------------------------------------------------------===//

int iree_coordinated_test_main(int argc, char** argv,
                               const iree_coordinated_test_config_t* config) {
  int child_result =
      iree_coordinated_test_dispatch_if_child(argc, argv, config);
  if (child_result >= 0) return child_result;
  return iree_coordinated_test_run(argc, argv, config);
}
