// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_FLAGS_H_
#define IREE_BASE_INTERNAL_FLAGS_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Flags configuration
//===----------------------------------------------------------------------===//

// 1 to enable command line parsing from argc/argv; 0 otherwise.
// When parsing is disabled flags are just variables that can still be queried
// and manually overridden by code if desired.
#if !defined(IREE_FLAGS_ENABLE_CLI)
#define IREE_FLAGS_ENABLE_CLI 1
#endif  // !IREE_FLAGS_ENABLE_CLI

// 1 to enable --flagfile= support.
#if !defined(IREE_FLAGS_ENABLE_FLAG_FILE)
// The feature only works when file IO is available.
#if IREE_FILE_IO_ENABLE
#define IREE_FLAGS_ENABLE_FLAG_FILE 1
#else
#define IREE_FLAGS_ENABLE_FLAG_FILE 0
#endif  // IREE_FILE_IO_ENABLE
#endif  // !IREE_FLAGS_ENABLE_FLAG_FILE

// Maximum number of flags that can be registered in a single binary.
#if !defined(IREE_FLAGS_CAPACITY)
#define IREE_FLAGS_CAPACITY 64
#endif  // !IREE_FLAGS_CAPACITY

//===----------------------------------------------------------------------===//
// Static initialization utility
//===----------------------------------------------------------------------===//
// This declares a static initialization function with the given name.
// Usage:
//   IREE_STATIC_INITIALIZER(initializer_name) {
//     // Do something here! Note that initialization order is undefined and
//     // what you do should be tolerant to that.
//
//     // If you want a finalizer (you probably don't; they may not get run)
//     // then you can use atexit:
//     atexit(some_finalizer_fn);
//   }

#ifdef __cplusplus

#define IREE_STATIC_INITIALIZER(f) \
  static void f(void);             \
  struct f##_t_ {                  \
    f##_t_(void) { f(); }          \
  };                               \
  static f##_t_ f##_;              \
  static void f(void)

#elif defined(IREE_COMPILER_MSVC)

// `__attribute__((constructor))`-like behavior in MSVC. See:
// https://docs.microsoft.com/en-us/cpp/c-runtime-library/crt-initialization?view=msvc-160

#pragma section(".CRT$XCU", read)
#define IREE_STATIC_INITIALIZER_IMPL(f, p)                 \
  static void f(void);                                     \
  __declspec(allocate(".CRT$XCU")) void (*f##_)(void) = f; \
  __pragma(comment(linker, "/include:" p #f "_")) static void f(void)
#ifdef _WIN64
#define IREE_STATIC_INITIALIZER(f) IREE_STATIC_INITIALIZER_IMPL(f, "")
#else
#define IREE_STATIC_INITIALIZER(f) IREE_STATIC_INITIALIZER_IMPL(f, "_")
#endif  // _WIN64

#else

#define IREE_STATIC_INITIALIZER(f)                  \
  static void f(void) __attribute__((constructor)); \
  static void f(void)

#endif  // __cplusplus / MSVC

//===----------------------------------------------------------------------===//
// Flag definition
//===----------------------------------------------------------------------===//

enum iree_flag_dump_mode_bits_t {
  IREE_FLAG_DUMP_MODE_DEFAULT = 0u,
  IREE_FLAG_DUMP_MODE_VERBOSE = 1u << 0,
};
typedef uint32_t iree_flag_dump_mode_t;

#define IREE_FLAG_CTYPE_bool bool
#define IREE_FLAG_CTYPE_int32_t int32_t
#define IREE_FLAG_CTYPE_int64_t int64_t
#define IREE_FLAG_CTYPE_float float
#define IREE_FLAG_CTYPE_double double
#define IREE_FLAG_CTYPE_string const char*

#if IREE_FLAGS_ENABLE_CLI == 1

// Types of flags supported by the parser.
typedef enum iree_flag_type_e {
  // Empty/unspecified sentinel.
  IREE_FLAG_TYPE_none = 0,
  // Custom parsing callback; see IREE_FLAG_CALLBACK.
  IREE_FLAG_TYPE_callback = 1,
  // Boolean flag:
  //  --foo (set true)
  //  --foo=true | --foo=false
  IREE_FLAG_TYPE_bool,
  // 32-bit integer flag:
  //  --foo=123
  IREE_FLAG_TYPE_int32_t,
  // 64-bit integer flag:
  //  --foo=123
  IREE_FLAG_TYPE_int64_t,
  // 32-bit floating-point flag:
  //  --foo=1.2
  IREE_FLAG_TYPE_float,
  // 64-bit floating-point flag:
  //  --foo=1.2
  IREE_FLAG_TYPE_double,
  // String flag:
  //  --foo=abc
  //  --foo="a b c"
  // Holds a reference to constant string data; assigned values must remain
  // live for as long as the flag value references them.
  IREE_FLAG_TYPE_string,
} iree_flag_type_t;

// Custom callback issued for each time the flag is seen during parsing.
// The |value| provided will already be trimmed and may be empty. For
// compatibility with non-IREE APIs there will be a NUL terminator immediately
// following the flag value in memory such that `value.data` can be used as a
// C-string.
typedef iree_status_t(IREE_API_PTR* iree_flag_parse_callback_fn_t)(
    iree_string_view_t flag_name, void* storage, iree_string_view_t value);

// Custom callback issued for each time the flag is to be printed.
// The callback should print the flag and its value to |file|.
// Example: `--my_flag=value\n`
typedef void(IREE_API_PTR* iree_flag_print_callback_fn_t)(
    iree_string_view_t flag_name, void* storage, FILE* file);

int iree_flag_register(const char* file, int line, iree_flag_type_t type,
                       void* storage,
                       iree_flag_parse_callback_fn_t parse_callback,
                       iree_flag_print_callback_fn_t print_callback,
                       iree_string_view_t name, iree_string_view_t description);

// Defines a flag with the given |type| and |name|.
//
// Conceptually the flag is just a variable and can be loaded/stored:
//   IREE_FLAG(bool, foo, true, "hello");
//  =>
//   static bool FLAG_foo = true;
//  ...
//   if (FLAG_foo) do_something();
//
// If flag parsing is enabled with IREE_FLAGS_ENABLE_CLI == 1 then the flag
// value can be specified on the command line with --name:
//   --foo
//   --foo=true
//
// See iree_flag_type_t for the types supported and how they are parsed.
#define IREE_FLAG(type, name, default_value, description)                      \
  static IREE_FLAG_CTYPE_##type FLAG_##name = (default_value);                 \
  IREE_STATIC_INITIALIZER(iree_flag_register_##name) {                         \
    iree_flag_register(__FILE__, __LINE__, IREE_FLAG_TYPE_##type,              \
                       (void**)&(FLAG_##name), /*parse_callback=*/NULL,        \
                       /*print_callback=*/NULL, iree_make_cstring_view(#name), \
                       iree_make_cstring_view(description));                   \
  }

// Defines a flag issues |callback| for custom parsing.
//
// Usage:
//  iree_status_t parse_callback(const char* flag_name, void* storage,
//                               iree_string_view_t value) {
//    // Parse |value| and store in |storage|, however you want.
//    // Returning IREE_STATUS_INVALID_ARGUMENT will trigger --help.
//    int* storage_ptr = (int*)storage;
//    printf("hello! %d", (*storage_ptr)++);
//    return iree_ok_status();
//  }
//  void print_callback(const char* flag_name, void* storage, FILE* file) {
//    // Print the value in |storage|, however you want. For repeated fields
//    // you can print multiple separated by newlines.
//    int* storage_ptr = (int*)storage;
//    fprintf(file, "--say_hello=%d\n", *storage_ptr);
//  }
//  int my_storage = 0;
//  IREE_FLAG_CALLBACK(parse_callback, print_callback, &my_storage,
//                     say_hello, "Say hello!");
#define IREE_FLAG_CALLBACK(parse_callback, print_callback, storage, name, \
                           description)                                   \
  IREE_STATIC_INITIALIZER(iree_flag_register_##name) {                    \
    iree_flag_register(__FILE__, __LINE__, IREE_FLAG_TYPE_callback,       \
                       (void*)storage, parse_callback, print_callback,    \
                       iree_make_cstring_view(#name),                     \
                       iree_make_cstring_view(description));              \
  }

#else

#define IREE_FLAG(type, name, default_value, description) \
  static const IREE_FLAG_CTYPE_##type FLAG_##name = (default_value);

#define IREE_FLAG_CALLBACK(parse_callback, print_callback, storage, name, \
                           description)

#endif  // IREE_FLAGS_ENABLE_CLI

//===----------------------------------------------------------------------===//
// List flag utilities
//===----------------------------------------------------------------------===//

// A list of string views referencing flag storage.
typedef struct iree_flag_string_list_t {
  // Total number of values in the list.
  iree_host_size_t count;
  // Value list or NULL if no values.
  const iree_string_view_t* values;
} iree_flag_string_list_t;

#if IREE_FLAGS_ENABLE_CLI == 1

// Internal storage; do not use.
typedef struct iree_flag_string_list_storage_t {
  iree_host_size_t capacity;
  iree_host_size_t count;
  union {
    iree_string_view_t inline_value;  // only if count == 1
    iree_string_view_t* values;       // only if count > 1
  };
} iree_flag_string_list_storage_t;
iree_status_t iree_flag_string_list_parse(iree_string_view_t flag_name,
                                          void* storage,
                                          iree_string_view_t value);
void iree_flag_string_list_print(iree_string_view_t flag_name, void* storage,
                                 FILE* file);

// Defines a repeated flag representing a dynamically sized list of values.
//
// Usage:
//   IREE_FLAG_LIST(string, foo, "hello");
//   ...
//   const iree_flag_string_list_t list = FLAG_foo_list();
//   for (iree_host_size_t i = 0; i < list.count; ++i) {
//     printf("value: %.*s", (int)list.values[i].size, list.values[i].data);
//   }
//   ...
//   ./binary --foo=a --foo=b
//   > value: a
//   > value: b
#define IREE_FLAG_LIST(type, name, description)                             \
  static iree_flag_##type##_list_storage_t FLAG_##name##_storage = {        \
      /*.capacity=*/1 /* inline by default */,                              \
      /*.count=*/0,                                                         \
  };                                                                        \
  IREE_FLAG_CALLBACK(iree_flag_##type##_list_parse,                         \
                     iree_flag_##type##_list_print, &FLAG_##name##_storage, \
                     name, description);                                    \
  static const iree_flag_##type##_list_t FLAG_##name##_list(void) {         \
    const iree_flag_##type##_list_t list = {                                \
        /*.count=*/FLAG_##name##_storage.count,                             \
        /*.values=*/FLAG_##name##_storage.count == 1                        \
            ? &FLAG_##name##_storage.inline_value                           \
            : FLAG_##name##_storage.values,                                 \
    };                                                                      \
    return list;                                                            \
  }

#else

#define IREE_FLAG_LIST(type, name, description)                     \
  static const iree_flag_##type##_list_t FLAG_##name##_list(void) { \
    return (iree_flag_##type##_list_t){0, NULL};                    \
  }

#endif  // IREE_FLAGS_ENABLE_CLI

//===----------------------------------------------------------------------===//
// Flag parsing
//===----------------------------------------------------------------------===//

// Controls how flag parsing is performed.
enum iree_flags_parse_mode_bits_t {
  IREE_FLAGS_PARSE_MODE_DEFAULT = 0,
  // Do not error out on undefined flags; leave them in the list.
  // Useful when needing to chain multiple flag parsers together.
  IREE_FLAGS_PARSE_MODE_UNDEFINED_OK = 1u << 0,
  // Continues parsing and returns success without exiting when `--help` is
  // encountered. This allows for IREE flag parsing to happen before another
  // external library parses its flags. `--help` will remain in the flag set
  // such that the subsequent parsing can find it.
  IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP = 1u << 1,
};
typedef uint32_t iree_flags_parse_mode_t;

// Sets the usage information printed when --help is passed on the command line.
// Both strings must remain live for the lifetime of the program.
void iree_flags_set_usage(const char* program_name, const char* usage);

// Parses flags from the given command line arguments.
// All flag-style arguments ('--foo', '-f', etc) will be consumed and argc/argv
// will be updated to contain only the program name (index 0) and any remaining
// positional arguments.
//
// Returns 0 if all flags were parsed and execution should continue.
// Returns >0 if execution should be cancelled such as when --help is used.
// Returns <0 if parsing fails.
//
// Usage:
//   extern "C" int main(int argc, char** argv) {
//     iree_status_t status = iree_flags_parse(&argc, &argv);
//     if (!iree_status_is_ok(status)) { exit(1); }
//     consume_positional_args(argc, argv);
//     return 0;
//   }
//
// Example:
//   argc = 4, argv = ['program', 'abc', '--flag=2']
// Results:
//   argc = 2, argv = ['program', 'abc']
iree_status_t iree_flags_parse(iree_flags_parse_mode_t mode, int* argc,
                               char*** argv);

// Parses flags as with iree_flags_parse but will use exit() or abort().
// WARNING: this almost always what you want in a command line tool and *never*
// what you want when embedded in a host process. You don't want to have a flag
// typo and shut down your entire server/sandbox/Android app/etc.
void iree_flags_parse_checked(iree_flags_parse_mode_t mode, int* argc,
                              char*** argv);

// Dumps all flags and their current values to the given |file|.
void iree_flags_dump(iree_flag_dump_mode_t mode, FILE* file);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_INTERNAL_FLAGS_H_
