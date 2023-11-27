// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/flags.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"

#if IREE_FLAGS_ENABLE_CLI == 1

#include "iree/base/internal/debugging.h"

//===----------------------------------------------------------------------===//
// Flag manipulation utilities
//===----------------------------------------------------------------------===//

static iree_status_t iree_flags_leaky_allocator_ctl(
    void* self, iree_allocator_command_t command, const void* params,
    void** inout_ptr) {
  IREE_LEAK_CHECK_DISABLE_PUSH();
  iree_status_t status =
      iree_allocator_system_ctl(/*self=*/NULL, command, params, inout_ptr);
  IREE_LEAK_CHECK_DISABLE_POP();
  return status;
}

// Allocates heap memory that is leaked without triggering leak checkers.
// We do this so that we have valid memory for the lifetime of the process.
// The memory may still be freed but if not will not hurt anything (besides the
// private working set size).
static iree_allocator_t iree_flags_leaky_allocator(void) {
  iree_allocator_t allocator = {
      .self = NULL,
      .ctl = iree_flags_leaky_allocator_ctl,
  };
  return allocator;
}

//===----------------------------------------------------------------------===//
// Flag registry
//===----------------------------------------------------------------------===//

// Storage for registered flags.
typedef struct iree_flag_t {
  // __FILE__ of flag definition.
  const char* file;
  // __LINE__ of flag definition.
  int line;
  // Defines what data is at |storage| and how to parse/print it.
  iree_flag_type_t type;
  // Registered callback to issue when the flag is parsed, if any.
  iree_flag_parse_callback_fn_t parse_callback;
  // Registered callback to issue when the flag is to be printed, if any.
  iree_flag_print_callback_fn_t print_callback;
  // Direct reference to the variable storing the flag value of |type|.
  void* storage;
  // Name of the flag on the command line ('foo' => '--foo=value').
  iree_string_view_t name;
  // Short description string.
  iree_string_view_t description;
} iree_flag_t;

// State used for flag registration and reflection.
typedef struct iree_flag_registry_t {
  const char* program_name;
  const char* usage;

  // Total number of entries in the |flags| list.
  int flag_count;
  // All registered flags in the executable in an undefined order.
  iree_flag_t flags[IREE_FLAGS_CAPACITY];
} iree_flag_registry_t;

// Global flags state.
// This will persist for the lifetime of the program so that flags can be
// reparsed/dumped. If you're concerned about the .data overhead then you
// probably just want to disable the CLI support for flags entirely.
static iree_flag_registry_t iree_flag_registry = {
    .program_name = NULL,
    .usage = NULL,
    .flag_count = 0,
};

int iree_flag_register(const char* file, int line, iree_flag_type_t type,
                       void* storage,
                       iree_flag_parse_callback_fn_t parse_callback,
                       iree_flag_print_callback_fn_t print_callback,
                       iree_string_view_t name,
                       iree_string_view_t description) {
  // TODO(benvanik): make the registry a linked list and externalize the
  // flag storage - then no need for a fixed count. If you're hitting this then
  // file an issue :)
  iree_flag_registry_t* registry = &iree_flag_registry;
  IREE_ASSERT_LE(registry->flag_count + 1, IREE_FLAGS_CAPACITY,
                 "flag registry overflow; too many flags registered");
  int flag_ordinal = registry->flag_count++;
  iree_flag_t* flag = &registry->flags[flag_ordinal];
  flag->file = file;
  flag->line = line;
  flag->type = type;
  flag->parse_callback = parse_callback;
  flag->print_callback = print_callback;
  flag->storage = storage;
  flag->name = name;
  flag->description = description;
  return flag_ordinal;
}

// Returns the flag registration with the given |name| or NULL if not found.
static iree_flag_t* iree_flag_lookup(iree_string_view_t name) {
  iree_flag_registry_t* registry = &iree_flag_registry;
  for (int i = 0; i < registry->flag_count; ++i) {
    iree_flag_t* flag = &registry->flags[i];
    if (iree_string_view_equal(flag->name, name)) {
      return flag;
    }
  }
  return NULL;
}

static int iree_flag_cmp(const void* lhs_ptr, const void* rhs_ptr) {
  const iree_flag_t* lhs = (const iree_flag_t*)lhs_ptr;
  const iree_flag_t* rhs = (const iree_flag_t*)rhs_ptr;
  int ret = strcmp(lhs->file, rhs->file);
  if (ret == 0) {
    return lhs->line - rhs->line;
  }
  return ret;
}

// Sorts the flags in the flag registry by file > line.
static void iree_flag_registry_sort(iree_flag_registry_t* registry) {
  qsort(registry->flags, registry->flag_count, sizeof(iree_flag_t),
        iree_flag_cmp);
}

//===----------------------------------------------------------------------===//
// List parsing/printing
//===----------------------------------------------------------------------===//

iree_status_t iree_flag_string_list_parse(iree_string_view_t flag_name,
                                          void* storage,
                                          iree_string_view_t value) {
  iree_flag_string_list_storage_t* flag =
      (iree_flag_string_list_storage_t*)storage;
  if (flag->count == 0) {  // currently empty
    // Inline storage (common case).
    flag->count = 1;
    flag->inline_value = value;
  } else if (flag->count == 1) {  // currently inline
    // Switching from inline storage to external storage.
    iree_host_size_t new_capacity = 4;
    iree_string_view_t* values = NULL;
    IREE_TRACE_ZONE_BEGIN_NAMED(z_outline,
                                "iree_flag_string_list_parse_outline");
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z_outline,
        iree_allocator_malloc(iree_flags_leaky_allocator(),
                              sizeof(iree_string_view_t) * new_capacity,
                              (void**)&values));
    values[0] = flag->inline_value;
    flag->capacity = new_capacity;
    flag->values = values;
    flag->values[flag->count++] = value;
    IREE_TRACE_ZONE_END(z_outline);
  } else if (flag->count < flag->capacity) {  // external storage available
    // Stash in external storage list.
    flag->values[flag->count++] = value;
  } else {  // external storage full
    // Growing external storage list.
    IREE_TRACE_ZONE_BEGIN_NAMED(z_grow, "iree_flag_string_list_parse_grow");
    iree_host_size_t new_capacity = iree_max(4, flag->capacity * 2);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z_grow,
        iree_allocator_realloc(iree_flags_leaky_allocator(),
                               sizeof(iree_string_view_t) * new_capacity,
                               (void**)&flag->values));
    flag->capacity = new_capacity;
    flag->values[flag->count++] = value;
    IREE_TRACE_ZONE_END(z_grow);
  }
  return iree_ok_status();
}

void iree_flag_string_list_print(iree_string_view_t flag_name, void* storage,
                                 FILE* file) {
  iree_flag_string_list_storage_t* flag =
      (iree_flag_string_list_storage_t*)storage;
  if (flag->count == 0) {
    fprintf(file, "# --%.*s=...\n", (int)flag_name.size, flag_name.data);
  } else if (flag->count == 1) {
    fprintf(file, "--%.*s=%.*s\n", (int)flag_name.size, flag_name.data,
            (int)flag->inline_value.size, flag->inline_value.data);
  } else {
    for (iree_host_size_t i = 0; i < flag->count; ++i) {
      const iree_string_view_t value = flag->values[i];
      fprintf(file, "--%.*s=%.*s\n", (int)flag_name.size, flag_name.data,
              (int)value.size, value.data);
    }
  }
}

//===----------------------------------------------------------------------===//
// Flag parsing/printing
//===----------------------------------------------------------------------===//

void iree_flags_set_usage(const char* program_name, const char* usage) {
  iree_flag_registry_t* registry = &iree_flag_registry;
  registry->program_name = program_name;
  registry->usage = usage;
}

// Parses a flag value from the given string and stores it.
static iree_status_t iree_flag_parse(iree_flag_t* flag,
                                     iree_string_view_t value) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, flag->name.data, flag->name.size);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, value.data, value.size);

  // Insert NUL on the flag value. This is safe as the value is either coming
  // from C argv memory which is mutable or a flagfile that we loaded into
  // memory ourselves.
  char* str_value = (char*)value.data;
  if (value.size > 0) {
    str_value[value.size] = 0;
  }

  iree_status_t status = iree_ok_status();
  switch (flag->type) {
    case IREE_FLAG_TYPE_callback:
      status = flag->parse_callback(flag->name, flag->storage, value);
      break;
    case IREE_FLAG_TYPE_bool:
      if (value.size == 0 || strcmp(str_value, "true") == 0 ||
          strcmp(str_value, "1") == 0) {
        *(bool*)flag->storage = true;
      } else {
        *(bool*)flag->storage = false;
      }
      break;
    case IREE_FLAG_TYPE_int32_t:
      *(int32_t*)flag->storage = value.size ? atoi(str_value) : 0;
      break;
    case IREE_FLAG_TYPE_int64_t:
      *(int64_t*)flag->storage = value.size ? atoll(str_value) : 0;
      break;
    case IREE_FLAG_TYPE_float:
      *(float*)flag->storage = value.size ? (float)atof(str_value) : 0.0f;
      break;
    case IREE_FLAG_TYPE_double:
      *(double*)flag->storage = value.size ? atof(str_value) : 0.0;
      break;
    case IREE_FLAG_TYPE_string: {
      iree_host_size_t str_length = value.size;
      if (str_length > 2) {
        // Strip double quotes: "foo" -> foo.
        // This may not be worth the complexity.
        if (str_value[0] == '"' && str_value[str_length - 1] == '"') {
          str_value[str_length - 1] = 0;
          ++str_value;
          str_length = str_length - 2;
        }
      }
      *(const char**)flag->storage = str_value;
      break;
    }
    default:
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "invalid flag type %u", flag->type);
      break;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Prints a flag value to |file| (like 'true' or '5.43').
static void iree_flag_print(FILE* file, iree_flag_t* flag) {
  if (flag->type == IREE_FLAG_TYPE_callback) {
    flag->print_callback(flag->name, flag->storage, file);
    return;
  }
  fprintf(file, "--%.*s", (int)flag->name.size, flag->name.data);
  if (flag->storage == NULL) return;
  switch (flag->type) {
    case IREE_FLAG_TYPE_bool:
      fprintf(file, "=%s", (*(bool*)flag->storage) ? "true" : "false");
      break;
    case IREE_FLAG_TYPE_int32_t:
      fprintf(file, "=%" PRId32, *(int32_t*)flag->storage);
      break;
    case IREE_FLAG_TYPE_int64_t:
      fprintf(file, "=%" PRId64, *(int64_t*)flag->storage);
      break;
    case IREE_FLAG_TYPE_float:
      fprintf(file, "=%g", *(float*)flag->storage);
      break;
    case IREE_FLAG_TYPE_double:
      fprintf(file, "=%g", *(double*)flag->storage);
      break;
    case IREE_FLAG_TYPE_string:
      fprintf(file, "=\"%s\"", *(const char**)flag->storage);
      break;
    default:
      fprintf(file, "=<INVALID>");
      break;
  }
  fprintf(file, "\n");
}

// Dumps a flag definition and value to |file|.
static void iree_flag_dump(iree_flag_dump_mode_t mode, FILE* file,
                           iree_flag_t* flag) {
  if (iree_all_bits_set(mode, IREE_FLAG_DUMP_MODE_VERBOSE)) {
    if (!iree_string_view_is_empty(flag->description)) {
      iree_string_view_t description = flag->description;
      while (!iree_string_view_is_empty(description)) {
        iree_string_view_t line;
        iree_string_view_split(description, '\n', &line, &description);
        if (!iree_string_view_is_empty(line)) {
          fprintf(file, "# %.*s\n", (int)line.size, line.data);
        }
      }
    }
  }
  iree_flag_print(file, flag);
}

static iree_status_t iree_flags_parse_help(iree_string_view_t flag_name,
                                           void* storage,
                                           iree_string_view_t value) {
  iree_flag_registry_t* registry = &iree_flag_registry;

  fprintf(stdout,
          "# "
          "===================================================================="
          "========\n");
  fprintf(stdout, "# 👻 IREE: %s\n",
          registry->program_name ? registry->program_name : "");
  fprintf(stdout,
          "# "
          "===================================================================="
          "========\n\n");
  if (registry->usage) {
    fprintf(stdout, "%s\n", registry->usage);
  }
  iree_flags_dump(IREE_FLAG_DUMP_MODE_VERBOSE, stdout);
  fprintf(stdout, "\n");

  return iree_ok_status();
}
static void iree_flags_print_help(iree_string_view_t flag_name, void* storage,
                                  FILE* file) {
  fprintf(file, "# --%.*s\n", (int)flag_name.size, flag_name.data);
}
IREE_FLAG_CALLBACK(iree_flags_parse_help, iree_flags_print_help, NULL, help,
                   "Displays command line usage information.");

// Removes argument |arg| from the argument list.
static void iree_flags_remove_arg(int arg, int* argc_ptr, char*** argv_ptr) {
  int argc = *argc_ptr;
  char** argv = *argv_ptr;
  memmove(&argv[arg], &argv[arg + 1], (argc - arg - 1) * sizeof(char*));
  *argc_ptr = argc - 1;
}

iree_status_t iree_flags_parse(iree_flags_parse_mode_t mode, int* argc_ptr,
                               char*** argv_ptr) {
  if (argc_ptr == NULL || argv_ptr == NULL || *argc_ptr == 0) {
    // No flags; that's fine - in some environments flags aren't supported.
    return iree_ok_status();
  }

  // Always sort the registry; though we may parse flags multiple times this is
  // not a hot path and this is easier than trying to keep track of whether we
  // need to or not.
  iree_flag_registry_sort(&iree_flag_registry);

  int argc = *argc_ptr;
  char** argv = *argv_ptr;

  for (int arg_ordinal = 1; arg_ordinal < argc; ++arg_ordinal) {
    iree_string_view_t arg = iree_make_cstring_view(argv[arg_ordinal]);

    // Strip whitespace.
    arg = iree_string_view_trim(arg);

    // Position arguments are ignored; they may appear anywhere in the list.
    if (!iree_string_view_starts_with(arg, iree_make_cstring_view("--"))) {
      continue;
    }

    // Strip `--`.
    arg = iree_string_view_remove_prefix(arg, 2);

    // Split into `flag_name` = `flag_value`.
    iree_string_view_t flag_name;
    iree_string_view_t flag_value;
    iree_string_view_split(arg, '=', &flag_name, &flag_value);
    flag_name = iree_string_view_trim(flag_name);
    flag_value = iree_string_view_trim(flag_value);

    // Lookup the flag by name.
    iree_flag_t* flag = iree_flag_lookup(flag_name);
    if (!flag) {
      // If --undefok allows undefined flags then we just skip this one. Note
      // that we leave it in the argument list so that subsequent flag parsers
      // can try to handle it.
      if (iree_all_bits_set(mode, IREE_FLAGS_PARSE_MODE_UNDEFINED_OK)) {
        continue;
      }
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "flag '%.*s' not recognized", (int)flag_name.size,
                              flag_name.data);
    }

    // Parse and store the flag value.
    IREE_RETURN_IF_ERROR(iree_flag_parse(flag, flag_value));

    // --help gets special handling due to interop with external libraries that
    // may also need to find it. If indicated we keep --help in the argument
    // list and don't exit.
    if (iree_string_view_equal(flag_name, iree_make_cstring_view("help"))) {
      if (iree_all_bits_set(mode, IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP)) {
        continue;  // don't remove the arg below
      }
      exit(0);  // --help exits by default.
    }

    // Splice out the flag from the argv list.
    iree_flags_remove_arg(arg_ordinal, &argc, &argv);
    --arg_ordinal;
  }

  *argc_ptr = argc;
  return iree_ok_status();
}

void iree_flags_parse_checked(iree_flags_parse_mode_t mode, int* argc,
                              char*** argv) {
  IREE_TRACE_ZONE_BEGIN(z0);
  for (int i = 0; i < *argc; ++i) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, (*argv)[i]);
  }
  iree_status_t status = iree_flags_parse(mode, argc, argv);
  IREE_TRACE_ZONE_END(z0);
  if (iree_status_is_ok(status)) return;

  fprintf(stderr, "\x1b[31mFLAGS ERROR: (╯°□°)╯︵👻\x1b[0m\n");
  iree_status_fprint(stderr, status);
  fflush(stderr);

  exit(EXIT_FAILURE);
}

void iree_flags_dump(iree_flag_dump_mode_t mode, FILE* file) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Always sort the registry; though we may dump flags multiple times this is
  // not a hot path and this is easier than trying to keep track of whether we
  // need to or not.
  iree_flag_registry_sort(&iree_flag_registry);

  const char* last_file = NULL;
  for (size_t i = 0; i < iree_flag_registry.flag_count; ++i) {
    iree_flag_t* flag = &iree_flag_registry.flags[i];
    if (iree_all_bits_set(mode, IREE_FLAG_DUMP_MODE_VERBOSE)) {
      if (last_file) {
        fprintf(file, "\n");
      }
      if (!last_file || strcmp(flag->file, last_file) != 0) {
        fprintf(file,
                "# "
                "===-----------------------------------------------------------"
                "-----------===\n");
        fprintf(file, "# Flags in %s\n", flag->file);
        fprintf(file,
                "# "
                "===-----------------------------------------------------------"
                "-----------===\n\n");
        last_file = flag->file;
      }
    }
    iree_flag_dump(mode, file, flag);
  }

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// --flagfile= support
//===----------------------------------------------------------------------===//
// NOTE: this is conditionally enabled as some platforms may not have IO.

#if IREE_FLAGS_ENABLE_FLAG_FILE == 1

#include "iree/base/internal/file_io.h"

// Parses a newline-separated list of flags from a file.
static iree_status_t iree_flags_parse_file(iree_string_view_t file_path) {
  // Read file contents.
  // NOTE: we intentionally leak the contents here so that the flags remain in
  // memory in case they are referenced.
  // NOTE: safe to use file_path.data here as it will always have a NUL
  // terminator.
  iree_allocator_t allocator = iree_flags_leaky_allocator();
  iree_file_contents_t* file_contents = NULL;
  IREE_RETURN_IF_ERROR(
      iree_file_read_contents(file_path.data, IREE_FILE_READ_FLAG_DEFAULT,
                              allocator, &file_contents),
      "while trying to parse flagfile");

  // Run through the file line-by-line.
  int line_number = 0;
  iree_string_view_t contents =
      iree_make_string_view((const char*)file_contents->buffer.data,
                            file_contents->buffer.data_length);
  while (!iree_string_view_is_empty(contents)) {
    // Split into a single line and the entire rest of the file contents.
    iree_string_view_t line;
    iree_string_view_split(contents, '\n', &line, &contents);
    ++line_number;

    // Strip whitespace.
    line = iree_string_view_trim(line);
    if (iree_string_view_is_empty(line)) continue;

    // Ignore comments.
    if (iree_string_view_starts_with(line, iree_make_cstring_view("#")) ||
        iree_string_view_starts_with(line, iree_make_cstring_view("//"))) {
      continue;
    }

    // Strip `--`.
    if (!iree_string_view_starts_with(line, iree_make_cstring_view("--"))) {
      // Positional arguments can't be specified in flag files.
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "%.*s:%d: positional arguments not allowed in flag files",
          (int)file_path.size, file_path.data, line_number);
    }
    line = iree_string_view_remove_prefix(line, 2);

    // Split into `flag_name` = `flag_value`.
    iree_string_view_t flag_name;
    iree_string_view_t flag_value;
    iree_string_view_split(line, '=', &flag_name, &flag_value);
    flag_name = iree_string_view_trim(flag_name);
    flag_value = iree_string_view_trim(flag_value);

    // Lookup the flag by name.
    iree_flag_t* flag = iree_flag_lookup(flag_name);
    if (!flag) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s:%d: flag '%.*s' not recognized",
                              (int)file_path.size, file_path.data, line_number,
                              (int)flag_name.size, flag_name.data);
    }

    // Parse the flag value.
    IREE_RETURN_IF_ERROR(iree_flag_parse(flag, flag_value),
                         "%.*s:%d: while parsing flag '%.*s'",
                         (int)file_path.size, file_path.data, line_number,
                         (int)line.size, line.data);
  }

  // NOTE: we intentionally leak the memory as flags may continue to reference
  // segments of it for their string values.
  return iree_ok_status();
}

static iree_status_t iree_flags_parse_flagfile(iree_string_view_t flag_name,
                                               void* storage,
                                               iree_string_view_t value) {
  if (iree_string_view_is_empty(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--%.*s= requires a file path", (int)flag_name.size,
                            flag_name.data);
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, value.data, value.size);
  iree_status_t status = iree_flags_parse_file(value);
  IREE_TRACE_ZONE_END(z0);

  return status;
}
static void iree_flags_print_flagfile(iree_string_view_t flag_name,
                                      void* storage, FILE* file) {
  fprintf(file, "# --%.*s=[path]\n", (int)flag_name.size, flag_name.data);
}
IREE_FLAG_CALLBACK(iree_flags_parse_flagfile, iree_flags_print_flagfile, NULL,
                   flagfile,
                   "Parses a newline-separated list of flags from a file.\n"
                   "Flags are parsed at the point where the flagfile is "
                   "specified\nand following flags may override the parsed "
                   "values.\nNOTE: this --help output is a flagfile! Pipe "
                   "this to a file, tweak the\noptions from their defaults, "
                   "and pass it back in using --flagfile=.");

#endif  // IREE_FLAGS_ENABLE_FLAG_FILE

#else

void iree_flags_set_usage(const char* program_name, const char* usage) {}

int iree_flags_parse(iree_flags_parse_mode_t mode, int* argc, char*** argv) {
  return 0;
}

void iree_flags_parse_checked(iree_flags_parse_mode_t mode, int* argc,
                              char*** argv) {}

void iree_flags_dump(iree_flag_dump_mode_t mode, FILE* file) {}

#endif  // IREE_FLAGS_ENABLE_CLI
