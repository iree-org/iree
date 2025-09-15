// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_H_
#define IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_H_

// NOTE: this file is designed to be a standalone header: it is embedded in the
// compiler and must not take any dependencies on the runtime HAL code.
// Changes here will require changes to the compiler and must be versioned as if
// this was a schema: backwards-incompatible changes require version bumps or
// the ability to feature-detect at runtime.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

//===----------------------------------------------------------------------===//
// Common utilities included to reduce dependencies
//===----------------------------------------------------------------------===//

// `restrict` keyword, not supported by some older compilers.
// We define our own macro in case dependencies use `restrict` differently.
#if defined(_MSC_VER) && _MSC_VER >= 1900
#define IREE_RESTRICT __restrict
#elif defined(_MSC_VER)
#define IREE_RESTRICT
#elif defined(__cplusplus)
#define IREE_RESTRICT __restrict__
#else
#define IREE_RESTRICT restrict
#endif  // _MSC_VER

//===----------------------------------------------------------------------===//
// Runtime feature support metadata
//===----------------------------------------------------------------------===//

// Defines a bitfield of features that the library requires or supports.
enum iree_hal_executable_library_feature_bits_t {
  IREE_HAL_EXECUTABLE_LIBRARY_FEATURE_NONE = 0u,
  // TODO(benvanik): declare features for debugging/coverage/printf/etc.
  // These will control which symbols are injected into the library at runtime.
};
typedef uint32_t iree_hal_executable_library_features_t;

// Defines a set of supported sanitizers that libraries may be compiled with.
// Loaders can use this declaration to check as to whether the library is
// compatible with the hosting environment for cases where the sanitizer
// requires host support.
typedef enum iree_hal_executable_library_sanitizer_kind_e {
  IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE = 0,
  // Indicates the library is compiled to use AddressSanitizer:
  // https://clang.llvm.org/docs/AddressSanitizer.html
  // Equivalent compiler flag: -fsanitize=address
  IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_ADDRESS = 1,
  // Indicates the library is compiled to use MemorySanitizer:
  // https://clang.llvm.org/docs/MemorySanitizer.html
  // Equivalent compiler flag: -fsanitize=memory
  IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_MEMORY = 2,
  // Indicates the library is compiled to use ThreadSanitizer:
  // https://clang.llvm.org/docs/ThreadSanitizer.html
  // Equivalent compiler flag: -fsanitize=thread
  IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_THREAD = 3,
  // Indicates the library is compiled to use UndefinedBehaviorSanitizer:
  // https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
  // Equivalent compiler flag: -fsanitize=undefined
  IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_UNDEFINED = 4,

  IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_MAX_ENUM = INT32_MAX,
} iree_hal_executable_library_sanitizer_kind_t;

//===----------------------------------------------------------------------===//
// Versioning and interface querying
//===----------------------------------------------------------------------===//

typedef struct iree_hal_executable_environment_v0_t
    iree_hal_executable_environment_v0_t;

// Version code indicating the minimum required runtime structures.
// Runtimes cannot load executables with newer versions but may be able to load
// older versions if backward compatibility is enabled.
//
// NOTE: until we hit v1 the versioning scheme here is not set in stone.
// We may want to make this major release number, date codes (0x20220307),
// or some semantic versioning we track in whatever spec we end up having.
typedef uint32_t iree_hal_executable_library_version_t;

#define IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0_6 0x00000006u

// The latest version of the library API; can be used to populate the
// iree_hal_executable_library_header_t::version when building libraries.
#define IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST \
  IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0_6

// A header present at the top of all versions of the library API used by the
// runtime to ensure version compatibility.
typedef struct iree_hal_executable_library_header_t {
  // Version of the API this library was built with, which was likely the value
  // of IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST.
  iree_hal_executable_library_version_t version;

  // Name used for logging/diagnostics.
  const char* name;

  // Bitfield of features required/supported by this executable.
  iree_hal_executable_library_features_t features;

  // Which sanitizer the library is compiled to use, if any.
  // Libraries meant for use with a particular sanitizer will are only usable
  // with hosting code that is using the same sanitizer.
  iree_hal_executable_library_sanitizer_kind_t sanitizer;
} iree_hal_executable_library_header_t;

// Exported function from dynamic libraries for querying library information.
//
// The provided |max_version| is the maximum version the caller supports;
// callees must return NULL if their lowest available version is greater
// than the max version supported by the caller.
//
// The provided |environment| field contains information about the hosting
// execution environment that the executable may use to specialize its
// implementation, such as using specific imports or exporting
// architecture-specific dispatch routines. Some environmental properties may
// change per-invocation such as the CPU info when performing dispatches on
// heterogenous processors that may change over the lifetime of the program.
typedef const iree_hal_executable_library_header_t** (
    *iree_hal_executable_library_query_fn_t)(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment);

// Function name exported from dynamic libraries (pass to dlsym).
#define IREE_HAL_EXECUTABLE_LIBRARY_EXPORT_NAME \
  "iree_hal_executable_library_query"

//===----------------------------------------------------------------------===//
// IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0_*
//===----------------------------------------------------------------------===//

// Function signature of imported functions for use in the executable.
// Each call takes opaque parameters as defined by the imported function.
// Both the compiler and the runtime must agree on the parameter format
// (including struct alignment and packing) and doing so is outside the scope
// of this API. In general one should only pass precisely what they need
// (pointers directly into buffers being manipulated, arguments, etc) and not
// try to replicate the dispatch structure (workgroup information and bindings)
// so that the imported functions can be versioned independently from this
// specification.
//
// Returns 0 on success and non-zero on failure. Failures will cause device loss
// and should only be used to communicate serious issues that should abort all
// execution within the current device. Buffer overflows are a good example of
// a useful failure though the HAL does not mandate that all overflows are
// caught and only that they are not harmful - clamping byte ranges and never
// returning a failure is sufficient.
typedef int (*iree_hal_executable_import_v0_t)(void* params, void* context,
                                               void* reserved);

// A thunk function used to call an import.
// All imports must be called through this function by passing the import
// function pointer as the first argument followed by the arguments of the
// import function itself.
typedef int (*iree_hal_executable_import_thunk_v0_t)(
    iree_hal_executable_import_v0_t fn_ptr, void* params, void* context,
    void* reserved);

// Declares imports available to the executable library at runtime.
// To enable linker isolation, ABI shimming, and import multi-versioning we use
// this import table exclusively and do not allow platform-level linking. If it
// were allowed the deployment situation gets significantly more complex as the
// libraries containing the imported symbols will differ on all platforms, will
// have the platform-dependent ABI (Windows, MacOS, etc), and may not be
// available at all (bare-metal).
//
// Static libraries may choose to still dynamically link against external
// symbols without using this table as in that scenario much of the above
// concerns do not apply: all code is being linked together into the same binary
// and symbol availability is known during build-time linking. Static linking
// also enables LTO to strip any import not used by any executables in contrast
// to the dynamic style elsewhere.
//
// Represented as a struct-of-arrays for more efficient packing and more
// locality during lookup. Each subarray - when not omitted and NULL - is
// indexed by import ordinal and has up to |count| entries.
typedef struct iree_hal_executable_import_table_v0_t {
  // Total number of imports in the table.
  uint32_t count;

  // Import symbol name encoding the name and whether it is weak.
  // Example: `?mylib_some_fn_v2`
  //   `?`:
  //     Indicates when an import is optional. If the import of the specified
  //     version is not found the table entry will be NULL. When omitted if the
  //     import is unavailable loading will fail.
  //   `mylib_...`:
  //     Prefix indicating the owner of the function; symbols have a global
  //     namespace and this is used to reduce collisions.
  //   `some_fn...`:
  //     Name of the function used to link to the imports available in the
  //     hosting executable.
  //   `..._v2`:
  //     Function-specified version number used to allow multiple versions to
  //     to be imported. For backward compatibility one could import both
  //     `some_fn_v1?` and `some_fn_v2?` and use whichever is available.
  //     Note that this is just a convention for the suffix and can be anything.
  //
  // The symbol table is sorted ascending alphabetical (by strcmp).
  const char* const* symbols;
} iree_hal_executable_import_table_v0_t;

// Maximum number of data fields in iree_hal_processor_v0_t.
#define IREE_HAL_PROCESSOR_DATA_CAPACITY_V0 8

// Architecture-specific CPU information available to executables.
// This encodes zero or more fields of opaque processor data.
// The intent is that this structure can be put in .rodata when there are no
// runtime features that need to be queried.
//
// The format of the data is architecture-specific as by construction no value
// will ever be used in a compiled binary from another architecture. This
// allows us to simplify this interface as we can't for example load the same
// executable library for both aarch64 on riscv32 and don't need to normalize
// any of the fields across them both.
//
// See iree/schemas/cpu_data.h for details.
typedef struct iree_hal_processor_v0_t {
  // Opaque architecture-specific encoding in 64-bit words.
  // This may represent a fixed-length data structure, a series of hardware
  // registers, or key-value pairs.
  //
  // The contents are opaque here as to support out-of-tree architectures. The
  // runtime code deriving the identifier/flags and providing it here is loosely
  // coupled with the compiler code emitting checks based on the identifier and
  // only those two places ever need to change.
  uint64_t data[IREE_HAL_PROCESSOR_DATA_CAPACITY_V0];
} iree_hal_processor_v0_t;
static_assert(sizeof(iree_hal_processor_v0_t) % sizeof(uint64_t) == 0,
              "8-byte alignment required");

// Defines the environment in which the executable is being used.
// Executables only have access to the information in this structure and must
// make all decisions based on it; this ensures executables are portable across
// operating environments (Linux, Mac, bare-metal, web, etc) by not having
// platform-specific syscalls and register query emulation.
typedef struct iree_hal_executable_environment_v0_t {
  // Specialization constants available to the executable, if any.
  // Contains as many as declared in the library header.
  const uint32_t* constants;

  // Thunk function for calling imports. All calls must be made through this.
  iree_hal_executable_import_thunk_v0_t import_thunk;
  // Optional imported functions available for use within the executable.
  // Contains one entry per imported function. If an import was marked as weak
  // then the corresponding entry may be NULL.
  const iree_hal_executable_import_v0_t* import_funcs;
  const void** import_contexts;

  // Optional architecture-specific CPU information.
  // In heterogenous processors this may represent any of the subarchitecture
  // types as it is derived from the core the calling thread is scheduled on.
  // Will be all zeros if unavailable.
  iree_hal_processor_v0_t processor;
} iree_hal_executable_environment_v0_t;

// Read-only per-dispatch state passed to each workgroup in a dispatch.
//
// We layout to try to fit everything commonly used into the first cache line
// (on archs with 64-bit pointers; 32-bit fits in a single line).
//
// For workgroup dimensions we allow the full 32-bit range on X and Y as those
// are the primary distribution dimensions. Z is the coarsest control and is
// usually in the 1-16 range; any higher and it can pessimize scheduling. Almost
// all GPUs also have this limitation (max Z of 65K) for the same reason.
typedef struct iree_hal_executable_dispatch_state_v0_t {
  // Workgroup size chosen for the dispatch. For compilation modes where the
  // workgroup size is constant this may be ignored.
  uint32_t workgroup_size_x;
  uint32_t workgroup_size_y;
  uint16_t workgroup_size_z;

  // Total number of available 4 byte push constant values in |constants|.
  uint16_t constant_count;

  // Total workgroup count for the dispatch. This is sourced from either the
  // original dispatch call (for iree_hal_command_buffer_dispatch) or the
  // indirection buffer (for iree_hal_command_buffer_dispatch_indirect).
  uint32_t workgroup_count_x;
  uint32_t workgroup_count_y;
  uint16_t workgroup_count_z;

  // Estimated maximum concurrent workgroups; loosely maps to the number of
  // processors allowed to execute the dispatch. The actual number will vary
  // based on competing dispatches and dynamic executor configuration.
  uint8_t max_concurrency;

  // Total number of binding base pointers in |binding_ptrs| and
  // |binding_lengths|. The set is packed densely based on which bindings are
  // used (known at compile-time).
  uint8_t binding_count;

  // |constant_count| values.
  const uint32_t* constants;
  // Base pointers to each binding buffer.
  void* const* binding_ptrs;
  // The length of each binding in bytes, 1:1 with |binding_ptrs|.
  const size_t* binding_lengths;

  // NOTE: the above fields are frequently accessed and should be kept together
  // to ensure cache-friendly behavior. The first instructions every dispatch
  // executes are loads from the fields and we want to avoid a cascade of
  // cache misses. Less-frequently used fields can follow.
} iree_hal_executable_dispatch_state_v0_t;
static_assert(sizeof(iree_hal_executable_dispatch_state_v0_t) <= 64,
              "try keeping dispatch state small enough to fit in a cache line");

// Read-only per-workgroup state passed to each workgroup in a dispatch.
//
// We layout to try to fit everything commonly used into the first cache line
// (on archs with 64-bit pointers; 32-bit fits in a single line).
typedef struct iree_hal_executable_workgroup_state_v0_t {
  // Workgroup ID of the currently executing workgroup.
  // This is in the range of 0-workgroup_count and each unique workgroup is to
  // perform workgroup_size invocations.
  uint32_t workgroup_id_x;
  uint32_t workgroup_id_y;
  uint16_t workgroup_id_z;

  // Reserved for future use.
  uint16_t reserved;

  // Logical processor identifier used to index into processor info fields.
  // Depending on the implementation this may be an ordinal, a bitfield, or an
  // opaque unique identifier.
  //
  // NOTE: we could steal bits from the |processor_id| if needed; today the ID
  // is the global ID but it really only needs to be within the current node
  // (8-bits, or 16-bit for single-node thousand-core future proofing).
  uint32_t processor_id;

  // Scratch memory available for use by the workgroup.
  // Requires a non-zero value to be specified for |local_memory_pages|; at
  // least the size specified will be available. This memory is transient and
  // exclusive to the workgroup. The provided pointer may be NULL if no
  // workgroup local memory was requested.
  void* local_memory;
  // Total number of bytes available in |local_memory|. This may be larger than
  // the requested amount.
  uint32_t local_memory_size;

  // +4 trailing bytes of free space
} iree_hal_executable_workgroup_state_v0_t;
static_assert(
    sizeof(iree_hal_executable_workgroup_state_v0_t) <= 64,
    "try keeping workgroup state small enough to fit in a cache line");

// Function signature of exported executable entry points.
// The same |environment| is passed to all dispatches.
// The same |dispatch_state| is passed to all workgroups within a dispatch.
// A unique |workgroup_state| is passed to every workgroup within a dispatch.
//
// Returns 0 on success and non-zero on failure. Failures will cause device loss
// and should only be used to communicate serious issues that should abort all
// execution within the current device. Buffer overflows are a good example of
// a useful failure though the HAL does not mandate that all overflows are
// caught and only that they are not harmful - clamping byte ranges and never
// returning a failure is sufficient.
typedef int (*iree_hal_executable_dispatch_v0_t)(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state);

// Bytes per page of workgroup local memory.
// This is chosen to match the common page size of devices.
#define IREE_HAL_EXECUTABLE_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE 4096

// Maximum number of constants that can be used by a single dispatch.
#define IREE_HAL_EXECUTABLE_MAX_CONSTANT_COUNT 64
// Maximum number of bindings that can be used by a single dispatch.
#define IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT 64

// Flags defining dispatch behavior.
enum iree_hal_executable_dispatch_flag_v0_bits_e {
  IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE = 0ull,
  // Contiguous workgroups in workgroup space process data sequentially.
  // Dispatch performance can benefit from scheduling multiple contiguous
  // workgroups on execution units that share caches.
  IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_SEQUENTIAL = 1ull << 0,
  // Workgroup size is dynamic at dispatch time.
  // The workgroup size specified on the export info is the minimum size and
  // granularity and any dynamic workgroup size chosen must be a multiple.
  IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_WORKGROUP_SIZE_DYNAMIC = 1ull << 1,
};
typedef uint64_t iree_hal_executable_dispatch_flags_v0_t;

// Attributes for exported dispatch functions defining how they are to be
// executed. Required for all dispatches to specify constant and binding counts.
typedef struct iree_hal_executable_dispatch_attrs_v0_t {
  // Flags defining dispatch behavior.
  iree_hal_executable_dispatch_flags_v0_t flags;
  // Number of IREE_HAL_EXECUTABLE_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE byte pages
  // (or 0) indicating how much workgroup local memory is required for the
  // dispatch. This is the size of the buffer referenced by the `local_memory`
  // argument.
  uint16_t local_memory_pages;
  // Total number of 32-bit constants used by the dispatch.
  uint8_t constant_count;
  // Total number of bindings used by the dispatch.
  uint8_t binding_count;
  // Constant workgroup size if specified by the compiler.
  uint32_t workgroup_size_x;
  uint32_t workgroup_size_y;
  uint16_t workgroup_size_z;
  // Total number of logical parameters.
  // Indicates the size of the parameter array for this export in the export
  // table.
  uint16_t parameter_count;
  // Unused. Must be 0.
  uint64_t reserved_1[5];
} iree_hal_executable_dispatch_attrs_v0_t;
static_assert(sizeof(iree_hal_executable_dispatch_attrs_v0_t) <= 64,
              "try keeping dispatch attrs small enough to fit in a cache line");

// Specifies the type of a parameter.
enum iree_hal_executable_dispatch_parameter_type_v0_e {
  // Parameter is a constant uniform value.
  // Passed to the dispatch in the constants table. The offset indicates the
  // byte offset from the start of the constants table. The size is the total
  // bytes the constant occupies in the constant table without padding.
  IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT = 0,
  // Parameter is a buffer binding.
  // Passed to the dispatch in the binding_ptrs table and the length is
  // available. The offset indicates which binding in the table this parameter
  // maps to. The parameter size is ignored.
  IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING = 1,
  // Parameter is a raw buffer pointer.
  // Passed to the dispatch in the constants table and the length is
  // unavailable. The offset indicates the byte offset from the start of the
  // constants table. The size is the width in bytes of the pointer (always the
  // machine pointer width).
  IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BUFFER_PTR = 2,
};
typedef uint8_t iree_hal_executable_dispatch_parameter_type_v0_t;

// Defines parameter handling behavior.
enum iree_hal_executable_dispatch_parameter_flag_v0_bits_e {
  IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE = 0,
};
typedef uint16_t iree_hal_executable_dispatch_parameter_flags_v0_t;

// Declares properties of a parameter to a dispatch function.
typedef struct iree_hal_executable_dispatch_parameter_v0_t {
  // Type of the parameter.
  iree_hal_executable_dispatch_parameter_type_v0_t type;
  // Size of the parameter in bytes. Does not contain padding.
  uint8_t size;
  // Flags indicating parameter behavior.
  iree_hal_executable_dispatch_parameter_flags_v0_t flags;
  // Ordinal in the parameter name table or -1 if unnamed.
  uint16_t name;
  // Offset of the parameter in bytes or binding ordinal, depending on type.
  uint16_t offset;
} iree_hal_executable_dispatch_parameter_v0_t;

// Information used to calculate occupancy.
typedef struct iree_hal_executable_dispatch_occupancy_v0_t {
  // TODO(benvanik): implement this. It's optional and not used yet but here
  // as a placeholder in the format.
  int reserved;
} iree_hal_executable_dispatch_occupancy_v0_t;

// Source location information for a dispatch function indicating what code was
// used to generate it. This only represents a single source snapshot, of which
// there may be multiple valid possibilities (source program in Python, imported
// high level framework .mlir, LLVM bitcode, etc).
typedef struct iree_hal_executable_source_location_v0_t {
  // The line within the file at |path|.
  uint32_t line;
  // The length of |path|.
  uint32_t path_length;
  // The path (absolute or relative) to the source file, NUL-terminated.
  const char* path;
} iree_hal_executable_source_location_v0_t;

// Table of source locations keyed by a string compilation stage name.
// Locations are sorted ascending by name.
typedef struct iree_hal_executable_stage_location_table_v0_t {
  // Total number of source locations in the table.
  uint32_t count;
  // Names identifying the locations 1:1 with the locations set, NUL-terminated.
  const char* const* names;
  // Source locations matching 1:1 with the names.
  const iree_hal_executable_source_location_v0_t* locations;
} iree_hal_executable_stage_location_table_v0_t;

// A table of exported functions arranged as a struct-of-arrays for more
// efficient packing and faster lookup. Each subarray - when not omitted and
// NULL - is indexed by export ordinal and has up to |count| entries.
typedef struct iree_hal_executable_export_table_v0_t {
  // Total number of exports in the table.
  uint32_t count;

  // Function pointers for each exported entry point.
  const iree_hal_executable_dispatch_v0_t* ptrs;

  // Table of attributes 1:1 with ptrs.
  const iree_hal_executable_dispatch_attrs_v0_t* attrs;

  // Optional parameter declarations per entry point in original logical order.
  // The offset of each parameter for a particular entry point may jump around
  // as alignment and padding are accounted for. Each entry point has as many
  // parameters as the attributes constant_count + binding_count indicate.
  // When omitted the function is assumed to follow the IREE HAL ABI.
  //
  // Example of a raw C function:
  //   int my_dispatch(uint32_t a, uint64_t b, void* c);
  // Params:
  // - "a": type=CONSTANT, offset=0, size=4
  // - "b": type=CONSTANT, offset=8, size=8
  // - "c": type=BUFFER_PTR, offset=0, size=8  (64-bit system)
  //
  // Example of a (hypothetical) HAL ABI-aware C function:
  //   int my_dispatch(
  //       iree_hal_abi_workgroup_count_t wg_count,
  //       iree_hal_abi_workgroup_size_t wg_size,
  //       uint32_t a,
  //       uint64_t b,
  //       iree_hal_abi_binding_t c);
  // Params:
  // - "a": type=CONSTANT, offset=0, size=4
  // - "b": type=CONSTANT, offset=8, size=8
  // - "c": type=BINDING, offset=0
  //
  // The packing logic for producing constants and bindings from a list of raw
  // parameters roughly follows:
  //   for (param, value) in zip(params[entry_point], param_values):
  //     if param.type == constant || buffer_ptr:
  //       memcpy(constants[param.offset / 4], &value, param.size)
  //     elif param.type == binding:
  //       bindings[param.offset] = make_binding(value, IREE_HAL_WHOLE_BUFFER)
  const iree_hal_executable_dispatch_parameter_v0_t** params;

  // Optional occupancy information used to calculate ideal workgroup count and
  // size. Information is intended to be derived from analysis or extracted from
  // the final machine code and be accurate enough to trust for distribution.
  // If unavailable then workgroup sizes must be specified on each export.
  const iree_hal_executable_dispatch_occupancy_v0_t* occupancy;

  // Optional table of export function entry point names 1:1 with ptrs.
  // These names are only used for tracing/debugging and can be omitted to save
  // binary size.
  const char* const* names;

  // Optional table of entry point tags 1:1 with ptrs.
  // Used to describe the entry point in a human-readable format useful for
  // verbose logging. The string values, when present, may be attached to
  // tracing/debugging events related to the entry point.
  const char* const* tags;

  // Optional string table of parameter names.
  // Names are NUL-terminated. May be omitted if names are not available.
  const char* const* parameter_names;

  // Optional table of source locations 1:1 with ptrs.
  // These are the canonical source location in the compiler.
  const iree_hal_executable_source_location_v0_t* source_locations;

  // Optional table of source locations by compilation stage 1:1 with ptrs.
  // These may provide additional internal compilation results at various
  // stages of compilation.
  const iree_hal_executable_stage_location_table_v0_t* stage_locations;
} iree_hal_executable_export_table_v0_t;

// A table declaring the executable-level constants that can be used to
// specialize the executable behavior.
typedef struct iree_hal_executable_constant_table_v0_t {
  // Total number of constants in the table.
  uint32_t count;
  // We could add more metadata here if we wanted to enable reflection.
} iree_hal_executable_constant_table_v0_t;

// An embedded file defined by an arbitrary path.
typedef struct iree_hal_executable_source_file_v0_t {
  // The length of |path| in bytes.
  uint32_t path_length;
  // The path (absolute or relative) of the source file, NUL-terminated.
  const char* path;
  // The length of |content| in bytes.
  uint32_t content_length;
  // The file contents (possibly binary).
  const uint8_t* content;
} iree_hal_executable_source_file_v0_t;

// A table listing zero or more embedded source files.
typedef struct iree_hal_executable_source_file_table_v0_t {
  // Total number of source files.
  uint32_t count;
  // Table of |count| source files.
  const iree_hal_executable_source_file_v0_t* files;
} iree_hal_executable_source_file_table_v0_t;

// Structure used for v0 library interfaces.
// The entire structure is designed to be read-only and able to live embedded in
// the binary .rdata section.
//
// The information held within the structure is not cached by the runtime.
// Implementations may choose to heap allocate this structure and modify its
// members at runtime so long as they observe the thread-safety guarantees.
// For example, a JIT may default all exports to JIT thunk functions and then
// atomically swap them out for the translated function pointers as they are
// available.
typedef struct iree_hal_executable_library_v0_t {
  // Version/metadata header.
  // Will have a version of IREE_HAL_EXECUTABLE_LIBRARY_VERSION_*.
  const iree_hal_executable_library_header_t* header;

  // Table of imported functions available to functions in the executable.
  iree_hal_executable_import_table_v0_t imports;

  // Table of exported functions from the executable.
  iree_hal_executable_export_table_v0_t exports;

  // Table of executable-level constants.
  iree_hal_executable_constant_table_v0_t constants;

  // Table of optional sources used for debugging.
  // Exports may reference locations within the sources by path.
  iree_hal_executable_source_file_table_v0_t sources;
} iree_hal_executable_library_v0_t;

#endif  // IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_H_
