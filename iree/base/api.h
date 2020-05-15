// Copyright 2019 Google LLC
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

// Dynamic Linkage
// -----------------------------------------------------------------------------
//
// Define IREE_API_NO_PROTOTYPES to disable function prototypes when linking and
// loading dynamically. This prevents accidental calls to functions without
// going through the resolved symbols.
//
// API Versioning
// -----------------------------------------------------------------------------
//
// The C API is designed to be versioned such that breaking changes either in
// ABI (data types, struct sizes, etc) or signatures (function arguments change)
// will result in a bump of the IREE_API_VERSION_LATEST value.
//
// When linked in statically the runtime should never have a version conflict,
// however dynamic linking where the runtime is a shared object loaded at
// runtime (via dlopen/etc) must always verify the version is as expected.
//
// In the current experimental state of the runtime the API may break frequently
// and the version is pinned at 0.
//
// Example:
//   void* library = dlopen("iree_rt.so", RTLD_LAZY | RTLD_LOCAL);
//   iree_api_version_t actual_version;
//   iree_status_t status = \
//       ((PFN_iree_api_version_check)dlsym(library, "iree_api_version_check"))(
//       IREE_API_VERSION_LATEST, &actual_version);
//   if (!iree_status_is_ok(status)) {
//     LOG(FATAL) << "Unsupported runtime API version " << actual_version;
//   }
//   dlclose(library);
//
// Object Ownership and Lifetime
// -----------------------------------------------------------------------------
//
// The API follows the CoreFoundation ownership policies:
// https://developer.apple.com/library/archive/documentation/CoreFoundation/Conceptual/CFMemoryMgmt/Concepts/Ownership.html
//
// These boil down to:
// * Objects returned from *_create or *_copy functions are owned by the caller
//   and must be released when the caller no longer needs them.
// * Objects returned from accessors are not owned by the caller and must be
//   retained by the caller if the object lifetime needs to be extended.
// * Objects passed to functions by argument may be retained by the callee if
//   required.
//
// Example:
//   iree_file_mapping_t* file_mapping;
//   s = iree_file_mapping_open_read(..., &file_mapping);
//   // file_mapping is now owned by this function.
//   s = iree_file_mapping_some_call(file_mapping, ...);
//   // Must release ownership when no longer required.
//   s = iree_file_mapping_release(file_mapping);
//
// String Formatting
// -----------------------------------------------------------------------------
//
// Functions that produce variable-length strings follow a standard usage
// pattern with the arguments:
//   `iree_host_size_t buffer_capacity`: total bytes including \0 available.
//   `char* buffer`: optional buffer to write into.
//   `iree_host_size_t* out_buffer_length`: required/actual length excluding \0.
//
// To query the size required for the output and allocate storage:
//   iree_host_size_t required_length = 0;
//   iree_format_xyz(/*buffer_capacity=*/0, /*buffer=*/NULL, &required_length);
//   iree_host_size_t buffer_capacity = required_length + 1;
//   char* buffer = iree_allocator_malloc(buffer_capacity);
//   iree_host_size_t actual_length = 0;
//   iree_format_xyz(buffer_capacity, buffer, &actual_length);
//   ASSERT(required_length == actual_length);
//
// To handle fixed-length maximum strings (common):
//   // Fails if the string is longer than 127 characters (127 + \0 >= 128).
//   char buffer[128];
//   IREE_RETURN_IF_ERROR(iree_format_xyz(sizeof(buffer), buffer, NULL));
//
// Try fixed-length and fallback to a dynamic allocation:
//   char inline_buffer[128];
//   iree_host_size_t required_length = 0;
//   iree_status_t inline_status = iree_format_xyz(sizeof(inline_buffer),
//                                                 inline_buffer,
//                                                 &required_length);
//   if (iree_status_is_out_of_range(inline_status)) {
//     // Spilled inline_buffer, need to allocate required_length bytes and
//     // try again.
//     // ... see above for example ...
//   } else if (iree_status_is_ok(inline_status)) {
//     // Fit inside inline_buffer, required_length contains actual length.
//   } else {
//     return inline_status;
//   }

#ifndef IREE_BASE_API_H_
#define IREE_BASE_API_H_

#include <memory.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
// The safe malloca that may fall back to heap in the case of stack overflows:
// https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/malloca?view=vs-2019
// Because that gets really annoying to deal with during error handling we just
// go for _alloca which may generate SEH exceptions if we blow the stack.
#include <malloc.h>
#define iree_alloca(sz) _alloca(sz)
#else
#include <alloca.h>
#define iree_alloca(sz) alloca(sz)
#endif  // _WIN32

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
#define IREE_API_EXPORT extern "C"
#else
#define IREE_API_EXPORT
#endif  // __cplusplus

#if defined(_WIN32)
#define IREE_API_CALL __stdcall
#define IREE_API_PTR IREE_API_CALL
#else
#define IREE_API_CALL
#define IREE_API_PTR
#endif  // _WIN32

#if (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
     __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define IREE_IS_LITTLE_ENDIAN 1
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define IREE_IS_BIG_ENDIAN 1
#elif defined(_WIN32)
#define IREE_IS_LITTLE_ENDIAN 1
#else
#error "IREE endian detection needs to be set up for your compiler"
#endif  // __BYTE_ORDER__

// Size, in bytes, of a buffer on the host.
typedef size_t iree_host_size_t;

// Size, in bytes, of a buffer on devices.
typedef uint64_t iree_device_size_t;
// Whole length of the underlying buffer.
#define IREE_WHOLE_BUFFER (iree_device_size_t(-1))

// Like absl::Time, represented as nanoseconds since unix epoch.
// TODO(benvanik): pick something easy to get into/outof time_t/etc.
typedef int64_t iree_time_t;
// Like absl::InfinitePast.
#define IREE_TIME_INFINITE_PAST INT64_MIN
// Like absl::InfiniteFuture.
#define IREE_TIME_INFINITE_FUTURE INT64_MAX

// Like absl::Duration, represented as relative nanoseconds.
typedef int64_t iree_duration_t;
// Like absl::InfiniteDuration.
#define IREE_DURATION_INFINITE INT64_MIN
// Like absl::ZeroDuration.
#define IREE_DURATION_ZERO 0

// A span of mutable bytes (ala std::span of uint8_t).
typedef struct {
  uint8_t* data;
  iree_host_size_t data_length;
} iree_byte_span_t;

#define iree_make_byte_span(data, data_length) \
  { (uint8_t*)(data), (data_length) }

// A span of constant bytes (ala std::span of const uint8_t).
typedef struct {
  const uint8_t* data;
  iree_host_size_t data_length;
} iree_const_byte_span_t;

#define iree_make_const_byte_span(data, data_length) \
  { (const uint8_t*)(data), (data_length) }

//===----------------------------------------------------------------------===//
// iree_status_t and error reporting
//===----------------------------------------------------------------------===//

// Well-known status codes matching iree::StatusCode.
// Note that any code within IREE_STATUS_CODE_MASK is valid even if not
// enumerated here. Always check for unhandled errors/have default conditions.
typedef enum {
  IREE_STATUS_OK = 0,
  IREE_STATUS_CANCELLED = 1,
  IREE_STATUS_UNKNOWN = 2,
  IREE_STATUS_INVALID_ARGUMENT = 3,
  IREE_STATUS_DEADLINE_EXCEEDED = 4,
  IREE_STATUS_NOT_FOUND = 5,
  IREE_STATUS_ALREADY_EXISTS = 6,
  IREE_STATUS_PERMISSION_DENIED = 7,
  IREE_STATUS_RESOURCE_EXHAUSTED = 8,
  IREE_STATUS_FAILED_PRECONDITION = 9,
  IREE_STATUS_ABORTED = 10,
  IREE_STATUS_OUT_OF_RANGE = 11,
  IREE_STATUS_UNIMPLEMENTED = 12,
  IREE_STATUS_INTERNAL = 13,
  IREE_STATUS_UNAVAILABLE = 14,
  IREE_STATUS_DATA_LOSS = 15,
  IREE_STATUS_UNAUTHENTICATED = 16,

  IREE_STATUS_CODE_MASK = 0x1F,
} iree_status_code_t;

// TODO(#265): add ABSL_MUST_USE_RESULT to iree_status_t.

// Opaque status structure containing an iree_status_code_t and optional status
// object with more detailed information and payloads.
//
// The status value uses the lower 5 bits to store the iree_status_code_t and
// the remaining uintptr_t bits to store an optional status payload pointer.
// An OK status will always be bit-equivalent to 0 to make success/failure
// checks as cheap as an integer non-zero comparison. As the payload is optional
// it's legal to construct an iree_status_t from an iree_status_code_t directly
// meaning `return IREE_STATUS_INTERNAL;` (etc) is valid, though not as useful
// as constructing via iree_make_status_* (which captures additional info).
typedef uintptr_t iree_status_t;

// Makes an iree_status_t with the given iree_status_code_t code.
#define iree_make_status(code) \
  (iree_status_t)(((iree_status_code_t)(code)) & IREE_STATUS_CODE_MASK)

// TODO(#265): string formatting constructors.
// TODO(#265): payload support (custom stack traces, etc).

// Returns the iree_status_code_t from an iree_status_t.
#define iree_status_code(value) \
  ((iree_status_code_t)(value & IREE_STATUS_CODE_MASK))

// Macros to check the value of a status code.
#define iree_status_is_ok(value) (value == IREE_STATUS_OK)
#define iree_status_is_cancelled(value) \
  (iree_status_code(value) == IREE_STATUS_CANCELLED)
#define iree_status_is_unknown(value) \
  (iree_status_code(value) == IREE_STATUS_UNKNOWN)
#define iree_status_is_invalid_argument(value) \
  (iree_status_code(value) == IREE_STATUS_INVALID_ARGUMENT)
#define iree_status_is_deadline_exceeded(value) \
  (iree_status_code(value) == IREE_STATUS_DEADLINE_EXCEEDED)
#define iree_status_is_not_found(value) \
  (iree_status_code(value) == IREE_STATUS_NOT_FOUND)
#define iree_status_is_already_exists(value) \
  (iree_status_code(value) == IREE_STATUS_ALREADY_EXISTS)
#define iree_status_is_permission_denied(value) \
  (iree_status_code(value) == IREE_STATUS_PERMISSION_DENIED)
#define iree_status_is_resource_exhausted(value) \
  (iree_status_code(value) == IREE_STATUS_RESOURCE_EXHAUSTED)
#define iree_status_is_failed_precondition(value) \
  (iree_status_code(value) == IREE_STATUS_FAILED_PRECONDITION)
#define iree_status_is_aborted(value) \
  (iree_status_code(value) == IREE_STATUS_ABORTED)
#define iree_status_is_out_of_range(value) \
  (iree_status_code(value) == IREE_STATUS_OUT_OF_RANGE)
#define iree_status_is_unimplemented(value) \
  (iree_status_code(value) == IREE_STATUS_UNIMPLEMENTED)
#define iree_status_is_internal(value) \
  (iree_status_code(value) == IREE_STATUS_INTERNAL)
#define iree_status_is_unavailable(value) \
  (iree_status_code(value) == IREE_STATUS_UNAVAILABLE)
#define iree_status_is_data_loss(value) \
  (iree_status_code(value) == IREE_STATUS_DATA_LOSS)
#define iree_status_is_unauthenticated(value) \
  (iree_status_code(value) == IREE_STATUS_UNAUTHENTICATED)

// TODO(#265): better logging of status checks.
#define IREE_CHECK_OK(expr) CHECK_EQ(IREE_STATUS_OK, iree_status_code(expr))
#define IREE_ASSERT_OK(expr) ASSERT_EQ(IREE_STATUS_OK, iree_status_code(expr))
#define IREE_EXPECT_OK(expr) EXPECT_EQ(IREE_STATUS_OK, iree_status_code(expr))

#define IREE_API_STATUS_MACROS_IMPL_CONCAT_INNER_(x, y) x##y
#define IREE_API_STATUS_MACROS_IMPL_CONCAT_(x, y) \
  IREE_API_STATUS_MACROS_IMPL_CONCAT_INNER_(x, y)
#define IREE_API_STATUS_MACROS_IMPL_RETURN_IF_API_ERROR_(var, expr) \
  iree_status_t var = (expr);                                       \
  if (var) return var;

// TODO(#265): variadic arguments to support appending strings.
// Propagates the error returned by (expr) by returning from the current
// function on non-OK status.
//
// Example:
//  iree_status_t OtherFunc(...);
//  iree_status_t MyFunc(...) {
//    IREE_RETURN_IF_ERROR(OtherFunc(...));
//    return IREE_STATUS_OK;
//  }
#define IREE_RETURN_IF_ERROR(expr)                  \
  IREE_API_STATUS_MACROS_IMPL_RETURN_IF_API_ERROR_( \
      IREE_API_STATUS_MACROS_IMPL_CONCAT_(__status_, __COUNTER__), (expr))

// TODO(#265): clean up the status object.
// Ignores the status result of (expr).
#define IREE_IGNORE_ERROR(expr) (void)(expr)

#ifndef IREE_API_NO_PROTOTYPES

// Returns a NUL-terminated string constant for the given status code, such as
// IREE_STATUS_UNAVAILABLE = "UNAVAILABLE". Do not rely on string-matching the
// result as the exact text may change.
IREE_API_EXPORT const char* IREE_API_CALL
iree_status_code_string(iree_status_code_t code);

#endif  // IREE_API_NO_PROTOTYPES

//===----------------------------------------------------------------------===//
// IREE Core API
//===----------------------------------------------------------------------===//

// Known versions of the API that can be referenced in code.
// Out-of-bounds values are possible in forward-versioned changes.
typedef enum {
  IREE_API_VERSION_0 = 0,
  // Always set to the latest version of the library from source.
  IREE_API_VERSION_LATEST = IREE_API_VERSION_0,
} iree_api_version_t;

#ifndef IREE_API_NO_PROTOTYPES

// Checks whether the |expected_version| of the caller matches the implemented
// version of |out_actual_version|. Forward compatibility of the API is
// supported but backward compatibility is not: newer binaries using older
// shared libraries of the runtime will fail.
//
// Returns IREE_STATUS_OUT_OF_RANGE if the actual version is not compatible with
// the expected version.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_api_version_check(iree_api_version_t expected_version,
                       iree_api_version_t* out_actual_version);

// Initializes IREE for use within a binary.
//
// Specifically, this parses any command line flags and performs module
// initialization (such as for tracing and dynamic driver registration). If
// your application is certain it does not need this functionality, this call
// may be skipped.
//
// |argc| and |argv| should contain any command line flags to parse.
// If there are no flags to parse, nullptr may be passed, but this should still
// be called so other initialization happens.
//
// This should typically be called early in some sort of main() function once,
// before calling most other API functions. Certain core API functions here
// such as iree_api_version_check, iree_allocator_malloc, and
// iree_allocator_free are safe to call before this.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_api_init(int* argc,
                                                          char*** argv);

#endif  // IREE_API_NO_PROTOTYPES

//===----------------------------------------------------------------------===//
// iree_allocator_t (std::allocator-like interface)
//===----------------------------------------------------------------------===//

// Defines how an allocation from an iree_allocator_t should be made.
typedef enum {
  // The contents of the allocation *must* be zeroed by the allocator prior to
  // returning. Allocators may be able to elide the zeroing if they allocate
  // fresh pages from the system. It is always safe to zero contents if the
  // behavior of the allocator is not under our control.
  IREE_ALLOCATION_MODE_ZERO_CONTENTS = 1 << 0,
  // Tries to reuse an existing allocation provided via |out_ptr| if possible.
  // If the existing allocation is not reused then it is freed as if a call to
  // iree_allocator_free had been called on it. If the allocation fails then
  // the provided existing allocation is unmodified.
  //
  // This models the C realloc behavior.
  IREE_ALLOCATION_MODE_TRY_REUSE_EXISTING = 1 << 1,
} iree_allocation_mode_t;

// An allocator for host-memory allocations.
// IREE will attempt to use this in place of the system malloc and free.
// Pass the IREE_ALLOCATOR_SYSTEM macro to use the system allocator.
typedef struct {
  // User-defined pointer passed to all functions.
  void* self;
  // Allocates |byte_length| of memory and stores the pointer in |out_ptr|.
  // Systems should align to 16 byte boundaries (or otherwise their natural
  // SIMD alignment). The runtime pools internally and small allocations
  // (usually) won't be made through this interface.
  iree_status_t(IREE_API_PTR* alloc)(void* self, iree_allocation_mode_t mode,
                                     iree_host_size_t byte_length,
                                     void** out_ptr);
  // Frees |ptr| from a previous alloc call.
  void(IREE_API_PTR* free)(void* self, void* ptr);
} iree_allocator_t;

// Allocates using the iree_allocator_malloc and iree_allocator_free methods.
// These will usually be backed by malloc and free.
#define IREE_ALLOCATOR_SYSTEM                                     \
  iree_allocator_t {                                              \
    0, iree_allocator_system_allocate, iree_allocator_system_free \
  }

// Does not perform any allocation or deallocation; used to wrap objects that
// are owned by external code/live in read-only memory/etc.
#define IREE_ALLOCATOR_NULL \
  iree_allocator_t { 0, 0, 0 }

#ifndef IREE_API_NO_PROTOTYPES

// Allocates a block of |byte_length| bytes from the given allocator.
// The contents of the returned memory is guaranteed to be zeroed.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_allocator_malloc(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr);

// Reallocates |out_ptr| to |byte_length| bytes with the given allocator.
// If the reallocation fails then the original |out_ptr| is unmodified.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_allocator_realloc(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr);

// Frees a previously-allocated block of memory to the given allocator.
IREE_API_EXPORT void IREE_API_CALL
iree_allocator_free(iree_allocator_t allocator, void* ptr);

// Allocates a block of |byte_length| bytes from the default system allocator.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_allocator_system_allocate(void* self, iree_allocation_mode_t mode,
                               iree_host_size_t byte_length, void** out_ptr);

// Frees a previously-allocated block of memory to the default system allocator.
IREE_API_EXPORT void IREE_API_CALL iree_allocator_system_free(void* self,
                                                              void* ptr);

#endif  // IREE_API_NO_PROTOTYPES

//===----------------------------------------------------------------------===//
// iree_string_view_t (like std::string_view/absl::string_view)
//===----------------------------------------------------------------------===//

// A string view (ala std::string_view) into a non-NUL-terminated string.
typedef struct {
  const char* data;
  iree_host_size_t size;
} iree_string_view_t;

// Empty string view initializer.
#define IREE_STRING_VIEW_EMPTY \
  { 0, 0 }

// Returns true if the given string view is the empty string.
#define iree_string_view_is_empty(sv) (((sv).data == NULL) || (sv.size == 0))

#ifndef IREE_API_NO_PROTOTYPES

// Returns a string view initialized with a reference to the given
// NUL-terminated string literal.
IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_make_cstring_view(const char* str);

// Like std::string::compare but with iree_string_view_t values.
IREE_API_EXPORT int IREE_API_CALL
iree_string_view_compare(iree_string_view_t lhs, iree_string_view_t rhs);

// Returns true if the string starts with the given prefix.
IREE_API_EXPORT bool IREE_API_CALL iree_string_view_starts_with(
    iree_string_view_t value, iree_string_view_t prefix);

// Splits |value| into two parts based on the first occurrence of |split_char|.
// Returns the index of the |split_char| in the original |value| or -1 if not
// found.
IREE_API_EXPORT int IREE_API_CALL iree_string_view_split(
    iree_string_view_t value, char split_char, iree_string_view_t* out_lhs,
    iree_string_view_t* out_rhs);

// Returns true if the given |value| matches |pattern| (normal * and ? rules).
// This accepts wildcards in the form of '*' and '?' for any delimited value.
// '*' will match zero or more of any character and '?' will match exactly one
// of any character.
//
// For example,
// 'foo-*-bar' matches: 'foo-123-bar', 'foo-456-789-bar'
// 'foo-10?' matches: 'foo-101', 'foo-102'
IREE_API_EXPORT bool IREE_API_CALL iree_string_view_match_pattern(
    iree_string_view_t value, iree_string_view_t pattern);

#endif  // IREE_API_NO_PROTOTYPES

//===----------------------------------------------------------------------===//
// iree::FileMapping
//===----------------------------------------------------------------------===//

typedef struct iree_file_mapping iree_file_mapping_t;

#ifndef IREE_API_NO_PROTOTYPES

// Opens a file at |path| for read-only access via a file mapping.
// |out_file_mapping| must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_file_mapping_open_read(iree_string_view_t path, iree_allocator_t allocator,
                            iree_file_mapping_t** out_file_mapping);

// Retains the given |file_mapping| for the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_file_mapping_retain(iree_file_mapping_t* file_mapping);

// Releases the given |file_mapping| from the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_file_mapping_release(iree_file_mapping_t* file_mapping);

// Returns a reference to the byte buffer the |file_mapping| backs.
// Though the returned buffer is non-const behavior is undefined if read-only
// mappings are written to (exceptions, segfaults, etc).
IREE_API_EXPORT iree_byte_span_t IREE_API_CALL
iree_file_mapping_data(iree_file_mapping_t* file_mapping);

#endif  // IREE_API_NO_PROTOTYPES

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_API_H_
