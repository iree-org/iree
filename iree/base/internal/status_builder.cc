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

#include "iree/base/internal/status_builder.h"

#include <cerrno>
#include <cstdio>

#include "iree/base/target_platform.h"

namespace iree {

StatusBuilder::Rep::Rep(const Rep& r)
    : stream_message(r.stream_message), stream(&stream_message) {}

StatusBuilder::StatusBuilder(Status&& original_status, SourceLocation location)
    : status_(exchange(original_status, original_status.code())) {}

StatusBuilder::StatusBuilder(Status&& original_status, SourceLocation location,
                             const char* format, ...)
    : status_(exchange(original_status, original_status.code())) {
  if (status_.ok()) return;
  va_list varargs_0, varargs_1;
  va_start(varargs_0, format);
  va_start(varargs_1, format);
  status_ =
      iree_status_annotate_vf(status_.release(), format, varargs_0, varargs_1);
  va_end(varargs_0);
  va_end(varargs_1);
}

StatusBuilder::StatusBuilder(StatusCode code, SourceLocation location)
    : status_(code, location, "") {}

StatusBuilder::StatusBuilder(StatusCode code, SourceLocation location,
                             const char* format, ...) {
  if (code == StatusCode::kOk) {
    status_ = StatusCode::kOk;
    return;
  }
  va_list varargs_0, varargs_1;
  va_start(varargs_0, format);
  va_start(varargs_1, format);
  status_ = iree_status_allocate_vf(static_cast<iree_status_code_t>(code),
                                    location.file_name(), location.line(),
                                    format, varargs_0, varargs_1);
  va_end(varargs_0);
  va_end(varargs_1);
}

void StatusBuilder::Flush() {
  if (!rep_ || rep_->stream_message.empty()) return;
  auto rep = std::move(rep_);
  status_ = iree_status_annotate_f(status_.release(), "%.*s",
                                   static_cast<int>(rep->stream_message.size()),
                                   rep->stream_message.data());
}

bool StatusBuilder::ok() const { return status_.ok(); }

StatusBuilder AbortedErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kAborted, location);
}

StatusBuilder AlreadyExistsErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kAlreadyExists, location);
}

StatusBuilder CancelledErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kCancelled, location);
}

StatusBuilder DataLossErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kDataLoss, location);
}

StatusBuilder DeadlineExceededErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kDeadlineExceeded, location);
}

StatusBuilder FailedPreconditionErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kFailedPrecondition, location);
}

StatusBuilder InternalErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kInternal, location);
}

StatusBuilder InvalidArgumentErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kInvalidArgument, location);
}

StatusBuilder NotFoundErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kNotFound, location);
}

StatusBuilder OutOfRangeErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kOutOfRange, location);
}

StatusBuilder PermissionDeniedErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kPermissionDenied, location);
}

StatusBuilder UnauthenticatedErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kUnauthenticated, location);
}

StatusBuilder ResourceExhaustedErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kResourceExhausted, location);
}

StatusBuilder UnavailableErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kUnavailable, location);
}

StatusBuilder UnimplementedErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kUnimplemented, location);
}

StatusBuilder UnknownErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kUnknown, location);
}

// Returns the code for |error_number|, which should be an |errno| value.
// See https://en.cppreference.com/w/cpp/error/errno_macros and similar refs.
static StatusCode ErrnoToCanonicalCode(int error_number) {
  switch (error_number) {
    case 0:
      return StatusCode::kOk;
    case EINVAL:        // Invalid argument
    case ENAMETOOLONG:  // Filename too long
    case E2BIG:         // Argument list too long
    case EDESTADDRREQ:  // Destination address required
    case EDOM:          // Mathematics argument out of domain of function
    case EFAULT:        // Bad address
    case EILSEQ:        // Illegal byte sequence
    case ENOPROTOOPT:   // Protocol not available
    case ENOSTR:        // Not a STREAM
    case ENOTSOCK:      // Not a socket
    case ENOTTY:        // Inappropriate I/O control operation
    case EPROTOTYPE:    // Protocol wrong type for socket
    case ESPIPE:        // Invalid seek
      return StatusCode::kInvalidArgument;
    case ETIMEDOUT:  // Connection timed out
    case ETIME:      // Timer expired
      return StatusCode::kDeadlineExceeded;
    case ENODEV:  // No such device
    case ENOENT:  // No such file or directory
#ifdef ENOMEDIUM
    case ENOMEDIUM:  // No medium found
#endif
    case ENXIO:  // No such device or address
    case ESRCH:  // No such process
      return StatusCode::kNotFound;
    case EEXIST:         // File exists
    case EADDRNOTAVAIL:  // Address not available
    case EALREADY:       // Connection already in progress
#ifdef ENOTUNIQ
    case ENOTUNIQ:  // Name not unique on network
#endif
      return StatusCode::kAlreadyExists;
    case EPERM:   // Operation not permitted
    case EACCES:  // Permission denied
#ifdef ENOKEY
    case ENOKEY:  // Required key not available
#endif
    case EROFS:  // Read only file system
      return StatusCode::kPermissionDenied;
    case ENOTEMPTY:   // Directory not empty
    case EISDIR:      // Is a directory
    case ENOTDIR:     // Not a directory
    case EADDRINUSE:  // Address already in use
    case EBADF:       // Invalid file descriptor
#ifdef EBADFD
    case EBADFD:  // File descriptor in bad state
#endif
    case EBUSY:    // Device or resource busy
    case ECHILD:   // No child processes
    case EISCONN:  // Socket is connected
#ifdef EISNAM
    case EISNAM:  // Is a named type file
#endif
#ifdef ENOTBLK
    case ENOTBLK:  // Block device required
#endif
    case ENOTCONN:  // The socket is not connected
    case EPIPE:     // Broken pipe
#ifdef ESHUTDOWN
    case ESHUTDOWN:  // Cannot send after transport endpoint shutdown
#endif
    case ETXTBSY:  // Text file busy
#ifdef EUNATCH
    case EUNATCH:  // Protocol driver not attached
#endif
      return StatusCode::kFailedPrecondition;
    case ENOSPC:  // No space left on device
#ifdef EDQUOT
    case EDQUOT:  // Disk quota exceeded
#endif
    case EMFILE:   // Too many open files
    case EMLINK:   // Too many links
    case ENFILE:   // Too many open files in system
    case ENOBUFS:  // No buffer space available
    case ENODATA:  // No message is available on the STREAM read queue
    case ENOMEM:   // Not enough space
    case ENOSR:    // No STREAM resources
#ifdef EUSERS
    case EUSERS:  // Too many users
#endif
      return StatusCode::kResourceExhausted;
#ifdef ECHRNG
    case ECHRNG:  // Channel number out of range
#endif
    case EFBIG:      // File too large
    case EOVERFLOW:  // Value too large to be stored in data type
    case ERANGE:     // Result too large
      return StatusCode::kOutOfRange;
#ifdef ENOPKG
    case ENOPKG:  // Package not installed
#endif
    case ENOSYS:        // Function not implemented
    case ENOTSUP:       // Operation not supported
    case EAFNOSUPPORT:  // Address family not supported
#ifdef EPFNOSUPPORT
    case EPFNOSUPPORT:  // Protocol family not supported
#endif
    case EPROTONOSUPPORT:  // Protocol not supported
#ifdef ESOCKTNOSUPPORT
    case ESOCKTNOSUPPORT:  // Socket type not supported
#endif
    case EXDEV:  // Improper link
      return StatusCode::kUnimplemented;
    case EAGAIN:  // Resource temporarily unavailable
#ifdef ECOMM
    case ECOMM:  // Communication error on send
#endif
    case ECONNREFUSED:  // Connection refused
    case ECONNABORTED:  // Connection aborted
    case ECONNRESET:    // Connection reset
    case EINTR:         // Interrupted function call
#ifdef EHOSTDOWN
    case EHOSTDOWN:  // Host is down
#endif
    case EHOSTUNREACH:  // Host is unreachable
    case ENETDOWN:      // Network is down
    case ENETRESET:     // Connection aborted by network
    case ENETUNREACH:   // Network unreachable
    case ENOLCK:        // No locks available
    case ENOLINK:       // Link has been severed
#ifdef ENONET
    case ENONET:  // Machine is not on the network
#endif
      return StatusCode::kUnavailable;
    case EDEADLK:  // Resource deadlock avoided
#ifdef ESTALE
    case ESTALE:  // Stale file handle
#endif
      return StatusCode::kAborted;
    case ECANCELED:  // Operation cancelled
      return StatusCode::kCancelled;
    default:
      return StatusCode::kUnknown;
  }
}

StatusBuilder ErrnoToCanonicalStatusBuilder(int error_number,
                                            SourceLocation location) {
  return StatusBuilder(ErrnoToCanonicalCode(error_number), location);
}

#if defined(IREE_PLATFORM_WINDOWS)

// Returns the code for |error| which should be a Win32 error dword.
static StatusCode Win32ErrorToCanonicalCode(uint32_t error) {
  switch (error) {
    case ERROR_SUCCESS:
      return StatusCode::kOk;
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
      return StatusCode::kNotFound;
    case ERROR_TOO_MANY_OPEN_FILES:
    case ERROR_OUTOFMEMORY:
    case ERROR_HANDLE_DISK_FULL:
    case ERROR_HANDLE_EOF:
      return StatusCode::kResourceExhausted;
    case ERROR_ACCESS_DENIED:
      return StatusCode::kPermissionDenied;
    case ERROR_INVALID_HANDLE:
      return StatusCode::kInvalidArgument;
    case ERROR_NOT_READY:
    case ERROR_READ_FAULT:
      return StatusCode::kUnavailable;
    case ERROR_WRITE_FAULT:
      return StatusCode::kDataLoss;
    case ERROR_NOT_SUPPORTED:
      return StatusCode::kUnimplemented;
    default:
      return StatusCode::kUnknown;
  }
}

StatusBuilder Win32ErrorToCanonicalStatusBuilder(uint32_t error,
                                                 SourceLocation location) {
  // TODO(benvanik): use FormatMessage; or defer until required?
  return StatusBuilder(Win32ErrorToCanonicalCode(error), location)
         << "<TBD>: " << error;
}

#endif  // IREE_PLATFORM_WINDOWS

}  // namespace iree
