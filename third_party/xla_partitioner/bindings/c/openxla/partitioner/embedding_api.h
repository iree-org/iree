// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Stable API for embedding the partitioner.

#ifndef OPENXLA_PARTITIONER_BINDINGS_C_EMBEDDING_API_H
#define OPENXLA_PARTITIONER_BINDINGS_C_EMBEDDING_API_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if (defined(_WIN32) || defined(__CYGWIN__))
// Visibility annotations disabled.
#define OPENXLA_PARTITIONER_EMBED_EXPORTED
#elif defined(_WIN32) || defined(__CYGWIN__)
// Windows visibility declarations.
#if OPENXLA_PARTITIONER_EMBED_BUILDING_LIBRARY
#define OPENXLA_PARTITIONER_EMBED_EXPORTED __declspec(dllexport)
#else
#define OPENXLA_PARTITIONER_EMBED_EXPORTED __declspec(dllimport)
#endif
#else
// Non-windows: use visibility attributes.
#define OPENXLA_PARTITIONER_EMBED_EXPORTED \
  __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque structures.
typedef struct openxla_partitioner_error_t openxla_partitioner_error_t;
typedef struct openxla_partitioner_session_t openxla_partitioner_session_t;
typedef struct openxla_partitioner_invocation_t
    openxla_partitioner_invocation_t;
typedef struct openxla_partitioner_source_t openxla_partitioner_source_t;
typedef struct openxla_partitioner_output_t openxla_partitioner_output_t;

//===----------------------------------------------------------------------===//
// Errors.
// Most compilation errors are routed through the diagnostic engine, but
// some errors are returned immediately. In this case, an API function will
// return an `openxla_partitioner_error_t*` which will be nullptr on success and
// an error on failure. When a non-null error is returned, it must be freed
// via `openxlaPartitionerErrorDestroy()`.
//===----------------------------------------------------------------------===//

// Destroys an error. Only non-nullptr errors must be destroyed, but it is
// legal to destroy nullptr.
OPENXLA_PARTITIONER_EMBED_EXPORTED void openxlaPartitionerErrorDestroy(
    openxla_partitioner_error_t *error);

// Gets the message associated with the error as a C-string. The string will be
// valid until the error is destroyed.
OPENXLA_PARTITIONER_EMBED_EXPORTED const char *
openxlaPartitionerErrorGetMessage(openxla_partitioner_error_t *error);

//===----------------------------------------------------------------------===//
// Global initialization.
//===----------------------------------------------------------------------===//

// Gets the version of the API that this partitioner exports.
// The version is encoded with the lower 16 bits containing the minor version
// and upper bits containing the major version.
// The partitioner API is versioned. Within a major version, symbols may be
// added, but existing symbols must not be removed or changed to alter
// previously exposed functionality. A major version bump implies an API
// break and no forward or backward compatibility is assumed across major
// versions.
OPENXLA_PARTITIONER_EMBED_EXPORTED int openxlaPartitionerGetAPIVersion();

// The partitioner must be globally initialized before further use.
// It is intended to be called as part of the hosting process's startup
// sequence. Any failures that it can encounter are fatal and will abort
// the process.
// It is legal to call this multiple times, and each call must be balanced
// by a call to |openxlaPartitionerGlobalShutdown|. The final shutdown call will
// permanently disable the partitioner for the process and subsequent calls
// to initialize will fail/abort. If this is not desirable, some higher level
// code must hold initialization open with its own call.
OPENXLA_PARTITIONER_EMBED_EXPORTED void openxlaPartitionerGlobalInitialize();

// Gets the build revision of the tool. In official releases, this
// will be a string with the build tag. In development builds, it may be an
// empty string. The returned is valid for as long as the partitioner is
// initialized.
OPENXLA_PARTITIONER_EMBED_EXPORTED const char *openxlaPartitionerGetRevision();

// Initializes the command line environment from an explicit argc/argv
// (typically the result of a prior call to openxlaPartitionerGetProcessCLArgs).
// This uses dark magic to setup the usual array of expected signal handlers.
// This API is not yet considered version-stable. If using out of tree, please
// contact the developers.
//
// Note that there is as yet no facility to register new global command line
// options from the C API. However, this facility should be sufficient for
// subordinating builtin command line options to a higher level integration
// by tunneling global options into the initialization sequence.
OPENXLA_PARTITIONER_EMBED_EXPORTED void openxlaPartitionerSetupGlobalCL(
    int argc, const char **argv, const char *banner,
    bool installSignalHandlers);

// Destroys any process level resources that the partitioner may have created.
// This must be called prior to library unloading.
OPENXLA_PARTITIONER_EMBED_EXPORTED void openxlaPartitionerGlobalShutdown();

//===----------------------------------------------------------------------===//
// Session management.
// A session represents a scope where one or more runs can be executed.
// Internally, it consists of an MLIRContext and a private set of session
// options. If the CL environment was initialized, session options will be
// bootstrapped from global flags.
//
// Session creation cannot fail in a non-fatal way.
//===----------------------------------------------------------------------===//

// Creates a new session (which must be destroyed by
// openxlaPartitionerSessionDestroy).
OPENXLA_PARTITIONER_EMBED_EXPORTED openxla_partitioner_session_t *
openxlaPartitionerSessionCreate();

// Destroys a session.
OPENXLA_PARTITIONER_EMBED_EXPORTED void openxlaPartitionerSessionDestroy(
    openxla_partitioner_session_t *session);

// Sets session-local flags. These are a subset of flags supported by CLI
// tools and are privately scoped.
OPENXLA_PARTITIONER_EMBED_EXPORTED openxla_partitioner_error_t *
openxlaPartitionerSessionSetFlags(openxla_partitioner_session_t *session,
                                  int argc, const char *const *argv);

// Gets textual flags actually in effect from any source. Optionally, only
// calls back for non-default valued flags.
OPENXLA_PARTITIONER_EMBED_EXPORTED void openxlaPartitionerSessionGetFlags(
    openxla_partitioner_session_t *session, bool nonDefaultOnly,
    void (*onFlag)(const char *flag, size_t length, void *), void *userData);

//===----------------------------------------------------------------------===//
// Run management.
// Runs execute against a session and represent a discrete invocation of the
// partitioner.
//===----------------------------------------------------------------------===//

enum openxla_partitioner_diagnostic_severity_t {
  OPENXLA_PARTITIONER_DIAGNOSTIC_SEVERITY_NOTE = 0,
  OPENXLA_PARTITIONER_DIAGNOSTIC_SEVERITY_WARNING = 1,
  OPENXLA_PARTITIONER_DIAGNOSTIC_SEVERITY_ERROR = 2,
  OPENXLA_PARTITIONER_DIAGNOSTIC_SEVERITY_REMARK = 3,
};

OPENXLA_PARTITIONER_EMBED_EXPORTED openxla_partitioner_invocation_t *
openxlaPartitionerInvocationCreate(openxla_partitioner_session_t *session);

// Enables a callback to receive diagnostics. This is targeted at API use of
// the partitioner, allowing fine grained collection of formatted diagnostic
// records. It is not completely identical to
// |openxlaPartitionerInvocationEnableConsoleDiagnostics| which produces output
// suitable for an interactive stream (including color detection, etc) and has
// additional features for reading source files, etc. With default flags, no
// system state outside of the session will be used (i.e. no debug information
// loaded from files, etc).
// The |flags| parameter is reserved for the future and must be 0.
// The |callback| may be invoked from any thread at any time prior to
// destruction of the invocation. The callback should not make any calls back
// into partitioner APIs.
// The |message| passes to the callback is only valid for the duration of
// the callback and the |messageSize| does not include a terminator nul.
OPENXLA_PARTITIONER_EMBED_EXPORTED void
openxlaPartitionerInvocationEnableCallbackDiagnostics(
    openxla_partitioner_invocation_t *inv, int flags,
    void (*callback)(enum openxla_partitioner_diagnostic_severity_t severity,
                     const char *message, size_t messageSize, void *userData),
    void *userData);

// Enables default, pretty-printed diagnostics to the console. This is usually
// the right thing to do for command-line tools, but other mechanisms are
// preferred for library use.
OPENXLA_PARTITIONER_EMBED_EXPORTED void
openxlaPartitionerInvocationEnableConsoleDiagnostics(
    openxla_partitioner_invocation_t *inv);

// Destroys a run.
OPENXLA_PARTITIONER_EMBED_EXPORTED void openxlaPartitionerInvocationDestroy(
    openxla_partitioner_invocation_t *inv);

// Sets a crash handler on the invocation. In the event of a crash, the callback
// will be invoked to create an output which will receive the crash dump.
// The callback should either set |*outOutput| to a new
// |openxla_partitioner_output_t| or return an error. Ownership of the output is
// passed to the caller. The implementation implicitly calls
// |openxlaPartitionerOutputKeep| on the output.
OPENXLA_PARTITIONER_EMBED_EXPORTED void
openxlaPartitionerInvocationSetCrashHandler(
    openxla_partitioner_invocation_t *inv, bool genLocalReproducer,
    openxla_partitioner_error_t *(*onCrashCallback)(
        openxla_partitioner_output_t **outOutput, void *userData),
    void *userData);

// Parses a source into this instance in preparation for performing a
// compilation action.
// Returns false and emits diagnostics on failure.
OPENXLA_PARTITIONER_EMBED_EXPORTED bool openxlaPartitionerInvocationParseSource(
    openxla_partitioner_invocation_t *inv,
    openxla_partitioner_source_t *source);

// Enables/disables verification of IR after each pass. Defaults to enabled.
OPENXLA_PARTITIONER_EMBED_EXPORTED void openxlaPartitionerInvocationSetVerifyIR(
    openxla_partitioner_invocation_t *inv, bool enable);

// Runs a compilation pipeline.
// Returns false and emits diagnostics on failure.
OPENXLA_PARTITIONER_EMBED_EXPORTED bool openxlaPartitionerInvocationPipeline(
    openxla_partitioner_invocation_t *inv, const char *pipeline_name);

// Outputs the current partitioner state as textual IR to the output.
OPENXLA_PARTITIONER_EMBED_EXPORTED openxla_partitioner_error_t *
openxlaPartitionerInvocationOutputIR(openxla_partitioner_invocation_t *inv,
                                     openxla_partitioner_output_t *output);

//===----------------------------------------------------------------------===//
// Sources.
// Compilation sources are loaded into openxla_partitioner_source_t instances.
// These instances reference an originating session and may contain a concrete
// buffer of memory. Generally, when processing a source, its backing buffer
// will be transferred out from under it (i.e. sources are single-use).
// The actual source instance must be kept live until all processing is
// completed as some methods of loading a source involve maintaining
// references to backing resources.
//===----------------------------------------------------------------------===//

// Destroy source instances.
OPENXLA_PARTITIONER_EMBED_EXPORTED void openxlaPartitionerSourceDestroy(
    openxla_partitioner_source_t *source);

// Opens the source from a file. This is used for normal text assembly file
// sources.
// Must be destroyed with openxlaPartitionerSourceDestroy().
OPENXLA_PARTITIONER_EMBED_EXPORTED openxla_partitioner_error_t *
openxlaPartitionerSourceOpenFile(openxla_partitioner_session_t *session,
                                 const char *filePath,
                                 openxla_partitioner_source_t **out_source);

// Wraps an existing buffer in memory.
// If |isNullTerminated| is true, then the null must be accounted for in the
// length. This is required for text buffers and it is permitted for binary
// buffers.
// Must be destroyed with openxlaPartitionerSourceDestroy().
OPENXLA_PARTITIONER_EMBED_EXPORTED openxla_partitioner_error_t *
openxlaPartitionerSourceWrapBuffer(openxla_partitioner_session_t *session,
                                   const char *bufferName, const char *buffer,
                                   size_t length, bool isNullTerminated,
                                   openxla_partitioner_source_t **out_source);

//===----------------------------------------------------------------------===//
// Outputs.
// Compilation outputs are standalone instances that are used to collect
// final compilation artifacts. In their most basic form, they are just
// wrappers around an output stream over some file. However, more advanced
// things can be enabled via additional APIs (i.e. allocating efficient
// temporary file handles, etc).
//
// Outputs are not bound to a session as they can outlive it or be disconnected
// from the actual process of compilation in arbitrary ways.
//===----------------------------------------------------------------------===//

// Destroy output instances.
OPENXLA_PARTITIONER_EMBED_EXPORTED void openxlaPartitionerOutputDestroy(
    openxla_partitioner_output_t *output);

// Opens a file for the output.
// Must be destroyed via openxlaPartitionerOutputDestroy().
OPENXLA_PARTITIONER_EMBED_EXPORTED openxla_partitioner_error_t *
openxlaPartitionerOutputOpenFile(const char *filePath,
                                 openxla_partitioner_output_t **out_output);

// Opens a file descriptor for output.
// Must be destroyed via openxlaPartitionerOutputDestroy().
OPENXLA_PARTITIONER_EMBED_EXPORTED openxla_partitioner_error_t *
openxlaPartitionerOutputOpenFD(int fd,
                               openxla_partitioner_output_t **out_output);

// Opens an output to in-memory storage. Use the API
// |openxlaPartitionerOutputMapMemory| to access the mapped contents once all
// output has been written.
OPENXLA_PARTITIONER_EMBED_EXPORTED openxla_partitioner_error_t *
openxlaPartitionerOutputOpenMembuffer(
    openxla_partitioner_output_t **out_output);

// Maps the contents of a partitioner output opened via
// |openxlaPartitionerOutputOpenMembuffer|. This may be something obtained via
// mmap or a more ordinary temporary buffer. This may fail in platform
// specific ways unless if the output was created via
// |openxlaPartitionerOutputOpenMembuffer|.
OPENXLA_PARTITIONER_EMBED_EXPORTED openxla_partitioner_error_t *
openxlaPartitionerOutputMapMemory(openxla_partitioner_output_t *output,
                                  void **contents, uint64_t *size);

// For file or other persistent outputs, by default they will be deleted on
// |openxlaPartitionerOutputDestroy| (or exit). It is necessary to call
// |openxlaPartitionerOutputKeep| in order to have them committed to their
// accessible place.
OPENXLA_PARTITIONER_EMBED_EXPORTED void openxlaPartitionerOutputKeep(
    openxla_partitioner_output_t *output);

// Writes arbitrary data to the output.
OPENXLA_PARTITIONER_EMBED_EXPORTED openxla_partitioner_error_t *
openxlaPartitionerOutputWrite(openxla_partitioner_output_t *output,
                              const void *data, size_t length);

#ifdef __cplusplus
}
#endif

#endif  // OPENXLA_PARTITIONER_BINDINGS_C_EMBEDDING_API_H
