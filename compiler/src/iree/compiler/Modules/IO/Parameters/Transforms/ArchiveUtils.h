// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_MODULES_IO_PARAMETERS_TRANSFORMS_ARCHIVEUTILS_H_
#define IREE_COMPILER_MODULES_IO_PARAMETERS_TRANSFORMS_ARCHIVEUTILS_H_

#include "llvm/Support/FileOutputBuffer.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"

#include "iree/base/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/formats/irpa/irpa_builder.h"
#include "iree/io/parameter_index.h"
#include "iree/io/stream.h"

namespace mlir::iree_compiler::IREE::IO::Parameters {

using ArchiveBuilder =
    std::unique_ptr<iree_io_parameter_archive_builder_t,
                    void (*)(iree_io_parameter_archive_builder_t *)>;
using FileHandle =
    std::unique_ptr<iree_io_file_handle_t, void (*)(iree_io_file_handle_t *)>;
using ParameterIndex = std::unique_ptr<iree_io_parameter_index_t,
                                       void (*)(iree_io_parameter_index_t *)>;
using Stream = std::unique_ptr<iree_io_stream_t, void (*)(iree_io_stream_t *)>;

// Wrapper around iree_io_stream for use when serializing constants.
class iree_io_stream_ostream : public llvm::raw_ostream {
public:
  explicit iree_io_stream_ostream(iree_io_stream_t *stream) : stream(stream) {
    iree_io_stream_retain(stream);
  }
  ~iree_io_stream_ostream() override { iree_io_stream_release(stream); }

private:
  uint64_t current_pos() const override {
    return iree_io_stream_offset(stream);
  }
  void write_impl(const char *ptr, size_t size) override {
    IREE_CHECK_OK(iree_io_stream_write(stream, size, ptr));
  }
  iree_io_stream_t *stream = NULL;
};

using ScopePath = std::pair<StringRef, StringRef>;

// Splits a `scope=path` string into two strings.
// If no `scope=` was specified the resulting scope string will be empty.
static inline ScopePath splitScopePath(StringRef scopePath) {
  size_t i = scopePath.find_first_of('=');
  if (i == StringRef::npos)
    return ScopePath("", scopePath);
  else
    return ScopePath(scopePath.substr(0, i), scopePath.substr(i + 1));
}

// Helper to interpret iree status messages and print the error message.
LogicalResult handleRuntimeError(Operation *op, iree_status_t status,
                                 StringRef failureMessage);

// Creates an empty archive builder.
FailureOr<ArchiveBuilder> createArchiveBuilder(Operation *op);

using FileStreamIndex =
    std::tuple<std::unique_ptr<llvm::FileOutputBuffer>, Stream, ParameterIndex>;

// Creates a parameter archive from |builder| and returns a file buffer and
// stream opened for writing the parameters at the offsets specified in the
// returned index.
FailureOr<FileStreamIndex> createParameterIndex(Operation *op,
                                                ArchiveBuilder builder,
                                                StringRef archivePath);

} // namespace mlir::iree_compiler::IREE::IO::Parameters

#endif // IREE_COMPILER_MODULES_IO_PARAMETERS_TRANSFORMS_ARCHIVEUTILS_H_
