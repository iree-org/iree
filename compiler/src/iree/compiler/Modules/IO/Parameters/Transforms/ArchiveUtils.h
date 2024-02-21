// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_MODULES_IO_PARAMETERS_TRANSFORMS_ARCHIVEUTILS_H_
#define IREE_COMPILER_MODULES_IO_PARAMETERS_TRANSFORMS_ARCHIVEUTILS_H_

#include "iree/base/api.h"
#include "iree/io/formats/irpa/irpa_builder.h"
#include "iree/tooling/parameter_util.h"

#include "llvm/Support/FileOutputBuffer.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler::IREE::IO::Parameters {

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

// Helper to interpret iree status messages and print the error message.
LogicalResult handleRuntimeError(Operation *op, iree_status_t status,
                                 StringRef failureMessage);

// Helper to write the parameter index constructed in the archive |builder|
// to the given |fileBuffer|. Populates a file, stream, and index handle on
// success for further writing of the data segments. The file, stream, and
// index handled must be released by the caller if this succeeds.
LogicalResult
writeParameterIndex(Operation *op, iree_allocator_t allocator,
                    iree_io_parameter_archive_builder_t &builder,
                    std::unique_ptr<llvm::FileOutputBuffer> &fileBuffer,
                    iree_io_file_handle_t **output_file_handle,
                    iree_io_stream_t **output_stream,
                    iree_io_parameter_index_t **output_built_index);

} // namespace mlir::iree_compiler::IREE::IO::Parameters

#endif // IREE_COMPILER_MODULES_IO_PARAMETERS_TRANSFORMS_ARCHIVEUTILS_H_
