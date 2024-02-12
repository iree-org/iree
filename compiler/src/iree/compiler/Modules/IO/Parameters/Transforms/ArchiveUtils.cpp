// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/io/formats/irpa/irpa_builder.h"
#include "iree/tooling/parameter_util.h"

#include "iree/compiler/Modules/IO/Parameters/Transforms/ArchiveUtils.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler::IREE::IO::Parameters {

LogicalResult handleRuntimeError(Operation *op, iree_status_t status,
                                 StringRef failureMessage) {
  if (iree_status_is_ok(status))
    return success();
  std::string message;
  message.resize(512);
  iree_host_size_t buffer_length;
  if (!iree_status_format(status, message.size(), &message[0],
                          &buffer_length)) {
    message.resize(buffer_length + 1);
    iree_status_format(status, message.size(), &message[0], &buffer_length);
  }
  message.resize(buffer_length);
  iree_status_ignore(status);
  return op->emitError() << failureMessage << message;
}

LogicalResult
writeParameterIndex(Operation *op, iree_allocator_t allocator,
                    iree_io_parameter_archive_builder_t &builder,
                    std::unique_ptr<llvm::FileOutputBuffer> &fileBuffer,
                    iree_io_file_handle_t **output_file_handle,
                    iree_io_stream_t **output_stream,
                    iree_io_parameter_index_t **output_built_index) {

  // Wrap the output file for use with the parameter archive builder.
  iree_io_file_handle_t *target_file_handle = NULL;
  iree_byte_span_t file_contents = iree_make_byte_span(
      fileBuffer->getBufferStart(), fileBuffer->getBufferSize());
  // Release callback is a no-op, the mapping is managed by the unique_ptr.
  iree_status_t status = iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_WRITE, file_contents,
      iree_io_file_handle_release_callback_null(), allocator,
      &target_file_handle);
  if (failed(handleRuntimeError(op, status,
                                "Failed to open output parameter archive"))) {
    iree_io_file_handle_release(target_file_handle);
    return failure();
  }

  // Wrap the target file in a stream.
  iree_io_stream_t *target_stream = NULL;
  status = iree_io_stream_open(IREE_IO_STREAM_MODE_WRITABLE, target_file_handle,
                               /*file_offset=*/0, allocator, &target_stream);
  if (failed(handleRuntimeError(
          op, status, "Failed to create I/O stream to output file"))) {
    iree_io_file_handle_release(target_file_handle);
    iree_io_stream_release(target_stream);
    return failure();
  }

  // Allocate an index we'll populate during building to allow us to get the
  // storage ranges of non-metadata parameters.
  iree_io_parameter_index_t *built_index = NULL;
  status = iree_io_parameter_index_create(allocator, &built_index);
  if (failed(handleRuntimeError(op, status,
                                "Failed to allocate parameter index"))) {
    iree_io_file_handle_release(target_file_handle);
    iree_io_stream_release(target_stream);
    iree_io_parameter_index_release(built_index);
    return failure();
  }

  // Commit the archive header to the file and produce an index referencing
  // it. This will allow us to know where to copy file contents.
  status = iree_io_parameter_archive_builder_write(&builder, target_file_handle,
                                                   /*file_offset=*/0,
                                                   target_stream, built_index);
  if (failed(handleRuntimeError(
          op, status,
          "Failed to write parameter index header to output file"))) {
    iree_io_file_handle_release(target_file_handle);
    iree_io_stream_release(target_stream);
    iree_io_parameter_index_release(built_index);
    return failure();
  }

  *output_file_handle = target_file_handle;
  *output_stream = target_stream;
  *output_built_index = built_index;
  return success();
}

} // namespace mlir::iree_compiler::IREE::IO::Parameters
