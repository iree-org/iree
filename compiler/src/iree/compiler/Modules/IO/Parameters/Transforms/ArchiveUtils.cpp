// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/IO/Parameters/Transforms/ArchiveUtils.h"

#include "llvm/Support/FileOutputBuffer.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler::IREE::IO::Parameters {

LogicalResult handleRuntimeError(Operation *op, iree_status_t status,
                                 StringRef failureMessage) {
  if (iree_status_is_ok(status))
    return success();
  iree_host_size_t buffer_length = 0;
  if (!iree_status_format(status, /*buffer_capacity=*/0,
                          /*buffer=*/nullptr, &buffer_length))
    return op->emitError() << failureMessage;
  std::string message;
  message.reserve(buffer_length);
  message.resize(buffer_length - 1);
  iree_status_format(status, message.capacity(), &message[0], &buffer_length);
  iree_status_ignore(status);
  return op->emitError() << failureMessage << "\n" << message;
}

FailureOr<ArchiveBuilder> createArchiveBuilder(Operation *op) {
  iree_allocator_t hostAllocator = iree_allocator_system();
  iree_io_parameter_archive_builder_t *builderPtr = NULL;
  if (failed(handleRuntimeError(op,
                                iree_allocator_malloc(hostAllocator,
                                                      sizeof(*builderPtr),
                                                      (void **)&builderPtr),
                                "allocating archive builder"))) {
    return failure();
  }
  ArchiveBuilder builder(
      builderPtr, +[](iree_io_parameter_archive_builder_t *builder) {
        iree_allocator_t host_allocator = builder->host_allocator;
        iree_io_parameter_archive_builder_deinitialize(builder);
        iree_allocator_free(host_allocator, builder);
      });
  iree_io_parameter_archive_builder_initialize(hostAllocator, builder.get());
  return std::move(builder);
}

FailureOr<FileStreamIndex> createParameterIndex(Operation *op,
                                                ArchiveBuilder builder,
                                                StringRef archivePath) {
  // Open a file of sufficient size for writing.
  iree_io_physical_size_t archiveLength =
      iree_io_parameter_archive_builder_total_size(builder.get());
  auto fileOr = llvm::FileOutputBuffer::create(archivePath, archiveLength);
  if (!fileOr) {
    return op->emitError()
           << "failed to create file output buffer at " << archivePath
           << " with error: "
           << llvm::errorToErrorCode(fileOr.takeError()).message();
  }
  std::unique_ptr<llvm::FileOutputBuffer> fileBuffer = std::move(*fileOr);

  // Wrap the output file for use with the parameter archive builder.
  // Release callback is a no-op, the mapping is managed by the unique_ptr.
  iree_io_file_handle_t *fileHandlePtr = nullptr;
  if (failed(handleRuntimeError(
          op,
          iree_io_file_handle_wrap_host_allocation(
              IREE_IO_FILE_ACCESS_WRITE,
              iree_make_byte_span(fileBuffer->getBufferStart(),
                                  fileBuffer->getBufferSize()),
              iree_io_file_handle_release_callback_null(),
              builder->host_allocator, &fileHandlePtr),
          "failed to open output parameter archive"))) {
    return failure();
  }
  auto fileHandle = FileHandle(fileHandlePtr, iree_io_file_handle_release);

  // Wrap the target file in a stream. The stream will retain the file handle.
  iree_io_stream_t *streamPtr = nullptr;
  if (failed(handleRuntimeError(
          op,
          iree_io_stream_open(IREE_IO_STREAM_MODE_WRITABLE, fileHandle.get(),
                              /*file_offset=*/0, builder->host_allocator,
                              &streamPtr),
          "failed to create I/O stream to output file"))) {
    return failure();
  }
  auto stream = Stream(streamPtr, iree_io_stream_release);

  // Allocate an index we'll populate during building to allow us to get the
  // storage ranges of non-metadata parameters.
  iree_io_parameter_index_t *indexPtr = nullptr;
  if (failed(handleRuntimeError(
          op,
          iree_io_parameter_index_create(builder->host_allocator, &indexPtr),
          "failed to allocate parameter index"))) {
    return failure();
  }
  auto index = ParameterIndex(indexPtr, iree_io_parameter_index_release);

  // Commit the archive header to the file and produce an index referencing
  // it. This will allow us to know where to copy file contents.
  if (failed(handleRuntimeError(
          op,
          iree_io_parameter_archive_builder_write(
              builder.get(), fileHandle.get(),
              /*file_offset=*/0, stream.get(), index.get()),
          "failed to write parameter index header to output file"))) {
    return failure();
  }

  return std::make_tuple(std::move(fileBuffer), std::move(stream),
                         std::move(index));
}

} // namespace mlir::iree_compiler::IREE::IO::Parameters
