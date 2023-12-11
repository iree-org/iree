// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_ARCHIVE_WRITER_H_
#define IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_ARCHIVE_WRITER_H_

#include <string>

#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Location.h"

namespace mlir::iree_compiler::IREE::VM {

// Interface for stateful bytecode module archive serialization.
//
// Intended usage:
//  - all embedded files are declared
//  - FlatBuffer is generated using the relative offsets of declared files
//  - FlatBuffer is written
//  - embedded files are flushed
class ArchiveWriter {
public:
  struct File {
    // Name of the file when exposed to users; informational only.
    std::string fileName;
    // Offset of the metadata/file from the end of the archive header file.
    uint64_t relativeOffset = 0;
    // Size of any optional metadata before the file begins, including padding.
    uint64_t prefixLength = 0;
    // Total size in bytes of the file on disk.
    uint64_t fileLength = 0;
    // Serializes the file contents to the output stream.
    std::function<LogicalResult(llvm::raw_ostream &os)> write;
  };

  virtual ~ArchiveWriter() = default;

  virtual bool supportsFiles() = 0;

  // Declares an embedded file in the archive and reserves a location for it.
  // The relative offset returned will be stable despite the variable-length
  // FlatBuffer header as it is relative to the header and not the archive 0.
  virtual File
  declareFile(std::string fileName, uint64_t fileAlignment, uint64_t fileLength,
              std::function<LogicalResult(llvm::raw_ostream &os)> write) = 0;

  // Writes an in-memory FlatBuffer to the archive as the header and flushes
  // all archive contents.
  virtual LogicalResult flush(FlatbufferBuilder &fbb) = 0;
};

// Textual JSON file archive containing only the FlatBuffer in textual form.
// Declared files are ignored and only the FlatBuffer is emitted.
//
// Archive structure:
//   {json text}
class JSONArchiveWriter : public ArchiveWriter {
public:
  explicit JSONArchiveWriter(Location loc, llvm::raw_ostream &os);
  ~JSONArchiveWriter() override;
  bool supportsFiles() override { return false; }
  File declareFile(
      std::string fileName, uint64_t fileAlignment, uint64_t fileLength,
      std::function<LogicalResult(llvm::raw_ostream &os)> write) override;
  LogicalResult flush(FlatbufferBuilder &fbb) override;

private:
  Location loc;
  llvm::raw_ostream &os;
};

// Flat file archive containing the FlatBuffer and trailing embedded files.
// No additional metadata beyond that in the FlatBuffer is emitted.
//
// Archive structure:
//   [4b flatbuffers_uoffset_t defining module FlatBuffer length]
//   [module FlatBuffer contents]
//   [zero padding to 64b alignment]
//   <<rodata base offset>>
//   [declared file 0]
//   [zero padding to 64b alignment]
//   [declared file 1]
//   ...
class FlatArchiveWriter : public ArchiveWriter {
public:
  explicit FlatArchiveWriter(Location loc, llvm::raw_ostream &os);
  ~FlatArchiveWriter() override;
  bool supportsFiles() override { return true; }
  File declareFile(
      std::string fileName, uint64_t fileAlignment, uint64_t fileLength,
      std::function<LogicalResult(llvm::raw_ostream &os)> write) override;
  LogicalResult flush(FlatbufferBuilder &fbb) override;

private:
  Location loc;
  llvm::raw_ostream &os;
  uint64_t tailFileOffset = 0; // unpadded
  SmallVector<File> files;
};

// Archive file containing .zip-compatible metadata.
// Allows the archive to be viewed/extracted using widely available tools.
// This does add a small amount of overhead to the file (N-NN KB depending on
// alignment requirements) and is mostly useful for debugging. Nothing in the
// runtime requires this information.
//
// Archive structure:
//  - [zip local file header for module]
//    [4b flatbuffers_uoffset_t defining module FlatBuffer length]
//    [module FlatBuffer contents]
//    [zero padding to 64b alignment]
//    <<rodata base offset>>
//  - [zip local file header for file 0]
//    [declared file 0 contents, aligned]
//  - [zip local file header for file 1]
//    ...
//  - [zip central directory]
//    [zip locators]
class ZIPArchiveWriter : public ArchiveWriter {
public:
  explicit ZIPArchiveWriter(Location loc, llvm::raw_ostream &os);
  ~ZIPArchiveWriter() override;
  bool supportsFiles() override { return true; }
  File declareFile(
      std::string fileName, uint64_t fileAlignment, uint64_t fileLength,
      std::function<LogicalResult(llvm::raw_ostream &os)> write) override;
  LogicalResult flush(FlatbufferBuilder &fbb) override;

private:
  Location loc;
  llvm::raw_ostream &os;
  uint64_t tailFileOffset = 0; // unpadded
  SmallVector<File> files;
};

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_ARCHIVE_WRITER_H_
