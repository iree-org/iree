// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_COMMON_COMPILER_H_
#define IREE_PJRT_PLUGIN_PJRT_COMMON_COMPILER_H_

#include <memory>
#include <string>

#include "iree_pjrt/common/debugging.h"
// TODO: Excise.
// #include "xla/pjrt/pjrt_executable.h"

namespace iree::pjrt {

class CompilerOutput {
 public:
  virtual ~CompilerOutput() = default;
  virtual void* GetData() = 0;
  virtual size_t GetDataSize() = 0;
};

// A single compilation job.
class CompilerJob {
 public:
  virtual ~CompilerJob() = default;

  // Enables crash dumping via an ArtifactDumper transaction. The transaction
  // must remain valid for the duration of the job.
  virtual void EnableCrashDumps(
      ArtifactDumper::Transaction* artifact_transaction) = 0;

  // Sets a flag on the compiler job. This should only be done during shared
  // setup of a job (or if the underlying session will not be re-used).
  // Returns false on failure.
  virtual bool SetFlag(const char* flag) = 0;
  // TODO: Excise.
  // virtual bool SetFlags(xla::CompileOptions options) = 0;

  // Gets all flags as a string. This is intended for debug printing a plausible
  // command line to reproduce compilation.
  virtual std::string GetFlags() = 0;

  // Parses the source buffer. The buffer must remain valid for the life of
  // the job. Some implementations will parse it immediately, while other
  // backends may need to defer processing it.
  // Returns false if parsing failed (diagnostics will be output).
  virtual bool ParseSourceBuffer(const void* buffer, size_t length) = 0;

  // Performs compilation and code emission.
  virtual std::unique_ptr<CompilerOutput> CompileStandardPipeline() = 0;

  // If an operation failed, then an additional error message may be
  // available.
  virtual std::string GetErrorMessage() = 0;
};

// Wraps invocations of the compiler, either in-process or via other means.
class AbstractCompiler {
 public:
  virtual ~AbstractCompiler() = default;

  // Starts a new compiler job.
  virtual std::unique_ptr<CompilerJob> StartJob() = 0;

  // Gets descriptive revision information which identifies the version of
  // the compiler and/or APIs of the compiler.
  virtual std::string GetRevision() = 0;

  // If an operation failed, then an additional error message may be
  // available.
  virtual std::string GetErrorMessage() = 0;
};

// An AbstractCompiler based on IREE.
class IREECompiler : public AbstractCompiler {
 public:
  std::unique_ptr<CompilerJob> StartJob() override;
  std::string GetRevision() override;
  std::string GetErrorMessage() override { return error_message_; }

 private:
  std::string error_message_;
};

// An AbstractCompiler based on the HLO partitioner.
class OpenXLAPartitioner : public AbstractCompiler {
 public:
  std::unique_ptr<CompilerJob> StartJob() override;
  std::string GetRevision() override;
  std::string GetErrorMessage() override { return error_message_; }

 private:
  std::string error_message_;
};

}  // namespace iree::pjrt

#endif  // IREE_PJRT_PLUGIN_PJRT_COMMON_COMPILER_H_
