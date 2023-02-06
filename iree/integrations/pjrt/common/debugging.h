// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_DEBUGGING_H_
#define IREE_PJRT_PLUGIN_PJRT_DEBUGGING_H_

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace iree::pjrt {

//===----------------------------------------------------------------------===//
// Logger
// The plugin API currently does not have any logging facilities, but since
// these are easier added later, we have a placeholder Logger that we thread
// through. It can be extended later.
//===----------------------------------------------------------------------===//

class Logger {
 public:
  Logger() = default;
  void debug(std::string_view message);
  void error(std::string_view message);
};

//===----------------------------------------------------------------------===//
// ArtifactDumper
// The typical modes of operation are:
//   - Completely disabled
//   - Enabled to persist all artifacts
//   - Enabled to persist artifacts on recoverable errors (i.e. compile errors,
//     etc)
//   - Enabled to persist artifacts on crash only
// Since we often don't know what the outcome of a transaction will be right
// away, the transaction can be held open for some period of time and cancelled
// later.
//
// In general, since this is a debugging facility, it swallows any errors,
// reporting them to the Logger.
//
// The default implementation can be instantiated and is hard-coded to
// !enabled().
//===----------------------------------------------------------------------===//

class ArtifactDumper {
 public:
  class Transaction {
   public:
    virtual ~Transaction() = default;

    // Writes an artifact with a transaction-unique label/index.
    virtual void WriteArtifact(std::string_view label,
                               std::string_view extension, int index,
                               std::string_view contents) = 0;

    // Allocates an artifact which can be written to a path externally.
    // Since not all dumpers may support writing to such a location on a local
    // filesystem, this returns an optional. If calling this function, it
    // is expected that the path is actually written to.
    virtual std::optional<std::string> AllocateArtifactPath(
        std::string_view label, std::string_view extension, int index) = 0;

    // Completes the transaction and retains all artifacts.
    virtual void Retain() = 0;

    // Completes the transaction and removes all artifacts.
    virtual void Cancel() = 0;
  };

  virtual ~ArtifactDumper();

  // Not virtual for quick checks in disabled state.
  bool enabled() { return enabled_; }

  // Allocates a new transaction which can be used to append related
  // artifacts. All artifacts associated with a transaction can be removed upon
  // a successful transaction, or they will be retained upon crash or
  // recoverable error.
  // Must only be called if enabled(). May return nullptr if saving should not
  // be done.
  virtual std::unique_ptr<Transaction> CreateTransaction();

  // Returns a string suitable for emitting to the debug log, describing
  // where and how artifacts will be retained.
  virtual std::string DebugString();

 protected:
  bool enabled_ = false;
};

// Dumps artifacts to a path on the file system.
class FilesArtifactDumper : public ArtifactDumper {
 public:
  class FilesTransaction;
  using PathCallback = std::function<std::optional<std::string>()>;

  FilesArtifactDumper(Logger& logger, PathCallback path_callback,
                      bool retain_all);
  ~FilesArtifactDumper() override;

  std::unique_ptr<Transaction> CreateTransaction() override;
  std::string DebugString() override;

 private:
  Logger& logger_;
  std::atomic<int64_t> next_transaction_id_{0};
  PathCallback path_callback_;
  bool retain_all_;
};

}  // namespace iree::pjrt

#endif  // IREE_PJRT_PLUGIN_PJRT_DEBUGGING_H_
