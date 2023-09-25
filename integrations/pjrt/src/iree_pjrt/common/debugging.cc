// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_pjrt/common/debugging.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace iree::pjrt {

//===----------------------------------------------------------------------===//
// Logger
//===----------------------------------------------------------------------===//

void Logger::debug(std::string_view message) {
  std::cerr << "[IREE-PJRT] DEBUG: " << message << std::endl;
}

void Logger::error(std::string_view message) {
  std::cerr << "[IREE-PJRT] ERROR: " << message << std::endl;
}

//===----------------------------------------------------------------------===//
// ArtifactDumper
//===----------------------------------------------------------------------===//

ArtifactDumper::~ArtifactDumper() = default;

std::unique_ptr<ArtifactDumper::Transaction>
ArtifactDumper::CreateTransaction() {
  return nullptr;
}

std::string ArtifactDumper::DebugString() { return std::string("disabled"); }

//===----------------------------------------------------------------------===//
// FilesArtifactDumper
//===----------------------------------------------------------------------===//

class FilesArtifactDumper::FilesTransaction final
    : public ArtifactDumper::Transaction {
 public:
  FilesTransaction(Logger& logger, std::filesystem::path base_path,
                   int64_t transaction_id, bool retain_all)
      : logger_(logger),
        base_path_(std::move(base_path)),
        transaction_id_(transaction_id),
        retain_all_(retain_all) {}
  ~FilesTransaction() { Retain(); }

  std::optional<std::string> AllocateArtifactPath(std::string_view label,
                                                  std::string_view extension,
                                                  int index) override {
    std::string basename = std::to_string(transaction_id_);
    basename.append("-");
    basename.append(label);
    if (index >= 0) {
      basename.append(std::to_string(index));
    }
    basename.append(".");
    basename.append(extension);

    auto file_path = base_path_ / basename;
    written_paths_.push_back(file_path);

    return file_path;
  }

  void WriteArtifact(std::string_view label, std::string_view extension,
                     int index, std::string_view contents) override {
    // Note that the API declares this as optional, but our implementation
    // always returns a path.
    auto file_path = AllocateArtifactPath(label, extension, index);

    std::ofstream fout;
    fout.open(*file_path, std::ofstream::out | std::ofstream::trunc |
                              std::ofstream::binary);
    fout.write(contents.data(), contents.size());
    fout.close();

    if (!fout.good()) {
      std::string message("I/O error dumping artifact: ");
      message.append(*file_path);
      logger_.error(message);
    }
  }

  void Retain() override {
    if (written_paths_.empty()) return;

    std::string message;
    message.append("Retained artifacts in: ");
    message.append(base_path_);
    for (auto& p : written_paths_) {
      message.append("\n  ");
      message.append(p.filename());
    }
    logger_.debug(message);

    written_paths_.clear();
  }

  void Cancel() override {
    if (retain_all_) {
      Retain();
      return;
    }

    for (auto& p : written_paths_) {
      std::error_code ec;
      std::filesystem::remove(p, ec);
      if (ec) {
        // Only carp as a debug message since there are legitimate reasons
        // this can happen depending on what is going on at the system level.
        std::string message("Error removing artifact: ");
        message.append(p);
        logger_.debug(message);
      }
    }

    written_paths_.clear();
  }

 private:
  Logger& logger_;
  std::vector<std::filesystem::path> written_paths_;
  std::filesystem::path base_path_;
  int64_t transaction_id_;
  bool retain_all_;
};

FilesArtifactDumper::FilesArtifactDumper(Logger& logger,
                                         PathCallback path_callback,
                                         bool retain_all)
    : logger_(logger),
      path_callback_(std::move(path_callback)),
      retain_all_(retain_all) {
  enabled_ = true;
}

FilesArtifactDumper::~FilesArtifactDumper() = default;

std::unique_ptr<ArtifactDumper::Transaction>
FilesArtifactDumper::CreateTransaction() {
  auto maybe_path = path_callback_();
  if (!maybe_path) {
    return nullptr;
  }

  std::filesystem::path path(*maybe_path);
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  if (ec) {
    std::string message("Error creating artifact directory '");
    message.append(path);
    message.append("' (artifact dumping disabled): ");
    message.append(ec.message());
    logger_.error(message);
    return nullptr;
  }

  return std::make_unique<FilesTransaction>(
      logger_, path, next_transaction_id_.fetch_add(1), retain_all_);
}

std::string FilesArtifactDumper::DebugString() {
  return std::string("dump to files");
}

}  // namespace iree::pjrt
