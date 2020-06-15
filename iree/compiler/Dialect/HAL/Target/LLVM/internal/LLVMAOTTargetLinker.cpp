// Copyright 2020 Google LLC
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

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMAOTTargetLinker.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

iree::StatusOr<std::string> linkLLVMAOTObjects(const std::string& objData) {
  // Link obj file with target linker toochain.
  std::string archiveFile, sharedLibFile;
  auto archiveFileStatus = iree::file_io::GetTempFile("objfile");
  if (!archiveFileStatus.ok()) return archiveFileStatus.status();
  archiveFile = archiveFileStatus.value();
  auto setArchiveDataStatus =
      iree::file_io::SetFileContents(archiveFile, objData);
  if (!setArchiveDataStatus.ok()) return archiveFileStatus;
  auto sharedLibFileStatus = iree::file_io::GetTempFile("dylibfile");
  if (!sharedLibFileStatus.ok()) return sharedLibFileStatus;
  sharedLibFile = sharedLibFileStatus.value();
  std::string linkingCmd =
      "/usr/bin/ld -shared " + archiveFile + " -o " + sharedLibFile;
  system(linkingCmd.c_str());
  return iree::file_io::GetFileContents(sharedLibFile);
}

iree::StatusOr<std::string> linkLLVMAOTObjectsWithLLD(
    const std::string& objData) {
  return iree::UnimplementedErrorBuilder(IREE_LOC)
         << "linkLLVMAOTObjectsWithLLD not implemented yet!";
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
