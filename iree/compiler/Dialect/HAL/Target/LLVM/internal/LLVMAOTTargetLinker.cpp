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

#include "iree/base/status.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

iree::StatusOr<std::string> linkLLVMAOTObjects(
    const std::string& linkerToolPath, const std::string& objData) {
  std::string archiveFile, sharedLibFile;
  ASSIGN_OR_RETURN(archiveFile, iree::file_io::GetTempFile("objfile"));
  RETURN_IF_ERROR(iree::file_io::SetFileContents(archiveFile, objData));
  ASSIGN_OR_RETURN(sharedLibFile, iree::file_io::GetTempFile("dylibfile"));
  std::string linkingCmd =
      linkerToolPath + " -shared " + archiveFile + " -o " + sharedLibFile;
  int systemRet = system(linkingCmd.c_str());
  if (systemRet != 0) {
    return iree::InternalErrorBuilder(IREE_LOC)
           << linkingCmd << " failed with exit code " << systemRet;
  }
  return iree::file_io::GetFileContents(sharedLibFile);
}

iree::StatusOr<std::string> linkLLVMAOTObjectsWithLLDElf(
    const std::string& objData) {
  return iree::UnimplementedErrorBuilder(IREE_LOC)
         << "linkLLVMAOTObjectsWithLLD not implemented yet!";
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
