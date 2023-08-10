// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/CasResources.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

#include <cstdlib>
#include <cstring>

#define DEBUG_TYPE "iree-cas-resource"
using llvm::dbgs;

namespace mlir::iree_compiler::IREE::Util {

static const uint64_t CURRENT_VERSION = 1;

//===----------------------------------------------------------------------===//
// PopulatedCasResource
//===----------------------------------------------------------------------===//

PopulatedCasResource::~PopulatedCasResource() {
  if (globalResource) {
    StringRef key = globalResource->getKey();
    LLVM_DEBUG(dbgs() << "[iree-cas-resource] Converting out of scope cas "
                         "resource to tombstone: "
                      << key << "\n");
    globalResource->getResource()->setBlob(
        CasResourceBuilder::getTombstoneBlob());
  }
}

//===----------------------------------------------------------------------===//
// CasResourceReader
//===----------------------------------------------------------------------===//

CasResourceReader::CasResourceReader(ArrayRef<char> data) : fullData(data) {
  if (fullData.size() < sizeof(CasResourceTrailer)) {
    valid = false;
  } else {
    size_t userDataSize = fullData.size() - sizeof(CasResourceTrailer);
    std::memcpy(&trailerCopy, &fullData[userDataSize],
                sizeof(CasResourceTrailer));
    userData = ArrayRef<char>(&fullData[0], userDataSize);
    if (trailerCopy.version != CURRENT_VERSION || trailerCopy.bomOne != 1 ||
        trailerCopy.bomZero != 0) {
      valid = false;
    } else {
      valid = true;
    }
  }
}

std::string CasResourceReader::getEncodedHashCode() {
  auto hashCode = getHashCode();
  return llvm::toHex(
      StringRef(reinterpret_cast<const char *>(&hashCode), sizeof(hashCode)));
}

//===----------------------------------------------------------------------===//
// CasResourceBuilder
//===----------------------------------------------------------------------===//

CasResourceBuilder CasResourceBuilder::allocateHeap(size_t dataSize) {
  size_t fullSize = dataSize + sizeof(CasResourceTrailer);
  return CasResourceBuilder(
      MutableArrayRef<char>(static_cast<char *>(malloc(fullSize)), fullSize),
      /*alignment=*/sizeof(double),
      /*deleter=*/[](void *data, size_t size, size_t align) {
        LLVM_DEBUG(dbgs() << "[iree-cas-resource] Free heap allocated (data="
                          << data << ", size=" << size << ", align=" << align
                          << ")\n");
        free(data);
      });
}

AsmResourceBlob CasResourceBuilder::getTombstoneBlob() {
  static CasResourceTrailer tombstoneTrailer = ([]() {
    CasResourceTrailer t;
    std::memset(&t, 0, sizeof(t));
    t.bomOne = 1;
    t.bomZero = 0;
    t.version = CURRENT_VERSION;
    t.dead = 1;
    return t;
  })();
  return UnmanagedAsmResourceBlob::allocateWithAlign(
      llvm::ArrayRef<char>(reinterpret_cast<char *>(&tombstoneTrailer),
                           sizeof(tombstoneTrailer)),
      /*align=*/sizeof(uint64_t));
}

CasResourceBuilder::~CasResourceBuilder() {
  if (deleter) {
    deleter((void *)fullData.data(), fullData.size(), alignment);
  }
}

CasResourceReader CasResourceBuilder::finalize() {
  if (!finalized) {
    CasResourceTrailer trailer;
    trailer.bomOne = 1;
    trailer.dead = 0;
    trailer.version = CURRENT_VERSION;
    trailer.bomZero = 0;

    // FIXME: This is very bad and just for testing. We need a hash function
    // that is stable across runs.
    trailer.hashCode = llvm::hash_value(getData());

    std::memcpy((void *)&fullData[fullData.size() - sizeof(trailer)],
                (const void *)&trailer, sizeof(trailer));
    finalized = true;
  }
  return CasResourceReader(fullData);
}

AsmResourceBlob CasResourceBuilder::createBlob() {
  finalize();
  // Transfer ownership.
  return AsmResourceBlob(fullData, alignment, std::move(deleter),
                         /*dataIsMutable=*/false);
}

//===----------------------------------------------------------------------===//
// CasManagerDialectInterface
//===----------------------------------------------------------------------===//

CasManagerDialectInterface::CasManagerDialectInterface(
    Dialect *dialect, UtilDialect::BlobManagerInterface &blobManager)
    : Base(dialect), blobManager(blobManager) {}
CasManagerDialectInterface::~CasManagerDialectInterface() {
  // Because this interface may be destroyed after the the blob manager
  // is torn down, we need to go through and detach anything we were
  // tracking so that it doesn't go through the normal tombstone swap
  // sequence on things that no longer exist. This isn't great.
  globalScope.detach();
  for (auto it : localScopes) {
    it.second.detach();
  }
}

void CasManagerDialectInterface::Scope::detach() {
  LLVM_DEBUG(dbgs() << "[iree-cas-resource] Detaching resources at context "
                       "destroy from scope "
                    << static_cast<void *>(this) << "\n");
  for (auto it : populatedResources) {
    Bucket &bucket = it.second;
    for (PopulatedCasResource::Reference &ref : bucket) {
      ref->detach();
    }
  }
}

CasManagerDialectInterface &
CasManagerDialectInterface::get(MLIRContext *context) {
  auto *iface =
      context->getOrLoadDialect<UtilDialect>()
          ->template getRegisteredInterface<CasManagerDialectInterface>();
  assert(iface && "util dialect does not register CasManagerDialectInterface");
  return *iface;
}

PopulatedCasResource::Reference
CasManagerDialectInterface::internGlobalResource(CasResourceBuilder builder) {
  return internResource(std::move(builder), globalScope);
}

PopulatedCasResource::Reference
CasManagerDialectInterface::internLocalResource(CasResourceBuilder builder,
                                                CasScopeAttr scopeAttr) {
  Scope &scope = localScopes[scopeAttr];
  // TODO: Really should be searching the global scope too before falling
  // back to creating.
  return internResource(std::move(builder), scope);
}

// Interns a local resource, accounted to a scope that is defined by
// CasScopeAttr::findOrCreateRootScope against the given operation.
PopulatedCasResource::Reference CasManagerDialectInterface::internLocalResource(
    CasResourceBuilder builder, Operation *findOrCreateScopeFor) {
  CasScopeAttr scope =
      CasScopeAttr::findOrCreateRootScope(findOrCreateScopeFor);
  return internLocalResource(std::move(builder), scope);
}

PopulatedCasResource::Reference
CasManagerDialectInterface::internResource(CasResourceBuilder builder,
                                           Scope &scope) {
  CasResourceReader reader = builder.finalize();
  assert(reader.isValid() && !reader.isDead() && "encoded blob is not valid");
  uint64_t hashCode = reader.getHashCode();
  auto foundIt = scope.populatedResources.find(hashCode);
  if (foundIt != scope.populatedResources.end()) {
    // Scan the bucket for an actual match.
    Bucket &bucket = foundIt->second;
    for (PopulatedCasResource::Reference &match : bucket) {
      assert(match->getBlob() && "blob is nullptr");
      CasResourceReader matchReader(match->getBlob()->getData());
      assert(matchReader.isValid() && !matchReader.isDead() &&
             "cas resource bucket contains invalid or dead blob");
      // Trivially reject on size mismatch.
      if (matchReader.getData().size() != reader.getData().size())
        continue;
      // Reject on content mismatch.
      if (std::memcmp(matchReader.getData().data(), reader.getData().data(),
                      reader.getData().size()) != 0)
        continue;
      // Matches.
      LLVM_DEBUG(
          dbgs() << "[iree-cas-resource] Return existing cas resource for "
                 << hashCode << "\n");
      return match;
    }
  }

  // If here, then intern failed.
  LLVM_DEBUG(dbgs() << "[iree-cas-resource] Interning new cas resource for "
                    << hashCode << "\n");
  std::string encodedHash = reader.getEncodedHashCode();
  AsmResourceBlob createdBlob = builder.createBlob();
  UtilDialect::CasResourceHandle globalHandle =
      blobManager.insert(encodedHash, std::move(createdBlob));
  PopulatedCasResource::Reference createdRef =
      std::make_shared<PopulatedCasResource>(std::move(globalHandle));
  scope.populatedResources[hashCode].push_back(createdRef);
  return createdRef;
}

void CasManagerDialectInterface::invalidateScope(CasScopeAttr scopeAttr) {
  LLVM_DEBUG(dbgs() << "[iree-cas-resource] Invalidate local scope "
                    << scopeAttr.getUniqueIndex() << "\n");
  localScopes.erase(scopeAttr);
}

} // namespace mlir::iree_compiler::IREE::Util
