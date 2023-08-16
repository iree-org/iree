// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/CASResources.h"

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
// PopulatedCASResource
//===----------------------------------------------------------------------===//

PopulatedCASResource::~PopulatedCASResource() {
  if (globalResource) {
    StringRef key = globalResource->getKey();
    LLVM_DEBUG(dbgs() << "[iree-cas-resource] Converting out of scope cas "
                         "resource to tombstone: "
                      << key << "\n");
    globalResource->getResource()->setBlob(
        CASResourceBuilder::getTombstoneBlob());
  }
}

//===----------------------------------------------------------------------===//
// CASResourceReader
//===----------------------------------------------------------------------===//

CASResourceReader::CASResourceReader(ArrayRef<char> data) : fullData(data) {
  if (fullData.size() < sizeof(CASResourceTrailer)) {
    valid = false;
  } else {
    size_t userDataSize = fullData.size() - sizeof(CASResourceTrailer);
    std::memcpy(&trailerCopy, &fullData[userDataSize],
                sizeof(CASResourceTrailer));
    userData = ArrayRef<char>(&fullData[0], userDataSize);
    if (trailerCopy.version != CURRENT_VERSION || trailerCopy.bomOne != 1 ||
        trailerCopy.bomZero != 0) {
      valid = false;
    } else {
      valid = true;
    }
  }
}

std::string CASResourceReader::getEncodedHashCode() {
  auto hashCode = getHashCode();
  return llvm::toHex(
      StringRef(reinterpret_cast<const char *>(&hashCode), sizeof(hashCode)));
}

//===----------------------------------------------------------------------===//
// CASResourceBuilder
//===----------------------------------------------------------------------===//

CASResourceBuilder CASResourceBuilder::allocateHeap(size_t dataSize) {
  size_t fullSize = dataSize + sizeof(CASResourceTrailer);
  return CASResourceBuilder(
      MutableArrayRef<char>(static_cast<char *>(malloc(fullSize)), fullSize),
      /*alignment=*/sizeof(double),
      /*deleter=*/[](void *data, size_t size, size_t align) {
        LLVM_DEBUG(dbgs() << "[iree-cas-resource] Free heap allocated (data="
                          << data << ", size=" << size << ", align=" << align
                          << ")\n");
        free(data);
      });
}

AsmResourceBlob CASResourceBuilder::getTombstoneBlob() {
  static CASResourceTrailer tombstoneTrailer = ([]() {
    CASResourceTrailer t;
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

CASResourceBuilder::~CASResourceBuilder() {
  if (deleter) {
    deleter((void *)fullData.data(), fullData.size(), alignment);
  }
}

CASResourceReader CASResourceBuilder::finalize() {
  if (!finalized) {
    CASResourceTrailer trailer;
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
  return CASResourceReader(fullData);
}

AsmResourceBlob CASResourceBuilder::createBlob() {
  finalize();
  // Transfer ownership.
  return AsmResourceBlob(fullData, alignment, std::move(deleter),
                         /*dataIsMutable=*/false);
}

//===----------------------------------------------------------------------===//
// CASManagerDialectInterface
//===----------------------------------------------------------------------===//

CASManagerDialectInterface::CASManagerDialectInterface(
    Dialect *dialect, UtilDialect::BlobManagerInterface &blobManager)
    : Base(dialect), blobManager(blobManager) {}
CASManagerDialectInterface::~CASManagerDialectInterface() {
  // Because this interface may be destroyed after the the blob manager
  // is torn down, we need to go through and detach anything we were
  // tracking so that it doesn't go through the normal tombstone swap
  // sequence on things that no longer exist. This isn't great.
  globalScope.detach();
  for (auto it : localScopes) {
    it.second.detach();
  }
}

void CASManagerDialectInterface::Scope::detach() {
  LLVM_DEBUG(dbgs() << "[iree-cas-resource] Detaching resources at context "
                       "destroy from scope "
                    << static_cast<void *>(this) << "\n");
  for (auto it : populatedResources) {
    Bucket &bucket = it.second;
    for (PopulatedCASResource::Reference &ref : bucket) {
      ref->detach();
    }
  }
}

CASManagerDialectInterface &
CASManagerDialectInterface::get(MLIRContext *context) {
  auto *iface =
      context->getOrLoadDialect<UtilDialect>()
          ->template getRegisteredInterface<CASManagerDialectInterface>();
  assert(iface && "util dialect does not register CASManagerDialectInterface");
  return *iface;
}

PopulatedCASResource::Reference
CASManagerDialectInterface::internGlobalResource(CASResourceBuilder builder) {
  return internResource(std::move(builder), globalScope);
}

PopulatedCASResource::Reference
CASManagerDialectInterface::internLocalResource(CASResourceBuilder builder,
                                                CASScopeAttr scopeAttr) {
  Scope &scope = localScopes[scopeAttr];
  // TODO: Really should be searching the global scope too before falling
  // back to creating.
  return internResource(std::move(builder), scope);
}

// Interns a local resource, accounted to a scope that is defined by
// CASScopeAttr::findOrCreateRootScope against the given operation.
PopulatedCASResource::Reference CASManagerDialectInterface::internLocalResource(
    CASResourceBuilder builder, Operation *findOrCreateScopeFor) {
  CASScopeAttr scope =
      CASScopeAttr::findOrCreateRootScope(findOrCreateScopeFor);
  return internLocalResource(std::move(builder), scope);
}

PopulatedCASResource::Reference
CASManagerDialectInterface::internResource(CASResourceBuilder builder,
                                           Scope &scope) {
  CASResourceReader reader = builder.finalize();
  assert(reader.isValid() && !reader.isDead() && "encoded blob is not valid");
  uint64_t hashCode = reader.getHashCode();
  auto foundIt = scope.populatedResources.find(hashCode);
  if (foundIt != scope.populatedResources.end()) {
    // Scan the bucket for an actual match.
    Bucket &bucket = foundIt->second;
    for (PopulatedCASResource::Reference &match : bucket) {
      assert(match->getBlob() && "blob is nullptr");
      CASResourceReader matchReader(match->getBlob()->getData());
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
  // Since we use the blob key as part of a resource handle in the IR, it must
  // syntactically be a valid identifier. Prefixing with a letter makes it legal
  // and also leaves us syntactic room in the future for stripped resources
  // with another prefix.
  std::string blobKey = std::string("R") + encodedHash;
  AsmResourceBlob createdBlob = builder.createBlob();
  UtilDialect::CASResourceHandle globalHandle =
      blobManager.insert(blobKey, std::move(createdBlob));
  PopulatedCASResource::Reference createdRef =
      std::make_shared<PopulatedCASResource>(std::move(globalHandle));
  scope.populatedResources[hashCode].push_back(createdRef);
  return createdRef;
}

void CASManagerDialectInterface::invalidateScope(CASScopeAttr scopeAttr) {
  LLVM_DEBUG(dbgs() << "[iree-cas-resource] Invalidate local scope "
                    << scopeAttr.getUniqueIndex() << "\n");
  localScopes.erase(scopeAttr);
}

} // namespace mlir::iree_compiler::IREE::Util
