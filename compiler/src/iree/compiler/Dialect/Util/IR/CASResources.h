// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_IR_CASRESOURCES_H_
#define IREE_COMPILER_DIALECT_UTIL_IR_CASRESOURCES_H_

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectInterface.h"

#include <memory>
#include <optional>
#include <vector>

namespace mlir::iree_compiler::IREE::Util {

// A PopulatedCASResource holds a live reference to a backing CASResourceHandle
// that has been globally registered in the dialect's blob manager.
// While the lifetime of the registration is context-scoped, the
// lifetime of the contents are bounded by PopulatedCASResource instances,
// which have shared_ptr semantics.
// When the CASResource is destructed, a tombstone blob will replace
// the existing data, freeing up storage.
class PopulatedCASResource
    : public std::enable_shared_from_this<PopulatedCASResource> {
public:
  using Reference = std::shared_ptr<PopulatedCASResource>;

  PopulatedCASResource(UtilDialect::CASResourceHandle globalResource)
      : globalResource(globalResource) {}
  ~PopulatedCASResource();
  void detach() { globalResource.reset(); }

  Reference ref() { return shared_from_this(); }

  UtilDialect::CASResourceHandle getGlobalResource() { return *globalResource; }
  AsmResourceBlob *getBlob() { return globalResource->getBlob(); }

private:
  // The backing resource, owned by the dialect and with context
  // scoped lifetime.
  std::optional<UtilDialect::CASResourceHandle> globalResource;
};

// CAS resource blobs have a trailer which follows the blob data
// and is always of a fixed size. By making this a trailer, we don't
// disrupt the native alignment of the main data and merely have to
// memcpy this record to access it in an aligned way.
struct CASResourceTrailer {
  // Stable hash code computed relative to the version. This hash
  // code is stable run-to-run and safe for serialization. It is
  // not guaranteed unique and is used as a bucket identifier, not
  // for guaranteed equality of values.
  uint64_t hashCode;

  // Always 1.
  uint64_t bomOne : 1;
  // If 1, then the CAS record is dead. this is usually the result of
  // a garbage collection process (which can mutate/free records but
  // does not de-index them).
  uint64_t dead : 1;
  uint64_t : 30;
  uint64_t version : 31;
  // Always 0.
  uint64_t bomZero : 1;
};
static_assert(sizeof(CASResourceTrailer) == 2 * sizeof(uint64_t));

// Given raw cas data, manages read access to it.
class CASResourceReader {
public:
  CASResourceReader() = default;
  CASResourceReader(ArrayRef<char> data);
  bool isValid() { return valid; }
  const CASResourceTrailer &getTrailer() { return trailerCopy; }
  ArrayRef<char> getData() { return userData; }
  bool isDead() { return trailerCopy.dead; }
  uint64_t getHashCode() { return trailerCopy.hashCode; }
  std::string getEncodedHashCode();

private:
  ArrayRef<char> fullData;
  CASResourceTrailer trailerCopy;
  ArrayRef<char> userData;
  bool valid = false;
};

// Builds a cas resource of a size known in advance.
class CASResourceBuilder {
public:
  ~CASResourceBuilder();
  CASResourceBuilder(CASResourceBuilder &&other)
      : fullData(other.fullData), alignment(other.alignment),
        deleter(std::move(other.deleter)), finalized(other.finalized) {}
  static CASResourceBuilder allocateHeap(size_t dataSize);
  static AsmResourceBlob getTombstoneBlob();

  MutableArrayRef<char> getData() {
    return MutableArrayRef<char>(fullData.data(),
                                 fullData.size() - sizeof(CASResourceTrailer));
  }

  // Access the data as a type. If the type does not evenly divide the
  // size, it is rounded down.
  template <typename T>
  MutableArrayRef<T> getTypedData() {
    auto data = getData();
    return MutableArrayRef<T>((T *)data.data(), data.size() / sizeof(T));
  }

private:
  CASResourceBuilder(MutableArrayRef<char> fullData, size_t alignment,
                     AsmResourceBlob::DeleterFn deleter)
      : fullData(fullData), alignment(alignment), deleter(std::move(deleter)) {}

  // Finalizes after all mutations have been performed.
  CASResourceReader finalize();

  // When done building, finalize and return the AsmResourceBlob.
  // This ends the life-cycle of the builder and it cannot be
  // used anymore.
  AsmResourceBlob createBlob();

  MutableArrayRef<char> fullData;
  size_t alignment;
  AsmResourceBlob::DeleterFn deleter;
  bool finalized = false;
  friend class CASManagerDialectInterface;
};

// Dialect interface implemented by the UtilDialect which mediates
// access to blob resources managed in a content-addressable database.
class CASManagerDialectInterface
    : public DialectInterface::Base<CASManagerDialectInterface> {
public:
  CASManagerDialectInterface(Dialect *dialect,
                             UtilDialect::BlobManagerInterface &blobManager);
  ~CASManagerDialectInterface();

  // Gets the interface from the context.
  static CASManagerDialectInterface &get(MLIRContext *context);

  // Interns a resource from a builder that has been populated.
  // The resource is added to the global scope.
  PopulatedCASResource::Reference
  internGlobalResource(CASResourceBuilder builder);

  // Interns a local resource, accounted to the given scope.
  PopulatedCASResource::Reference
  internLocalResource(CASResourceBuilder builder, CASScopeAttr scopeAttr);

  // Interns a local resource, accounted to a scope that is defined by
  // CASScopeAttr::findOrCreateRootScope against the given operation.
  PopulatedCASResource::Reference
  internLocalResource(CASResourceBuilder builder,
                      Operation *findOrCreateScopeFor);

  // Invalidates all resources associated to a scope.
  // TODO: Have an exclusion set to enable partial GC.
  void invalidateScope(CASScopeAttr scopeAttr);

private:
  // A bucket contains all populated resources with a given hash key.
  using Bucket = llvm::SmallVector<PopulatedCASResource::Reference>;
  struct Scope {
    llvm::DenseMap<uint64_t, Bucket> populatedResources;

    void detach();
  };

  PopulatedCASResource::Reference internResource(CASResourceBuilder builder,
                                                 Scope &scope);

  UtilDialect::BlobManagerInterface &blobManager;

  // A scope that is bounded by the context (effectively forever).
  // TODO: Protect these with RW locks.
  Scope globalScope;
  // Local scopes are defined against CASScopeAttr instances.
  llvm::DenseMap<CASScopeAttr, Scope> localScopes;
};

} // namespace mlir::iree_compiler::IREE::Util

#endif // IREE_COMPILER_DIALECT_UTIL_IR_CASRESOURCES_H_