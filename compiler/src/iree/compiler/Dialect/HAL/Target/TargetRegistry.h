// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_TARGETREGISTRY_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_TARGETREGISTRY_H_

#include <mutex>
#include <string>
#include <vector>

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"

namespace mlir::iree_compiler::IREE::HAL {

//===----------------------------------------------------------------------===//
// TargetRegistration
//===----------------------------------------------------------------------===//

template <typename T>
using TargetFactoryFn = std::function<std::shared_ptr<T>()>;

template <typename T>
class TargetRegistration {
public:
  TargetRegistration(TargetFactoryFn<T> fn) : initFn(std::move(fn)) {}
  std::shared_ptr<T> acquire() {
    std::call_once(initFlag, [&]() { cachedValue = initFn(); });
    return cachedValue;
  }

protected:
  TargetFactoryFn<T> initFn;
  std::once_flag initFlag;
  std::shared_ptr<T> cachedValue;
};
class TargetBackendRegistration : public TargetRegistration<TargetBackend> {
public:
  // TODO(#15468): remove the registerStaticGlobal mode once callers are
  // migrated and move the constructor to the template type.
  TargetBackendRegistration(StringRef name, TargetFactoryFn<TargetBackend> fn,
                            bool registerStaticGlobal = true);
};
class TargetDeviceRegistration : public TargetRegistration<TargetDevice> {
public:
  // TODO(#15468): remove the registerStaticGlobal mode once callers are
  // migrated and move the constructor to the template type.
  TargetDeviceRegistration(StringRef name, TargetFactoryFn<TargetDevice> fn,
                           bool registerStaticGlobal = true);
};

template <typename T>
class TargetFactoryList {
public:
  void add(llvm::StringRef name, TargetFactoryFn<T> fn) {
    entries.push_back(std::make_pair(name.str(), fn));
  }

private:
  llvm::SmallVector<std::pair<std::string, TargetFactoryFn<T>>> entries;
  friend class TargetRegistry;
};
class TargetBackendList : public TargetFactoryList<TargetBackend> {};
class TargetDeviceList : public TargetFactoryList<TargetDevice> {};

//===----------------------------------------------------------------------===//
// TargetRegistry
//===----------------------------------------------------------------------===//

// A concrete target registry.
class TargetRegistry {
public:
  // Returns the read-only global registry.
  // This is used by passes which depend on it from their default constructor.
  static const TargetRegistry &getGlobal();

  // Merge from a list of of target backends.
  // The receiving registry will own the registration entries.
  void mergeFrom(const TargetBackendList &targetBackends);
  // Merge from a list of of target devices.
  // The receiving registry will own the registration entries.
  void mergeFrom(const TargetDeviceList &targetDevices);

  // Initialize from an existing registry. This registry will not own the
  // backing registration entries. The source registry must remain live for the
  // life of this.
  // TODO(15468): remove the static registration and require only plugins.
  void mergeFrom(const TargetRegistry &registry);

  // Returns a list of registered target backends.
  std::vector<std::string> getRegisteredTargetBackends() const;
  // Returns a list of registered target devices.
  std::vector<std::string> getRegisteredTargetDevices() const;

  // Returns the target backend with the given name.
  std::shared_ptr<TargetBackend> getTargetBackend(StringRef targetName) const;
  // Returns the target device with the given name.
  std::shared_ptr<TargetDevice> getTargetDevice(StringRef targetName) const;

  // Returns one backend per entry in |targetNames|.
  SmallVector<std::shared_ptr<TargetBackend>>
  getTargetBackends(ArrayRef<std::string> targetNames) const;
  // Returns one device per entry in |targetNames|.
  SmallVector<std::shared_ptr<TargetDevice>>
  getTargetDevices(ArrayRef<std::string> targetNames) const;

private:
  llvm::StringMap<TargetBackendRegistration *> backendRegistrations;
  llvm::SmallVector<std::unique_ptr<TargetBackendRegistration>>
      ownedBackendRegistrations;
  llvm::StringMap<TargetDeviceRegistration *> deviceRegistrations;
  llvm::SmallVector<std::unique_ptr<TargetDeviceRegistration>>
      ownedDeviceRegistrations;

  // TODO(#15468): remove this when not used by LLVMCPU/VulkanSPIRV.
  friend class TargetBackendRegistration;
  friend class TargetDeviceRegistration;
};

} // namespace mlir::iree_compiler::IREE::HAL

//===----------------------------------------------------------------------===//
// TargetRegistryRef
//===----------------------------------------------------------------------===//

namespace llvm::cl {

struct TargetRegistryRef {
  const mlir::iree_compiler::IREE::HAL::TargetRegistry *value =
      &mlir::iree_compiler::IREE::HAL::TargetRegistry::getGlobal();
  TargetRegistryRef() = default;
  TargetRegistryRef(const mlir::iree_compiler::IREE::HAL::TargetRegistry &value)
      : value(&value) {}
  TargetRegistryRef(const mlir::iree_compiler::IREE::HAL::TargetRegistry *value)
      : value(value) {}
  operator bool() const noexcept {
    return value->getRegisteredTargetBackends() !=
           mlir::iree_compiler::IREE::HAL::TargetRegistry::getGlobal()
               .getRegisteredTargetBackends();
  }
  const mlir::iree_compiler::IREE::HAL::TargetRegistry *operator->() const {
    return value;
  }
};

extern template class basic_parser<TargetRegistryRef>;

template <>
class parser<TargetRegistryRef> : public basic_parser<TargetRegistryRef> {
public:
  parser(Option &O) : basic_parser(O) {}
  bool parse(Option &O, StringRef ArgName, StringRef Arg,
             TargetRegistryRef &Val);
  StringRef getValueName() const override { return "target registry"; }
  void printOptionDiff(const Option &O, TargetRegistryRef V,
                       const OptVal &Default, size_t GlobalWidth) const;
  void anchor() override;
};

} // namespace llvm::cl

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_TARGETREGISTRY_H_
