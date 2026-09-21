// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINAPI_PLUGINMANAGER_H_
#define IREE_COMPILER_PLUGINAPI_PLUGINMANAGER_H_

#include <optional>
#include <string>
#include <string_view>

#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/OptionUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir::iree_compiler {

class PluginManagerSession;
class PluginManager;

// Command line options for the plugin manager.
class PluginManagerOptions {
public:
  // Plugins to be activated in a session.
  llvm::SmallVector<std::string> plugins;

  // Print plugin information to stderr.
  bool printPluginInfo = false;

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<PluginManagerOptions>;
};

// Initialized before LLVM command-line parsing so plugins can add options.
class DynamicPluginRegistry {
public:
  DynamicPluginRegistry(const DynamicPluginRegistry &) = delete;
  DynamicPluginRegistry &operator=(const DynamicPluginRegistry &) = delete;

  enum class EnvPlugins { Disabled, Enabled };

  /// Empty until initialize().
  static DynamicPluginRegistry &get();

  /// Loads every plugin named in |args| and, if enabled, in IREE_LOAD_PLUGINS.
  /// Call once. The error joins every failure.
  [[nodiscard]] llvm::Error initialize(llvm::ArrayRef<const char *> args,
                                       EnvPlugins envPlugins);

  [[nodiscard]] bool registerPlugins(PluginManager *registrar) const;
  llvm::SmallVector<std::string> getLoadedPlugins() const;
  bool hasPluginPath(llvm::StringRef path) const;
  bool hasRegistrationFailures() const { return registrationFailed; }

private:
  struct Plugin {
    std::string path;
    std::string pluginId;
    // Never closed: registered code must outlive every session.
    void *library = nullptr;
    PluginRegistrationFunction registerFunction = nullptr;

    /// The id is read from the library.
    static llvm::Expected<Plugin> loadFromPath(llvm::StringRef path);
  };

  DynamicPluginRegistry() = default;

  void loadPluginsFromCL(llvm::ArrayRef<const char *> args);
  void loadPluginPathsFromEnv();
  void addPlugin(llvm::Expected<Plugin> plugin);

  bool initialized = false;
  mutable bool registrationFailed = false;
  llvm::Error loadErrors = llvm::Error::success();
  llvm::SmallVector<Plugin> plugins;
};

/// Loads once, however often it is called; later |args| are ignored. False if
/// any plugin failed, with the failures written to |os|.
bool initializeDynamicPlugins(llvm::ArrayRef<const char *> args,
                              llvm::raw_ostream &os);

/// Catches an --iree-load-plugin that reached llvm::cl without passing through
/// initializeDynamicPlugins, as through the C API.
bool verifyDynamicPluginFlags(llvm::raw_ostream &os);

// Manages global registrations for available plugins.
// Typically, there will be one PluginManager globally for the compiler, and
// it is initialized in ireeCompilerGlobalInitialize() based on statically
// compiled plugins or by querying environment variables and/or command
// line options to load dynamic plugins.
//
// At PluginManager initialization time, the only thing that is done is to
// record which plugins are registered and invoke their registration callback.
// This is responsible for registering flags and other global customizations.
//
// Most of the work of a plugin is done at session initialization time when
// an MLIRContext is available.
class PluginManager : public PluginRegistrar {
public:
  // Registers static plugins and previously loaded dynamic plugins. Returns
  // false on registration failure. With tolerateDynamicFailures, skips failed
  // dynamic registrations and returns false only for static failures.
  bool loadAvailablePlugins(bool tolerateDynamicFailures = false);

  // Calls through to AbstractPluginRegistration::globalInitialize for all
  // available plugins.
  void globalInitialize();

  // Calls through to AbstractPluginRegistration::registerPasses for all
  // available plugins.
  void registerPasses();

  // Calls through to AbstractPluginRegistration::initializeCLI for all
  // available plugins.
  void initializeCLI();

  // Calls through to AbstractPluginRegistration::registerGlobalDialects for all
  // available plugins.
  void registerGlobalDialects(DialectRegistry &registry);

  // Returns successfully registered plugin IDs in sorted order.
  llvm::SmallVector<std::string> getLoadedPlugins() const;

private:
  friend class DynamicPluginRegistry;
  friend class PluginManagerSession;
};

// Holds activated plugins for an |iree_compiler_session_t|.
class PluginManagerSession : public PipelineExtensions {
public:
  PluginManagerSession(PluginManager &pluginManager, OptionsBinder &binder,
                       PluginManagerOptions &options);

  // Initializes all plugins that should be activated by default.
  LogicalResult initializePlugins();

  // Invokes registerDialects() on all initialized plugins.
  void registerDialects(DialectRegistry &registry) override;

  // Activates plugins as configured.
  LogicalResult activatePlugins(MLIRContext *context);

  // Forward pipeline extensions.
  void extendInputConversionPreprocessingPassPipeline(
      OpPassManager &passManager,
      InputDialectOptions::Type inputType) override {
    for (auto *s : initializedSessions) {
      s->extendInputConversionPreprocessingPassPipeline(passManager, inputType);
    }
  }

  void populateCustomInputConversionTypes(
      llvm::StringSet<> &typeMnemonics) override {
    for (auto *s : initializedSessions) {
      s->populateCustomInputConversionTypes(typeMnemonics);
    }
  }

  void populateDetectedCustomInputConversionTypes(
      ModuleOp &module, llvm::StringSet<> &typeMnemonics) override {
    for (auto *s : initializedSessions) {
      s->populateDetectedCustomInputConversionTypes(module, typeMnemonics);
    }
  }

  bool extendCustomInputConversionPassPipeline(
      OpPassManager &passManager, std::string_view typeMnemonic) override {
    bool matched = false;
    for (auto *s : initializedSessions) {
      if (s->extendCustomInputConversionPassPipeline(passManager,
                                                     typeMnemonic)) {
        matched = true;
      }
    }
    return matched;
  }

  void extendPreprocessingPassPipeline(OpPassManager &passManager) override {
    for (auto *s : initializedSessions) {
      s->extendPreprocessingPassPipeline(passManager);
    }
  }

  // Populates the given list of HAL target devices for all initialized
  // plugins.
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &list);

  // Populates the given list of HAL target backends for all initialized
  // plugins.
  void populateHALTargetBackends(IREE::HAL::TargetBackendList &list);

private:
  PluginManagerOptions &options;
  // At construction, uninitialized plugin sessions are created for all
  // registered plugins so that CLI options can be set properly.
  llvm::StringMap<std::unique_ptr<AbstractPluginSession>> allPluginSessions;

  // All sessions that have opted to be default activated.
  llvm::StringMap<AbstractPluginSession *> defaultActivatedSessions;

  // Initialized list of plugins.
  llvm::SmallVector<AbstractPluginSession *> initializedSessions;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_PLUGINAPI_PLUGINMANAGER_H_
