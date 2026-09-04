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

// Built before llvm::cl so plugins can contribute CLI options.
class DynamicPluginRegistry {
public:
  DynamicPluginRegistry(const DynamicPluginRegistry &) = delete;
  DynamicPluginRegistry &operator=(const DynamicPluginRegistry &) = delete;

  enum class EnvPlugins { Disabled, Enabled };

  /// Empty until initialize(). Lives for the process, as do the libraries it
  /// records: getPermanentLibrary cannot unload them.
  static DynamicPluginRegistry &get();

  /// Loads every plugin named in |args| and, unless disabled, in
  /// IREE_LOAD_PLUGINS. Call at most once. The returned error joins every
  /// failure, so one bad plugin does not hide the next.
  [[nodiscard]] llvm::Error initialize(llvm::ArrayRef<const char *> args,
                                       EnvPlugins envPlugins);
  bool isInitialized() const { return initialized; }
  /// True once every plugin has loaded and resolved.
  bool isValid() const { return !loadFailed; }

  [[nodiscard]] bool registerPlugins(PluginRegistrar *registrar) const;
  llvm::SmallVector<std::string> getLoadedPlugins() const;

private:
  struct Plugin {
    std::string path;
    std::string pluginId;
    // RTLD_NOW, so a missing symbol fails here and not mid-compilation. Never
    // closed: a registered plugin's code must outlive every session.
    void *library = nullptr;
    PluginRegistrationFunction registerFunction = nullptr;

    /// The id comes from the library, so a caller needs only the path.
    static llvm::Expected<Plugin> loadFromPath(llvm::StringRef path);
  };

  DynamicPluginRegistry() = default;

  // Runs before llvm::cl: the plugins it loads still have options to add.
  void loadPluginsFromCL(llvm::ArrayRef<const char *> args);

  void loadPluginPathsFromEnv();

  /// Keeps a loaded plugin, or joins its failure onto loadErrors.
  void addPlugin(llvm::Expected<Plugin> plugin);

  bool initialized = false;
  llvm::Error loadErrors = llvm::Error::success();
  bool loadFailed = false;
  llvm::SmallVector<Plugin> plugins;
};

/// Loads the dynamic plugins named in |args| and in IREE_LOAD_PLUGINS, writing
/// failures to |os|. False if one failed. Loads once, however often it is
/// called.
bool initializeDynamicPlugins(llvm::ArrayRef<const char *> args,
                              llvm::raw_ostream &os);

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
  // Initializes the plugin manager. Since this may do shared library opening
  // and use failable initializers, it can fail. There probably isn't much to
  // do in that case but crash, but the choice is left to the caller.
  bool loadAvailablePlugins();

  // Calls through to AbstractPluginRegistration::globalInitialize for all
  // available plugins.
  void globalInitialize();

  // Calls through to AbstractPluginRegistration::registerPasses for all
  // available plugins.
  void registerPasses();

  // Calls through to AbstractPluginRegistration::initializeCLI for all
  // available plugins.
  void initializeCLI();

  // Calls through to AbstractPluginRegistration::registerDialects for all
  // available plugins.
  void registerGlobalDialects(DialectRegistry &registry);

  // Gets a list of all loaded plugin names.
  llvm::SmallVector<std::string> getLoadedPlugins() const;

private:
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
