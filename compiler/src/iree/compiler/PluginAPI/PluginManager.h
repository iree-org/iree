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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/DynamicLibrary.h"

namespace mlir::iree_compiler {

class PluginManager;
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

// Tracks dynamically loaded plugins and their registration callbacks.
// Created once early so plugins can contribute CLI options.
class DynamicPluginRegistry {
public:
  /// Creates the singleton instance and returns true on success; may only be
  /// called once.
  [[nodiscard]] static bool create(int argc, char **argv,
                                   bool allowEnvPlugins = true);

  /// Returns true if the singleton instance exists
  static bool hasInstance();

  /// Returns the singleton instance (must be created first).
  static DynamicPluginRegistry &get();

public:
  /// Registers all loaded plugins with the given registrar.
  [[nodiscard]] bool registerPlugins(PluginRegistrar *registrar) const;
  /// Returns the plugin identifiers that were loaded.
  llvm::SmallVector<std::string> getLoadedPlugins() const;
  /// Emits any load/registration errors to the given stream.
  void reportErrors(llvm::raw_ostream &os) const;
  /// True if all plugins loaded and resolved successfully.
  bool isValid() const;

private:
  struct Plugin {
    using RegisterFunction = bool (*)(mlir::iree_compiler::PluginRegistrar *);

    std::string path;
    std::string pluginId;
    llvm::sys::DynamicLibrary library;
    std::optional<std::string> error;
    RegisterFunction registerFunction = nullptr;

    /// Loads a plugin from a string of the form <plugin_id>=<path>.
    static Plugin loadFromString(std::string_view str);

    bool isValid() const { return !error.has_value(); }
  };

  // Singleton instance, only allowed to be created via `create()`.
  DynamicPluginRegistry() = default;

  // We need to manually parse out the plugin options before we initialize
  // the PluginManager since it needs to load plugins before we can parse the
  // full set of command line options.
  void loadPluginsFromCL(int argc, char **argv);

  void loadPluginPathsFromEnv();

private:
  llvm::SmallVector<Plugin, 4> plugins;
};

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
  PluginManager();

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
  llvm::SmallVector<std::string> getLoadedPlugins();

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
