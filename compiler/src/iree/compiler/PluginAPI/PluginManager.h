// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>
#include <string_view>
#include <vector>

#include "iree/compiler/Utils/OptionUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
class DialectRegistry;
class MLIRContext;
}  // namespace mlir

namespace mlir::iree_compiler {

class AbstractPluginSession;
class PluginManager;
struct PluginManagerSession;
class PluginRegistrar;

// Registration functions are exported with this signature.
using PluginRegistrationFunction = bool (*)(PluginRegistrar *);

// Empty options class that satisfies the contract for an OptionsBinder.
// Used by default if a plugin does not support options.
struct EmptyPluginOptions {
  static void bindOptions(OptionsBinder &binder) {}
};

// Abstract class representing a plugin registration. It is responsible for
// various global initialization and creation of plugin sessions that mirror
// the lifetime of and |iree_compiler_session_t| for when the plugin is
// activated.
//
// This is typically not instantiated directly but via the PluginSession
// CRTP helper which manages most details.
class AbstractPluginRegistration {
 public:
  AbstractPluginRegistration(std::string pluginId)
      : pluginId(std::move(pluginId)) {}
  virtual ~AbstractPluginRegistration();

  // Gets the plugin id. Valid for the life of the registration.
  std::string_view getPluginId() { return pluginId; }

  // Performs once-only global initialization. This is called prior to any
  // sessions being created and affects everything in the process. It is
  // best used for passes and other such things which require process level
  // initialization scope (i.e. such as hal targets or vendor library setup).
  // Since this happens unconditionally if a plugin is available, regardless
  // of whether activated, this should be used with caution and as a last
  // resort.
  // The default implementation does nothing.
  virtual void globalInitialize() {}

  // Initializes the process-global command line interface. This will be called
  // if the CLI is enabled, and if so, it indicates that created sessions
  // must configure any options from the global CLI environment.
  virtual void initializeCLI() {}

  // If a plugin is activated, performs all needed registrations into the
  // compiler session's DialectRegistry. By default, does nothing.
  // Note that this does global registration, regardless of whether the
  // plugin is loaded. It should be used sparingly for things that cannot
  // change behavior. It is safer to customize the context on a per-session
  // basis in a plugin session's activate() method (i.e. if registering
  // interfaces or behavior changes extensions).
  virtual void registerDialects(DialectRegistry &registry) {}

  // Creates an uninitialized session. If the CLI was initialized, then this
  // should also ensure that any command line options were managed properly into
  // the session instance. This will be called globally for all available
  // plugins so that option registration can happen first. It must have
  // no overhead beyond allocating some memory and setting up options.
  virtual std::unique_ptr<AbstractPluginSession> createUninitializedSession(
      OptionsBinder &localOptionsBinder) = 0;

 private:
  std::string pluginId;
};

// Primary base class that plugins extend to provide functionality and
// extensions to the compiler.
//
// A plugin session's life-cycle is bound to an |iree_compiler_session_t| for
// which it is activated (typically, a CLI will only have a single session, but
// APIs can create as many as they want within the same process).
//
// Most users will inherit from this class via the PluginSession CRTP helper,
// which adds some niceties and support for global command line option
// registration.
class AbstractPluginSession {
 public:
  virtual ~AbstractPluginSession();

  // Called after the session has been fully constructed. If it fails, then
  // it should emit an appropriate diagnostic.
  LogicalResult activate(MLIRContext *context);

 protected:
  // Called from the activate() method once pre-conditions are verified and the
  // context is set.
  virtual LogicalResult onActivate(){};

  MLIRContext *context = nullptr;
};

template <typename DerivedTy, typename OptionsTy = EmptyPluginOptions>
class PluginSession : public AbstractPluginSession {
 public:
  using Options = OptionsTy;
  const Options &getOptions() { return options; }

  // DerivedTy default implementations (no-op). Forwarded from the
  // AbstractPluginRegistration.
  static void globalInitialize() {}
  static void registerDialects(DialectRegistry &registry) {}

  struct Registration : public AbstractPluginRegistration {
    using AbstractPluginRegistration::AbstractPluginRegistration;
    void globalInitialize() override {
      // Forward to the CRTP derived type.
      DerivedTy::globalInitialize();
    }
    void initializeCLI() override {
      // Actually need to capture the reference, not a copy. So get a pointer.
      globalCLIOptions = &OptionsFromFlags<OptionsTy>::get();
    }
    void registerDialects(DialectRegistry &registry) override {
      // Forward to the CRTP derived type.
      DerivedTy::registerDialects(registry);
    }
    std::unique_ptr<AbstractPluginSession> createUninitializedSession(
        OptionsBinder &localOptionsBinder) override {
      auto instance = std::make_unique<DerivedTy>();
      if (globalCLIOptions) {
        // Bootstrap the local options with global CLI options.
        instance->options = *(*globalCLIOptions);
      }
      // And bind to the local binder for session level mnemonic customization.
      instance->options.bindOptions(localOptionsBinder);
      return instance;
    }
    std::optional<OptionsTy *> globalCLIOptions;
  };

 protected:
  OptionsTy options;
  friend class Registration;
};

// Interface to the registration system.
// Implemented by PluginManager.
class PluginRegistrar {
 public:
  // Register a plugin based on a registration class.
  void registerPlugin(std::unique_ptr<AbstractPluginRegistration> registration);

  // Registration helper which synthesizes a plugin registration based on
  // template parameters. It is expected that SessionTy extends the CRTP
  // base class PluginSession. The OptionsTy, if specified, must satisfy
  // the contract of an OptionsBinder.
  template <typename SessionTy>
  void registerPlugin(std::string pluginId) {
    auto registration =
        std::make_unique<typename SessionTy::Registration>(std::move(pluginId));
    registerPlugin(std::move(registration));
  }

 protected:
  llvm::StringMap<std::unique_ptr<AbstractPluginRegistration>> registrations;
};

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

  // Calls through to AbstractPluginRegistration::initializeCLI for all
  // available plugins.
  void initializeCLI();

  // Calls through to AbstractPluginRegistration::registerDialects for all
  // available plugins.
  void registerDialects(DialectRegistry &registry);

 private:
  friend class PluginManagerSession;
};

// Holds activated plugins for an |iree_compiler_session_t|.
class PluginManagerSession {
 public:
  PluginManagerSession(PluginManager &pluginManager, OptionsBinder &binder,
                       PluginManagerOptions &options);

  // Activates plugins as configured.
  LogicalResult activatePlugins(MLIRContext *context);

 private:
  PluginManagerOptions &options;
  // At construction, uninitialized plugin sessions are created for all
  // registered plugins so that CLI options can be set properly.
  llvm::StringMap<std::unique_ptr<AbstractPluginSession>> allPluginSessions;

  // Activation state.
  llvm::SmallVector<AbstractPluginSession *> activatedSessions;
};

}  // namespace mlir::iree_compiler
