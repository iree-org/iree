// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/executable_plugin_manager.h"

#include "iree/base/internal/synchronization.h"
#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// Plugin API compatibility checks
//===----------------------------------------------------------------------===//

// Ensures that the copies/shims we have in the plugin API match what's in the
// compiler. If there are compilation failures here it means we've changed one
// side and not the other. Note that additions in the runtime are fine so long
// as they don't disturb the existing values exposed to the plugin.

#define STATIC_ASSERT_EQ(a, b) \
  static_assert((a) == (b), "plugin/runtime API mismatch")

STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_OK, IREE_STATUS_OK);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_CANCELLED,
                 IREE_STATUS_CANCELLED);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_UNKNOWN,
                 IREE_STATUS_UNKNOWN);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_INVALID_ARGUMENT,
                 IREE_STATUS_INVALID_ARGUMENT);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_DEADLINE_EXCEEDED,
                 IREE_STATUS_DEADLINE_EXCEEDED);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_NOT_FOUND,
                 IREE_STATUS_NOT_FOUND);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_ALREADY_EXISTS,
                 IREE_STATUS_ALREADY_EXISTS);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_PERMISSION_DENIED,
                 IREE_STATUS_PERMISSION_DENIED);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_RESOURCE_EXHAUSTED,
                 IREE_STATUS_RESOURCE_EXHAUSTED);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_FAILED_PRECONDITION,
                 IREE_STATUS_FAILED_PRECONDITION);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_ABORTED,
                 IREE_STATUS_ABORTED);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_OUT_OF_RANGE,
                 IREE_STATUS_OUT_OF_RANGE);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_UNIMPLEMENTED,
                 IREE_STATUS_UNIMPLEMENTED);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_INTERNAL,
                 IREE_STATUS_INTERNAL);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_UNAVAILABLE,
                 IREE_STATUS_UNAVAILABLE);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_DATA_LOSS,
                 IREE_STATUS_DATA_LOSS);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_UNAUTHENTICATED,
                 IREE_STATUS_UNAUTHENTICATED);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_DEFERRED,
                 IREE_STATUS_DEFERRED);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_STATUS_CODE_MASK,
                 IREE_STATUS_CODE_MASK);

STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_ALLOCATOR_COMMAND_MALLOC,
                 IREE_ALLOCATOR_COMMAND_MALLOC);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_ALLOCATOR_COMMAND_CALLOC,
                 IREE_ALLOCATOR_COMMAND_CALLOC);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_ALLOCATOR_COMMAND_REALLOC,
                 IREE_ALLOCATOR_COMMAND_REALLOC);
STATIC_ASSERT_EQ(IREE_HAL_EXECUTABLE_PLUGIN_ALLOCATOR_COMMAND_FREE,
                 IREE_ALLOCATOR_COMMAND_FREE);

STATIC_ASSERT_EQ(sizeof(iree_hal_executable_plugin_allocator_alloc_params_t),
                 sizeof(iree_allocator_alloc_params_t));
STATIC_ASSERT_EQ(sizeof(iree_hal_executable_plugin_allocator_t),
                 sizeof(iree_allocator_t));

STATIC_ASSERT_EQ(sizeof(iree_hal_executable_plugin_string_view_t),
                 sizeof(iree_string_view_t));
STATIC_ASSERT_EQ(sizeof(iree_hal_executable_plugin_string_pair_t),
                 sizeof(iree_string_pair_t));

#undef STATIC_ASSERT_EQ

//===----------------------------------------------------------------------===//
// iree_hal_executable_plugin_t
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_executable_plugin_load(
    const iree_hal_executable_plugin_header_t** header_ptr,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_executable_plugin_t* plugin) {
  // The header may be NULL if the plugin API used isn't compatible.
  if (!header_ptr) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "plugin does not support this version of the runtime (%08X)",
        IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST);
  }
  plugin->library.header = header_ptr;
  const iree_hal_executable_plugin_header_t* header = *plugin->library.header;

  plugin->identifier = iree_make_cstring_view(header->name);

// Ensure features declared by the plugin are available/allowed.
#if !IREE_STATUS_MODE
  if (iree_all_bits_set(header->features,
                        IREE_HAL_EXECUTABLE_PLUGIN_FEATURE_FULL_STATUS)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "plugin `%.*s` is compiled with the full iree_status_t but the runtime "
        "is not; IREE_STATUS_MODE must be > 0",
        (int)plugin->identifier.size, plugin->identifier.data);
  }
#endif  // !IREE_STATUS_MODE

  // Ensure that if the plugin is built for a particular sanitizer that we also
  // were compiled with that sanitizer enabled.
  switch (header->sanitizer) {
    case IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_NONE:
      // Always safe even if the host has a sanitizer enabled; it just means
      // that we won't be able to catch anything from within the plugin,
      // however checks outside will (often) still trigger when guard pages are
      // dirtied/etc.
      break;
#if defined(IREE_SANITIZER_ADDRESS)
    case IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_ADDRESS:
      // ASAN is compiled into the host and we can load this library.
      break;
#else
    case IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_ADDRESS:
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "plugin `%.*s` is compiled with ASAN support but the host "
          "runtime is not compiled with it enabled; add -fsanitize=address to "
          "the runtime compilation options",
          (int)plugin->identifier.size, plugin->identifier.data);
#endif  // IREE_SANITIZER_ADDRESS
#if defined(IREE_SANITIZER_THREAD)
    case IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_THREAD:
      // TSAN is compiled into the host and we can load this library.
      break;
#else
    case IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_THREAD:
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "plugin `%.*s` is compiled with TSAN support but the host "
          "runtime is not compiled with it enabled; add -fsanitize=thread to "
          "the runtime compilation options",
          (int)plugin->identifier.size, plugin->identifier.data);
#endif  // IREE_SANITIZER_THREAD
    default:
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "plugin `%.*s` requires a sanitizer the host runtime is not "
          "compiled to enable/understand: %u",
          (int)plugin->identifier.size, plugin->identifier.data,
          (uint32_t)header->sanitizer);
  }

  // Everything a standalone plugin needs should be packed in here.
  iree_hal_executable_plugin_environment_v0_t environment = {
      .host_allocator =
          {
              .self = host_allocator.self,
              .ctl = (iree_hal_executable_plugin_allocator_ctl_fn_t)
                         host_allocator.ctl,
          },
  };

  // Plugin is probably good - let's try loading it! It could fail for any
  // reason and the caller will clean up.
  return (iree_status_t)plugin->library.v0->load(
      &environment, (size_t)param_count,
      (const iree_hal_executable_plugin_string_pair_t*)params, &plugin->self);
}

iree_status_t iree_hal_executable_plugin_initialize(
    const void* vtable, iree_hal_executable_plugin_features_t required_features,
    const iree_hal_executable_plugin_header_t** header_ptr,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_hal_executable_plugin_resolve_thunk_t resolve_thunk,
    iree_allocator_t host_allocator,
    iree_hal_executable_plugin_t* out_base_plugin) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // NOTE: if we fail below the caller will release us and will need these to be
  // properly initialized.
  iree_atomic_ref_count_init(&out_base_plugin->ref_count);
  out_base_plugin->vtable = vtable;
  memset(&out_base_plugin->library, 0, sizeof(out_base_plugin->library));
  out_base_plugin->self = NULL;
  out_base_plugin->resolve_thunk = resolve_thunk;

  // Try to load the plugin; this may fail if the plugin is not supported
  // (version, features, etc) or the plugin decides it doesn't like Tuesdays.
  iree_status_t status = iree_hal_executable_plugin_load(
      header_ptr, param_count, params, host_allocator, out_base_plugin);
  if (iree_status_is_ok(status)) {
    IREE_TRACE({
      const iree_hal_executable_plugin_header_t* header =
          out_base_plugin->library.v0->header;
      IREE_TRACE_ZONE_APPEND_TEXT(z0, header->name);
      IREE_TRACE_ZONE_APPEND_TEXT(z0, header->description);
    });
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_executable_plugin_destroy(iree_hal_executable_plugin_t* plugin) {
  IREE_ASSERT_ARGUMENT(plugin);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, plugin->identifier.data,
                              plugin->identifier.size);

  // Unload the plugin, if it has an unload method.
  if (plugin->library.v0 && plugin->library.v0->unload) {
    plugin->library.v0->unload(plugin->self);
  }
  memset(&plugin->library, 0, sizeof(plugin->library));
  plugin->self = NULL;

  plugin->vtable->destroy(plugin);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_executable_plugin_retain(iree_hal_executable_plugin_t* plugin) {
  if (IREE_LIKELY(plugin)) {
    iree_atomic_ref_count_inc(&plugin->ref_count);
  }
}

void iree_hal_executable_plugin_release(iree_hal_executable_plugin_t* plugin) {
  if (IREE_LIKELY(plugin) &&
      iree_atomic_ref_count_dec(&plugin->ref_count) == 1) {
    iree_hal_executable_plugin_destroy(plugin);
  }
}

// NOTE: must match iree_hal_executable_import_provider_t.resolve.
static iree_status_t iree_hal_executable_plugin_resolve(
    void* self, iree_host_size_t count, const char* const* symbol_names,
    void** out_fn_ptrs, void** out_fn_contexts,
    iree_hal_executable_import_resolution_t* out_resolution) {
  IREE_ASSERT_ARGUMENT(self);
  IREE_ASSERT_ARGUMENT(out_resolution);
  *out_resolution = 0;
  if (!count) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_executable_plugin_t* plugin = (iree_hal_executable_plugin_t*)self;
  IREE_TRACE_ZONE_APPEND_TEXT(z0, plugin->identifier.data,
                              plugin->identifier.size);

  const iree_hal_executable_plugin_resolve_params_v0_t params = {
      .count = (size_t)count,
      .symbol_names = symbol_names,
      .out_fn_ptrs = out_fn_ptrs,
      .out_fn_contexts = out_fn_contexts,
  };
  iree_hal_executable_plugin_resolution_t resolution = 0;
  iree_status_t status =
      plugin->resolve_thunk
          ? plugin->resolve_thunk(plugin->library.v0->resolve, plugin->self,
                                  &params, &resolution)
          : (iree_status_t)plugin->library.v0->resolve(plugin->self, &params,
                                                       &resolution);
  *out_resolution = (iree_hal_executable_import_resolution_t)resolution;

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_hal_executable_import_provider_t iree_hal_executable_plugin_provider(
    iree_hal_executable_plugin_t* plugin) {
  IREE_ASSERT_ARGUMENT(plugin);
  return (iree_hal_executable_import_provider_t){
      .self = plugin,
      .resolve = iree_hal_executable_plugin_resolve,
  };
}

//===----------------------------------------------------------------------===//
// Default registration hook
//===----------------------------------------------------------------------===//

#if defined(IREE_HAL_EXECUTABLE_PLUGIN_REGISTRATION_FN)

// Defined by the user and linked in to the binary:
extern iree_status_t IREE_HAL_EXECUTABLE_PLUGIN_REGISTRATION_FN(
    iree_hal_executable_plugin_manager_t* manager,
    iree_allocator_t host_allocator);

static iree_status_t iree_hal_executable_plugin_manager_register_external(
    iree_hal_executable_plugin_manager_t* manager,
    iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      IREE_HAL_EXECUTABLE_PLUGIN_REGISTRATION_FN(manager, host_allocator);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

#else

static iree_status_t iree_hal_executable_plugin_manager_register_external(
    iree_hal_executable_plugin_manager_t* manager,
    iree_allocator_t host_allocator) {
  return iree_ok_status();
}

#endif  // IREE_HAL_EXECUTABLE_PLUGIN_REGISTRATION_FN

//===----------------------------------------------------------------------===//
// iree_hal_executable_plugin_manager_t
//===----------------------------------------------------------------------===//

struct iree_hal_executable_plugin_manager_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;

  // Total number of available provider and plugin slots.
  int32_t capacity;

  // Guards plugin/provider list mutation.
  // We always hold a mutex around mutation but still allow provider reads. This
  // is safe because we only ever append to the provider list and if a thread
  // comes in and reads an old count while a new provider is being added it will
  // just miss the provider being added. We go through this trouble because we
  // can't hold a manager lock during resolution as if the import plugins are
  // expensive (JITing code, etc) they'd block the whole process.
  iree_slim_mutex_t mutex;

  // List of registered plugins. This may be a subset of the providers as not
  // all providers need a plugin backing them. We only use this to keep the
  // plugins live for the lifetime of the manager.
  int32_t plugin_count;
  iree_hal_executable_plugin_t** plugins;

  // List of available providers used during resolution.
  // Providers are scanned in reverse registration order and include both
  // unowned/external providers and ones backed by plugins owned by the
  // manager. Note that threads may read this while the provider list is being
  // appended.
  iree_atomic_int32_t provider_count;
  iree_hal_executable_import_provider_t providers[];
};

iree_status_t iree_hal_executable_plugin_manager_create(
    iree_host_size_t capacity, iree_allocator_t host_allocator,
    iree_hal_executable_plugin_manager_t** out_manager) {
  IREE_ASSERT_ARGUMENT(out_manager);
  *out_manager = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_executable_plugin_manager_t* manager = NULL;
  iree_host_size_t plugins_offset = iree_host_align(
      sizeof(*manager) + capacity * sizeof(manager->providers[0]),
      iree_max_align_t);
  iree_host_size_t total_size =
      plugins_offset + capacity * sizeof(*manager->plugins);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&manager));
  iree_atomic_ref_count_init(&manager->ref_count);
  manager->host_allocator = host_allocator;
  manager->capacity = capacity;
  iree_slim_mutex_initialize(&manager->mutex);
  manager->plugin_count = 0;
  manager->plugins =
      (iree_hal_executable_plugin_t**)((uintptr_t)manager + plugins_offset);

  // Register any externally-defined plugins by default. Dynamically registered
  // plugins can override these if they are registered later on.
  iree_status_t status = iree_hal_executable_plugin_manager_register_external(
      manager, host_allocator);

  if (iree_status_is_ok(status)) {
    *out_manager = manager;
  } else {
    iree_hal_executable_plugin_manager_release(manager);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_executable_plugin_manager_destroy(
    iree_hal_executable_plugin_manager_t* manager) {
  iree_allocator_t host_allocator = manager->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release in reverse registration order; likely not required but makes for
  // easier debugging of stateful providers (JITs and such that may have lots of
  // memory/etc we want to track).
  for (int32_t i = manager->plugin_count - 1; i >= 0; --i) {
    iree_hal_executable_plugin_release(manager->plugins[i]);
  }

  iree_slim_mutex_deinitialize(&manager->mutex);
  iree_allocator_free(host_allocator, manager);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_executable_plugin_manager_retain(
    iree_hal_executable_plugin_manager_t* manager) {
  if (IREE_LIKELY(manager)) {
    iree_atomic_ref_count_inc(&manager->ref_count);
  }
}

void iree_hal_executable_plugin_manager_release(
    iree_hal_executable_plugin_manager_t* manager) {
  if (IREE_LIKELY(manager) &&
      iree_atomic_ref_count_dec(&manager->ref_count) == 1) {
    iree_hal_executable_plugin_manager_destroy(manager);
  }
}

// Registers a |provider| and optional |plugin| as an atomic operation.
static iree_status_t iree_hal_executable_plugin_manager_register(
    iree_hal_executable_plugin_manager_t* manager,
    iree_hal_executable_import_provider_t provider,
    iree_hal_executable_plugin_t* plugin) {
  IREE_ASSERT_ARGUMENT(manager);
  if (provider.resolve == NULL) {
    // No-op provider; may happen on accident.
    return iree_ok_status();
  }

  // Hold the mutex to block other writers - readers are fine, though, as
  // they'll just get stale data.
  iree_slim_mutex_lock(&manager->mutex);

  // Get the next provider slot. Note that we don't yet increment it as we need
  // to put the provider in there first.
  int32_t slot = iree_atomic_load_int32(&manager->provider_count,
                                        iree_memory_order_acquire);
  if (slot >= manager->capacity) {
    iree_slim_mutex_unlock(&manager->mutex);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "import manager capacity of %d reached",
                            manager->capacity);
  }

  // Stash the provider and the plugin, if any.
  manager->providers[slot] = provider;
  if (plugin) {
    iree_hal_executable_plugin_retain(plugin);
    manager->plugins[manager->plugin_count++] = plugin;
  }

  // Mark the slot as valid now that the provider is in it.
  iree_atomic_fetch_add_int32(&manager->provider_count, 1,
                              iree_memory_order_release);

  iree_slim_mutex_unlock(&manager->mutex);
  return iree_ok_status();
}

iree_status_t iree_hal_executable_plugin_manager_register_provider(
    iree_hal_executable_plugin_manager_t* manager,
    iree_hal_executable_import_provider_t provider) {
  return iree_hal_executable_plugin_manager_register(manager, provider, NULL);
}

iree_status_t iree_hal_executable_plugin_manager_register_plugin(
    iree_hal_executable_plugin_manager_t* manager,
    iree_hal_executable_plugin_t* plugin) {
  IREE_ASSERT_ARGUMENT(plugin);
  return iree_hal_executable_plugin_manager_register(
      manager, iree_hal_executable_plugin_provider(plugin), plugin);
}

// Resolves |count| imports given |symbol_names| and stores pointers to their
// implementation in |out_fn_ptrs| and optional contexts in |out_fn_contexts|.
//
// A symbol name starting with `?` indicates that the symbol is optional and is
// allowed to be resolved to NULL. Such cases will always return OK but set the
// IREE_HAL_EXECUTABLE_IMPORT_RESOLUTION_MISSING_OPTIONAL resolution bit.
//
// Any already resolved function pointers will be skipped and left unmodified.
// When there's only partial availability of required imports any available
// ones will still be populated and NOT_FOUND will is returned. This allows for
// looping over multiple providers to populate what they can and only fails out
// if all providers return NOT_FOUND for a required import.
//
// Symbol names must be sorted alphabetically so if we cared we could use this
// information to more efficiently resolve the symbols from providers (O(n)
// walk vs potential O(nlogn)/O(n^2)).
//
// NOTE: this matches the iree_hal_executable_import_provider_t.resolve
// function signature so that it can be directly used as a provider.
static iree_status_t iree_hal_executable_plugin_manager_resolve(
    iree_hal_executable_plugin_manager_t* manager, iree_host_size_t count,
    const char* const* symbol_names, void** out_fn_ptrs, void** out_fn_contexts,
    iree_hal_executable_import_resolution_t* out_resolution) {
  if (!count) return iree_ok_status();
  IREE_ASSERT_ARGUMENT(manager);
  IREE_ASSERT_ARGUMENT(out_fn_ptrs);
  IREE_ASSERT_ARGUMENT(out_fn_contexts);
  if (out_resolution) *out_resolution = 0;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, count);

  // Fetch the valid provider count.
  // This may end up missing providers that get registered during/after we scan
  // but that's ok: multithreaded registration/resolution is non-deterministic
  // by nature. Not holding the lock here means we allow multiple threads to
  // resolve imports at the same time.
  int32_t provider_count = iree_atomic_load_int32(&manager->provider_count,
                                                  iree_memory_order_acquire);

  // Scan in reverse registration order so that more recently registered
  // providers get queried first. try_resolve will populate any function
  // pointers it can and ignore already populated ones (or be mean and overwrite
  // them, but please don't!). After resolving if a provider can't resolve a
  // symbol it will return NOT_FOUND but we only really care on the final
  // provider in the scan.
  bool all_required_resolved = false;
  iree_hal_executable_import_resolution_t resolution = 0;
  for (int32_t i = provider_count - 1; i >= 0; --i) {
    iree_hal_executable_import_provider_t provider = manager->providers[i];
    IREE_ASSERT(provider.resolve);  // checked on registration
    iree_status_t provider_status =
        provider.resolve(provider.self, count, symbol_names, out_fn_ptrs,
                         out_fn_contexts, &resolution);
    if (iree_status_is_ok(provider_status)) {
      // Found all required but may be missing some optional imports. If so
      // we'll need to continue scanning.
      all_required_resolved = true;
      if (iree_all_bits_set(
              resolution,
              IREE_HAL_EXECUTABLE_IMPORT_RESOLUTION_MISSING_OPTIONAL)) {
        continue;
      }
      // All required + optional found, end the scan early.
      break;
    } else if (iree_status_is_not_found(provider_status)) {
      // One or more required symbols not found, keep scanning.
      iree_status_ignore(provider_status);
      continue;
    } else {
      // Other failure we need to propagate (may be JIT issues or something).
      return provider_status;
    }
  }

  // If any required imports are missing we'll fail now with a listing.
  // Note that some optional imports may not be resolved (check |resolution|).
  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(!all_required_resolved)) {
#if IREE_STATUS_MODE
    iree_host_size_t missing_count = 0;
    iree_string_builder_t builder;
    iree_string_builder_initialize(manager->host_allocator, &builder);
    for (iree_host_size_t i = 0; i < count; ++i) {
      if (out_fn_ptrs[i] != NULL) continue;
      if (iree_hal_executable_import_is_optional(symbol_names[i])) continue;
      if (missing_count > 0) {
        IREE_IGNORE_ERROR(
            iree_string_builder_append_string(&builder, IREE_SV(", ")));
      }
      IREE_IGNORE_ERROR(
          iree_string_builder_append_cstring(&builder, symbol_names[i]));
      ++missing_count;
    }
    status =
        iree_make_status(IREE_STATUS_NOT_FOUND,
                         "missing %zu required executable imports: [%.*s]",
                         missing_count, (int)iree_string_builder_size(&builder),
                         iree_string_builder_buffer(&builder));
    iree_string_builder_deinitialize(&builder);
#else
    status = iree_status_from_code(IREE_STATUS_NOT_FOUND);
#endif  // IREE_STATUS_MODE
  }

  if (out_resolution) *out_resolution = resolution;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_hal_executable_import_provider_t
iree_hal_executable_plugin_manager_provider(
    iree_hal_executable_plugin_manager_t* manager) {
  return (iree_hal_executable_import_provider_t){
      .self = manager,
      .resolve = manager ? (iree_hal_executable_import_provider_resolve_fn_t)
                               iree_hal_executable_plugin_manager_resolve
                         : NULL,
  };
}
