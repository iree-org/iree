// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdbool.h>
#include <string.h>

#include "experimental/streaming/internal.h"
#include "iree/base/internal/call_once.h"
#include "iree/base/internal/math.h"
#include "iree/base/internal/synchronization.h"

static void iree_hal_streaming_context_symbol_map_expunge_module(
    iree_hal_streaming_context_symbol_map_t* map,
    iree_hal_streaming_module_registration_t* registration);

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

#define IREE_HAL_STREAMING_SYMBOL_REGISTRY_DEFAULT_CAPACITY 64
#define IREE_HAL_STREAMING_SYMBOL_MAP_DEFAULT_CAPACITY 16

// Indicates an empty entry (implicitly terminating a chain).
#define IREE_HAL_STREAMING_SYMBOL_MAP_EMPTY_KEY ((void*)0)
// Indicates a deleted entry (linear scan must proceed).
#define IREE_HAL_STREAMING_SYMBOL_MAP_TOMBSTONE_KEY ((void*)1)

static inline bool iree_hal_streaming_symbol_map_is_valid_key(void* key) {
  return key != IREE_HAL_STREAMING_SYMBOL_MAP_EMPTY_KEY &&
         key != IREE_HAL_STREAMING_SYMBOL_MAP_TOMBSTONE_KEY;
}

// Hash function for host pointers.
// DO NOT SUBMIT evaluate/document source
static inline uint64_t iree_hal_streaming_symbol_pointer_hash(void* ptr) {
  // Simple hash for pointers - mix bits.
  uint64_t hash = (uint64_t)ptr;
  hash ^= hash >> 33;
  hash *= 0xff51afd7ed558ccdull;
  hash ^= hash >> 33;
  hash *= 0xc4ceb9fe1a85ec53ull;
  hash ^= hash >> 33;
  return hash;
}

//===----------------------------------------------------------------------===//
// Global Symbol Registry
//===----------------------------------------------------------------------===//

// Static global registry instance, lazily initialized.
static iree_hal_streaming_global_symbol_registry_t*
    iree_hal_streaming_global_symbol_registry_ptr = NULL;

// One-time initialization function for the global registry.
static void iree_hal_streaming_initialize_global_registry(void) {
  iree_status_t status = iree_hal_streaming_global_symbol_registry_allocate(
      iree_allocator_system(), &iree_hal_streaming_global_symbol_registry_ptr);
  if (!iree_status_is_ok(status)) {
    // Log error but continue - registry will be NULL.
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }
}

iree_hal_streaming_global_symbol_registry_t*
iree_hal_streaming_global_symbol_registry(void) {
  static iree_once_flag once = IREE_ONCE_FLAG_INIT;
  iree_call_once(&once, iree_hal_streaming_initialize_global_registry);
  IREE_ASSERT(iree_hal_streaming_global_symbol_registry_ptr);
  return iree_hal_streaming_global_symbol_registry_ptr;
}

iree_status_t iree_hal_streaming_global_symbol_registry_allocate(
    iree_allocator_t host_allocator,
    iree_hal_streaming_global_symbol_registry_t** out_registry) {
  IREE_ASSERT_ARGUMENT(out_registry);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_registry = NULL;

  // Allocate registry.
  iree_hal_streaming_global_symbol_registry_t* registry = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*registry),
                                (void**)&registry));
  registry->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&registry->mutex);

  // Allocate initial module pointer array.
  registry->module_capacity = 16;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, registry->module_capacity * sizeof(registry->modules[0]),
      (void**)&registry->modules);

  if (iree_status_is_ok(status)) {
    *out_registry = registry;
  } else {
    iree_hal_streaming_global_symbol_registry_free(registry);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_streaming_global_symbol_registry_free(
    iree_hal_streaming_global_symbol_registry_t* registry) {
  if (!registry) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = registry->host_allocator;

  // Free module registrations and their symbols.
  for (iree_host_size_t i = 0; i < registry->module_count; ++i) {
    if (registry->modules[i]) {
      iree_allocator_free(host_allocator, registry->modules[i]->symbols);
      iree_allocator_free(host_allocator, registry->modules[i]);
    }
  }
  iree_allocator_free(host_allocator, registry->modules);

  iree_slim_mutex_deinitialize(&registry->mutex);
  iree_allocator_free(host_allocator, registry);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_streaming_global_symbol_registry_grow_unsafe(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_host_size_t new_capacity) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_module_registration_t** new_modules = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(registry->host_allocator,
                                new_capacity * sizeof(new_modules[0]),
                                (void**)&new_modules));
  memcpy(new_modules, registry->modules,
         registry->module_count * sizeof(new_modules[0]));
  iree_allocator_free(registry->host_allocator, registry->modules);

  registry->modules = new_modules;
  registry->module_capacity = new_capacity;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_global_symbol_registry_register_module(
    iree_hal_streaming_global_symbol_registry_t* registry,
    const void* module_binary,
    iree_hal_streaming_module_registration_t** out_module) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(out_module);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_module = NULL;

  iree_slim_mutex_lock(&registry->mutex);

  // Grow the module table if needed.
  iree_status_t status = iree_ok_status();
  if (registry->module_count >= registry->module_capacity) {
    const iree_host_size_t new_capacity =
        iree_max(registry->module_capacity + 1, registry->module_capacity * 2);
    status = iree_hal_streaming_global_symbol_registry_grow_unsafe(
        registry, new_capacity);
  }

  // Allocate a new module registration dynamically.
  iree_hal_streaming_module_registration_t* module = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(registry->host_allocator, sizeof(*module),
                                   (void**)&module);
  }

  if (iree_status_is_ok(status)) {
    module->module_binary = module_binary;

    // Allocate initial symbols array.
    module->symbol_capacity = 32;
    status = iree_allocator_malloc(
        registry->host_allocator,
        module->symbol_capacity * sizeof(module->symbols[0]),
        (void**)&module->symbols);
  }

  if (iree_status_is_ok(status)) {
    registry->modules[registry->module_count++] = module;
    *out_module = module;
  } else {
    if (module) {
      iree_allocator_free(registry->host_allocator, module->symbols);
      iree_allocator_free(registry->host_allocator, module);
    }
  }

  iree_slim_mutex_unlock(&registry->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_global_symbol_registry_unregister_module(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_hal_streaming_module_registration_t* module) {
  IREE_ASSERT_ARGUMENT(registry);
  if (!module) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&registry->mutex);

  // Find the module in the list.
  iree_host_size_t module_index = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < registry->module_count; ++i) {
    if (registry->modules[i] == module) {
      module_index = i;
      break;
    }
  }
  if (module_index == IREE_HOST_SIZE_MAX) {
    iree_slim_mutex_unlock(&registry->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_NOT_FOUND, "module not registered");
  }

  // Notify all context maps to remove symbols from this module.
  iree_hal_streaming_context_symbol_map_t* context_map =
      registry->context_maps_head;
  while (context_map) {
    iree_hal_streaming_context_symbol_map_expunge_module(context_map, module);
    context_map = context_map->next;
  }

  // Free the module registration.
  iree_allocator_free(registry->host_allocator, module->symbols);
  iree_allocator_free(registry->host_allocator, module);

  // Remove the module pointer by shifting remaining pointers in the list.
  if (module_index < registry->module_count - 1) {
    memmove(&registry->modules[module_index],
            &registry->modules[module_index + 1],
            (registry->module_count - module_index - 1) *
                sizeof(registry->modules[0]));
  }
  --registry->module_count;

  iree_slim_mutex_unlock(&registry->mutex);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_module_registration_grow_unsafe(
    iree_hal_streaming_module_registration_t* module,
    iree_host_size_t new_capacity, iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_symbol_registration_t* new_symbols = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator,
                                new_capacity * sizeof(new_symbols[0]),
                                (void**)&new_symbols));
  memcpy(new_symbols, module->symbols,
         module->symbol_count * sizeof(module->symbols[0]));
  iree_allocator_free(host_allocator, module->symbols);
  module->symbols = new_symbols;
  module->symbol_capacity = new_capacity;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_global_symbol_registry_insert_function(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_hal_streaming_module_registration_t* module, void* host_function,
    const char* device_name, uint32_t thread_limit,
    uint32_t shared_size_bytes) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(host_function);
  IREE_ASSERT_ARGUMENT(device_name);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&registry->mutex);

  // Check if we need to grow the module's symbols array.
  iree_status_t status = iree_ok_status();
  if (module->symbol_count >= module->symbol_capacity) {
    const iree_host_size_t new_capacity =
        iree_max(module->symbol_capacity + 1, module->symbol_capacity * 2);
    status = iree_hal_streaming_module_registration_grow_unsafe(
        module, new_capacity, registry->host_allocator);
  }

  if (iree_status_is_ok(status)) {
    // Add symbol to module's symbols array.
    iree_hal_streaming_symbol_registration_t* symbol =
        &module->symbols[module->symbol_count++];

    // Fill in registration.
    symbol->host_pointer = host_function;
    symbol->type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
    symbol->device_name = device_name;  // direct pointer, no copy
    symbol->module = module;
    symbol->params.function.thread_limit = thread_limit;
    symbol->params.function.shared_size_bytes = shared_size_bytes;
  }

  iree_slim_mutex_unlock(&registry->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_global_symbol_registry_insert_variable(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_hal_streaming_module_registration_t* module, void* host_variable,
    const char* device_name, size_t size, uint32_t alignment) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(host_variable);
  IREE_ASSERT_ARGUMENT(device_name);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&registry->mutex);

  // Check if we need to grow the module's symbols array.
  iree_status_t status = iree_ok_status();
  if (module->symbol_count >= module->symbol_capacity) {
    const iree_host_size_t new_capacity =
        iree_max(module->symbol_capacity + 1, module->symbol_capacity * 2);
    status = iree_hal_streaming_module_registration_grow_unsafe(
        module, new_capacity, registry->host_allocator);
  }

  if (iree_status_is_ok(status)) {
    // Add symbol to module's symbols array.
    iree_hal_streaming_symbol_registration_t* symbol =
        &module->symbols[module->symbol_count++];

    // Fill in registration (device_name points directly to fat binary string).
    symbol->host_pointer = host_variable;
    symbol->type = IREE_HAL_STREAMING_SYMBOL_TYPE_GLOBAL;
    symbol->device_name = device_name;  // direct pointer, no copy
    symbol->module = module;
    symbol->params.variable.size = size;
    symbol->params.variable.alignment = alignment;
  }

  iree_slim_mutex_unlock(&registry->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Slowly looks up a registration by host pointer.
// We assume that this only ever happens on a context map miss and that we have
// very few of those after startup.
// Returns NULL if not found.
// Thread-safe (takes lock internally).
static const iree_hal_streaming_symbol_registration_t*
iree_hal_streaming_global_symbol_registry_lookup(
    iree_hal_streaming_global_symbol_registry_t* registry, void* host_pointer) {
  if (!registry || !host_pointer) return NULL;

  iree_slim_mutex_lock(&registry->mutex);

  // Linear scan through all modules and their symbols.
  const iree_hal_streaming_symbol_registration_t* result = NULL;
  for (iree_host_size_t i = 0; i < registry->module_count; ++i) {
    iree_hal_streaming_module_registration_t* module = registry->modules[i];
    if (!module) continue;
    for (iree_host_size_t j = 0; j < module->symbol_count; ++j) {
      if (module->symbols[j].host_pointer == host_pointer) {
        result = &module->symbols[j];
        break;
      }
    }
    if (result) break;
  }

  iree_slim_mutex_unlock(&registry->mutex);
  return result;
}

//===----------------------------------------------------------------------===//
// Context Symbol Map
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_context_symbol_map_initialize(
    iree_hal_streaming_context_t* context, iree_host_size_t initial_capacity,
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_allocator_t host_allocator,
    iree_hal_streaming_context_symbol_map_t* out_map) {
  IREE_ASSERT_ARGUMENT(out_map);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_map, 0, sizeof(*out_map));

  out_map->context = context;
  out_map->host_allocator = host_allocator;

  // Start with a small capacity.
  if (initial_capacity == 0) {
    initial_capacity = IREE_HAL_STREAMING_SYMBOL_MAP_DEFAULT_CAPACITY;
  }

  // Round capacity up to the next power of 2 (if not already).
  out_map->capacity =
      (iree_host_size_t)iree_math_round_up_to_pow2_u64(initial_capacity);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator,
                                out_map->capacity * sizeof(out_map->entries[0]),
                                (void**)&out_map->entries));

  // Register with the global registry (so we can listen for notifications).
  iree_slim_mutex_lock(&registry->mutex);
  out_map->registry = registry;
  out_map->next = registry->context_maps_head;
  if (registry->context_maps_head) {
    registry->context_maps_head->prev = out_map;
  }
  registry->context_maps_head = out_map;
  iree_slim_mutex_unlock(&registry->mutex);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_streaming_context_symbol_map_deinitialize(
    iree_hal_streaming_context_symbol_map_t* map) {
  if (!map || !map->entries) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = map->host_allocator;
  iree_hal_streaming_global_symbol_registry_t* registry = map->registry;

  // Release all loaded modules.
  iree_hal_streaming_context_module_entry_t* module_entry = map->modules;
  while (module_entry) {
    iree_hal_streaming_context_module_entry_t* next = module_entry->next;
    if (module_entry->module) {
      iree_hal_streaming_module_release(module_entry->module);
    }
    iree_allocator_free(host_allocator, module_entry);
    module_entry = next;
  }

  // Unregister from global registry, if registered.
  if (registry) {
    iree_slim_mutex_lock(&registry->mutex);
    if (map->prev) {
      map->prev->next = map->next;
    } else if (registry->context_maps_head == map) {
      registry->context_maps_head = map->next;
    }
    if (map->next) {
      map->next->prev = map->prev;
    }
    iree_slim_mutex_unlock(&registry->mutex);
  }

  iree_allocator_free(host_allocator, map->entries);

  IREE_TRACE_ZONE_END(z0);
}

// Grows the hash table to accommodate at least the specified capacity.
// Rehashes all existing entries into the new table.
static iree_status_t iree_hal_streaming_context_symbol_map_grow(
    iree_hal_streaming_context_symbol_map_t* map,
    iree_host_size_t new_min_capacity) {
  IREE_ASSERT_ARGUMENT(map);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Round up to the next power of 2 for optimal hash distribution.
  iree_host_size_t new_capacity =
      (iree_host_size_t)iree_math_round_up_to_pow2_u64(new_min_capacity);
  if (new_capacity <= map->capacity) {
    new_capacity = map->capacity * 2;
  }

  // Allocate the new table.
  iree_hal_streaming_context_symbol_entry_t* new_entries = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(map->host_allocator,
                                new_capacity * sizeof(new_entries[0]),
                                (void**)&new_entries));
  memset(new_entries, 0, new_capacity * sizeof(new_entries[0]));

  // Rehash all existing entries into the new table.
  for (iree_host_size_t i = 0; i < map->capacity; ++i) {
    void* key = map->entries[i].key;
    if (!iree_hal_streaming_symbol_map_is_valid_key(key)) {
      continue;  // Skip empty and tombstone entries.
    }

    // Find a slot in the new table.
    const uint64_t hash = iree_hal_streaming_symbol_pointer_hash(key);
    uint32_t index = hash & (new_capacity - 1);
    for (iree_host_size_t j = 0; j < new_capacity; ++j) {
      if (new_entries[index].key == IREE_HAL_STREAMING_SYMBOL_MAP_EMPTY_KEY) {
        new_entries[index].key = key;
        new_entries[index].symbol = map->entries[i].symbol;
        break;
      }
      index = (index + 1) & (new_capacity - 1);
    }
  }

  // Free the old table and swap with the new one.
  iree_allocator_free(map->host_allocator, map->entries);
  map->entries = new_entries;
  map->capacity = new_capacity;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Loads a module and populates the context symbol map with its symbols.
// This is called when a symbol is requested for use in a context but the module
// hasn't been instantiated in it yet.
static iree_status_t iree_hal_streaming_context_symbol_map_prepare_module(
    iree_hal_streaming_context_symbol_map_t* map,
    iree_hal_streaming_module_registration_t* registration) {
  IREE_ASSERT_ARGUMENT(map);
  IREE_ASSERT_ARGUMENT(registration);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check if we've already loaded this module.
  for (iree_hal_streaming_context_module_entry_t* module_entry = map->modules;
       module_entry != NULL; module_entry = module_entry->next) {
    if (module_entry->registration == registration) {
      // Module already loaded - no-op.
      // We could instead try to see if there are any new registered symbols and
      // process those (keeping track of a symbol count we last prepared with).
      // It's not really expected that static initializers will re-register
      // binaries, though.
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();
    }
  }

  // Grow the hash table to fit our new total count, if needed.
  const iree_host_size_t new_count = map->count + registration->symbol_count;
  if (new_count > map->capacity * 3 / 4) {
    // Need to resize the hash table to maintain good load factor.
    // We want to keep the load factor below 75% for good performance.
    iree_host_size_t new_min_capacity = (new_count * 4) / 3;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_context_symbol_map_grow(map, new_min_capacity));
  }

  // Allocate tracking entry.
  iree_hal_streaming_context_module_entry_t* entry = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(map->host_allocator, sizeof(*entry),
                                (void**)&entry));
  entry->registration = registration;
  entry->module = NULL;

  // Module not loaded yet - load it now.
  // This requires creating the module from the fat binary.
  iree_hal_executable_caching_mode_t caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA |
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION;
  iree_const_byte_span_t module_data =
      iree_make_const_byte_span((const uint8_t*)registration->module_binary,
                                /*infer*/ 0);
  iree_hal_streaming_module_t* module = NULL;
  iree_status_t status = iree_hal_streaming_module_create_from_memory(
      map->context, caching_mode, module_data, map->host_allocator,
      &entry->module);
  if (iree_status_is_ok(status)) {
    // Insert all symbols from the module into the hash table.
    for (iree_host_size_t i = 0; i < registration->symbol_count; ++i) {
      iree_string_view_t symbol_name = module->symbols[i].name;
      void* symbol_host_ptr = registration->symbols[i].host_pointer;
      // Find the corresponding compiled symbol in the module.
      iree_hal_streaming_symbol_t* symbol = NULL;
      for (iree_host_size_t j = 0; j < module->symbol_count; ++j) {
        if (iree_string_view_equal(symbol_name, module->symbols[j].name)) {
          symbol = &module->symbols[j];
          break;
        }
      }
      if (!symbol) {
        // Registered symbol not found.
        status = iree_make_status(
            IREE_STATUS_NOT_FOUND,
            "registered symbol `%.*s` not found in loaded module",
            (int)symbol_name.size, symbol_name.data);
        break;
      }

      // Insert into hash table.
      const uint64_t hash =
          iree_hal_streaming_symbol_pointer_hash(symbol_host_ptr);
      uint32_t idx = hash & (map->capacity - 1);
      for (iree_host_size_t j = 0; j < map->capacity; ++j) {
        if (map->entries[idx].key == IREE_HAL_STREAMING_SYMBOL_MAP_EMPTY_KEY ||
            map->entries[idx].key ==
                IREE_HAL_STREAMING_SYMBOL_MAP_TOMBSTONE_KEY) {
          map->entries[idx].key = symbol_host_ptr;
          map->entries[idx].symbol = symbol;
          ++map->count;
          break;
        }
        idx = (idx + 1) & (map->capacity - 1);
      }
    }
  }

  if (iree_status_is_ok(status)) {
    // Link module into listing.
    entry->next = map->modules;
    map->modules = entry;
  } else {
    iree_hal_streaming_module_release(entry->module);
    iree_allocator_free(map->host_allocator, entry);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Removes all symbols from a module and unloads it from the context map.
// Called when a module is unregistered from the global registry.
// No-op if the module was never registered.
static void iree_hal_streaming_context_symbol_map_expunge_module(
    iree_hal_streaming_context_symbol_map_t* map,
    iree_hal_streaming_module_registration_t* registration) {
  IREE_ASSERT_ARGUMENT(map);
  IREE_ASSERT_ARGUMENT(registration);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Find the module in the loaded modules list and unlink it.
  iree_hal_streaming_context_module_entry_t* prev_entry = NULL;
  iree_hal_streaming_context_module_entry_t* module_entry = map->modules;
  while (module_entry) {
    if (module_entry->registration == registration) {
      // Found the module - remove it from the list.
      if (prev_entry) {
        prev_entry->next = module_entry->next;
      } else {
        map->modules = module_entry->next;
      }
      break;
    }
    prev_entry = module_entry;
    module_entry = module_entry->next;
  }
  if (!module_entry) {
    // Module was not loaded in this context, no-op.
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Walk all symbols in the module registration and remove them from the hash
  // table.
  for (iree_host_size_t i = 0; i < registration->symbol_count; ++i) {
    void* host_pointer = registration->symbols[i].host_pointer;
    const uint64_t hash = iree_hal_streaming_symbol_pointer_hash(host_pointer);
    uint32_t index = hash & (map->capacity - 1);
    for (iree_host_size_t j = 0; j < map->capacity; ++j) {
      if (map->entries[index].key == host_pointer) {
        // Found it - replace with tombstone.
        map->entries[index].key = IREE_HAL_STREAMING_SYMBOL_MAP_TOMBSTONE_KEY;
        map->entries[index].symbol = NULL;
        --map->count;
        break;
      } else if (map->entries[index].key ==
                 IREE_HAL_STREAMING_SYMBOL_MAP_EMPTY_KEY) {
        break;  // not in the table
      }
      index = (index + 1) & (map->capacity - 1);  // continue linear probe
    }
  }

  // Release the module and free the entry.
  iree_hal_streaming_module_release(module_entry->module);
  iree_allocator_free(map->host_allocator, module_entry);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_streaming_context_symbol_map_lookup(
    iree_hal_streaming_context_symbol_map_t* map, void* host_pointer,
    iree_hal_streaming_symbol_t** out_symbol) {
  IREE_ASSERT_ARGUMENT(map);
  IREE_ASSERT_ARGUMENT(out_symbol);

  // Check for invalid keys.
  if (!host_pointer ||
      !iree_hal_streaming_symbol_map_is_valid_key(host_pointer)) {
    *out_symbol = NULL;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid host pointer");
  }

  // Fast path: check context-local map (no locks).
  const uint64_t hash = iree_hal_streaming_symbol_pointer_hash(host_pointer);
  uint32_t index = hash & (map->capacity - 1);
  for (iree_host_size_t i = 0; i < map->capacity; ++i) {
    const void* entry_key = map->entries[index].key;
    if (entry_key == host_pointer) {
      *out_symbol = map->entries[index].symbol;  // hit
      return iree_ok_status();
    } else if (entry_key == IREE_HAL_STREAMING_SYMBOL_MAP_EMPTY_KEY) {
      break;  // not found in local map
    }
    index = (index + 1) & (map->capacity - 1);  // continue linear probe
  }

  // Slow path: check global registry.
  const iree_hal_streaming_symbol_registration_t* registration =
      iree_hal_streaming_global_symbol_registry_lookup(map->registry,
                                                       host_pointer);
  if (!registration) {
    // Not found - return identity.
    *out_symbol = (iree_hal_streaming_symbol_t*)host_pointer;
    return iree_ok_status();
  }

  // Ensure the module is loaded and its symbols are in the hash table.
  IREE_RETURN_IF_ERROR(iree_hal_streaming_context_symbol_map_prepare_module(
                           map, registration->module),
                       "preparing statically registered module for context");

  // Now look up again in the hash table using the original hash.
  // The symbol should be there now after module preparation.
  for (iree_host_size_t i = 0; i < map->capacity; ++i) {
    const void* entry_key = map->entries[index].key;
    if (entry_key == host_pointer) {
      *out_symbol = map->entries[index].symbol;  // hit
      return iree_ok_status();
    } else if (entry_key == IREE_HAL_STREAMING_SYMBOL_MAP_EMPTY_KEY) {
      break;  // still not found (shouldn't happen)
    }
    index = (index + 1) & (map->capacity - 1);  // continue linear probe
  }

  // Symbol not found even after loading module (shouldn't happen).
  *out_symbol = (iree_hal_streaming_symbol_t*)host_pointer;
  return iree_ok_status();
}
