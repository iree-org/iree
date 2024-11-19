// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_UTIL_TREE_H_
#define IREE_HAL_DRIVERS_HIP_UTIL_TREE_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_hip_util_tree_node_t iree_hal_hip_util_tree_node_t;
typedef struct iree_hal_hip_util_tree_t iree_hal_hip_util_tree_t;

typedef enum iree_hal_hip_util_tree_walk_type_e {
  IREE_TREE_WALK_PREORDER,
  IREE_TREE_WALK_INORDER,
  IREE_TREE_WALK_POSTORDER,
} iree_hal_hip_util_tree_walk_type_t;

//===----------------------------------------------------------------------===//
// iree_hal_hip_util_tree_node_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_hip_util_tree_node_t {
  bool red;
  iree_hal_hip_util_tree_node_t* left;
  iree_hal_hip_util_tree_node_t* right;
  iree_hal_hip_util_tree_node_t* parent;
  iree_host_size_t key;
  bool is_sentinel;
  uint8_t* data;
} iree_hal_hip_util_tree_node_t;

// Returns the value associated with the given node.
void* iree_hal_hip_util_tree_node_get_value(
    const iree_hal_hip_util_tree_node_t* node);

// Returns the key for the given node.
iree_host_size_t iree_hal_hip_util_tree_node_get_key(
    const iree_hal_hip_util_tree_node_t* node);

// Callback function for the iree_hip_util_tree_walk.
//
// This is provided the node and user_data for every node in the tree. A return
// of false from this function will cause the tree walk to complete without
// walking any further nodes.
typedef bool (*iree_hal_hip_util_tree_walk_callback_fn_t)(
    iree_hal_hip_util_tree_node_t* node, void* user_data);

// Walks the entire tree invoking the callback for every node in the tree.
void iree_hal_hip_util_tree_walk(
    const iree_hal_hip_util_tree_t* tree,
    iree_hal_hip_util_tree_walk_type_t walk_type,
    iree_hal_hip_util_tree_walk_callback_fn_t callback, void* user_data);

// Returns the next node in the tree or NULL.
iree_hal_hip_util_tree_node_t* iree_hal_hip_util_tree_node_next(
    iree_hal_hip_util_tree_node_t* node);

// Returns the previous node in the tree or NULL.
iree_hal_hip_util_tree_node_t* iree_hal_hip_util_tree_node_prev(
    iree_hal_hip_util_tree_node_t* node);

//===----------------------------------------------------------------------===//
// iree_hal_hip_util_tree_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_hip_util_tree_t {
  iree_allocator_t allocator;
  iree_host_size_t element_size;
  iree_hal_hip_util_tree_node_t* root;
  iree_host_size_t size;
  iree_hal_hip_util_tree_node_t* cache;  // Cache for deleted nodes
  iree_hal_hip_util_tree_node_t nil;
  uint8_t* initial_node_cache;
  iree_host_size_t initial_node_cache_size;
} iree_hal_hip_util_tree_t;

// Initializes the tree for values of |element_size|.
//
// If |initial_node_cache| is not null then it points to
// a block of memory that will be used to hold nodes before
// the tree ever tries to use the allocator.
void iree_hal_hip_util_tree_initialize(iree_allocator_t allocator,
                                       iree_host_size_t element_size,
                                       void* initial_node_cache,
                                       iree_host_size_t initial_node_cache_size,
                                       iree_hal_hip_util_tree_t* out_tree);

// Deinitializes the tree and frees any memory that was allocated.
void iree_hal_hip_util_tree_deinitialize(iree_hal_hip_util_tree_t* tree);

// Returns the number of bytes that are allocated for every value in the tree.
iree_host_size_t iree_hal_hip_util_tree_element_size(
    const iree_hal_hip_util_tree_t* tree);

// Inserts a new node into the tree with the given |key|.
//
// If the key is already present in the tree an error is returned.
iree_status_t iree_hal_hip_util_tree_insert(
    iree_hal_hip_util_tree_t* tree, iree_host_size_t key,
    iree_hal_hip_util_tree_node_t** out_data);

// Returns the number of elements in the tree.
iree_host_size_t iree_hal_hip_util_tree_size(
    const iree_hal_hip_util_tree_t* tree);

// Moves a node that already exists in the tree to a new location with the given
// key.
iree_status_t iree_hal_hip_util_tree_move_node(
    iree_hal_hip_util_tree_t* tree, iree_hal_hip_util_tree_node_t* node,
    iree_host_size_t new_key);

// Returns the node in the tree that has a given key.
//
// Returns NULL if the key could not be found in the tree.
iree_hal_hip_util_tree_node_t* iree_hal_hip_util_tree_get(
    const iree_hal_hip_util_tree_t* tree, iree_host_size_t key);

// Returns the first node in the tree that has a key that is >= |key| or NULL.
iree_hal_hip_util_tree_node_t* iree_hal_hip_util_tree_lower_bound(
    const iree_hal_hip_util_tree_t* tree, iree_host_size_t key);

// Returns the first node in the tree that has a key that is > |key| or NULL;
iree_hal_hip_util_tree_node_t* iree_hal_hip_util_tree_upper_bound(
    const iree_hal_hip_util_tree_t* tree, iree_host_size_t key);

// Returns the node in the tree with the smallest |key| or NULL.
iree_hal_hip_util_tree_node_t* iree_hal_hip_util_tree_first(
    const iree_hal_hip_util_tree_t* tree);

// Returns the node in the tree with the largest |key| or NULL.
iree_hal_hip_util_tree_node_t* iree_hal_hip_util_tree_last(
    const iree_hal_hip_util_tree_t* tree);

// Erases the given node from the tree.
void iree_hal_hip_util_tree_erase(iree_hal_hip_util_tree_t* tree,
                                  iree_hal_hip_util_tree_node_t* node);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HIP_UTIL_TREE_H_
