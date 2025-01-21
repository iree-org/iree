// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/util/tree.h"

static iree_hal_hip_util_tree_node_t*
iree_hal_hip_util_tree_get_node_from_cache(iree_hal_hip_util_tree_t* tree) {
  if (tree->cache) {
    iree_hal_hip_util_tree_node_t* node = tree->cache;
    tree->cache = node->right;
    return node;
  }
  return NULL;
}

static void iree_hal_hip_util_tree_add_node_to_cache(
    iree_hal_hip_util_tree_t* tree, iree_hal_hip_util_tree_node_t* node) {
  node->right = tree->cache;
  tree->cache = node;
}

static void iree_hal_hip_util_tree_delete_node(
    iree_hal_hip_util_tree_t* tree, iree_hal_hip_util_tree_node_t* node) {
  if (node != &tree->nil) {
    iree_hal_hip_util_tree_add_node_to_cache(tree, node);
  }
}

static iree_status_t iree_hal_hip_util_tree_allocate_node(
    iree_hal_hip_util_tree_t* tree, iree_hal_hip_util_tree_node_t** out_node) {
  *out_node = NULL;
  iree_hal_hip_util_tree_node_t* node =
      iree_hal_hip_util_tree_get_node_from_cache(tree);
  if (node) {
    memset(node, 0, sizeof(*node) + tree->element_size);
  } else {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        tree->allocator, sizeof(*node) + tree->element_size, (void**)&node));
  }
  *out_node = node;
  node->data = (uint8_t*)node + sizeof(*node);
  return iree_ok_status();
}

static bool iree_hal_hip_util_tree_free_node(
    iree_hal_hip_util_tree_node_t* node, void* user_data) {
  iree_hal_hip_util_tree_t* tree = (iree_hal_hip_util_tree_t*)user_data;
  if ((uint8_t*)node >= tree->initial_node_cache &&
      (uint8_t*)node <
          tree->initial_node_cache + tree->initial_node_cache_size) {
    return true;
  }
  iree_allocator_free(tree->allocator, node);
  return true;
}

static void iree_hal_hip_util_tree_rotate_left(
    iree_hal_hip_util_tree_t* tree, iree_hal_hip_util_tree_node_t* node) {
  iree_hal_hip_util_tree_node_t* right_child = node->right;
  node->right = right_child->left;
  if (right_child->left != &tree->nil) {
    right_child->left->parent = node;
  }
  right_child->parent = node->parent;
  if (node->parent == NULL) {
    tree->root = right_child;
  } else if (node == node->parent->left) {
    node->parent->left = right_child;
  } else {
    node->parent->right = right_child;
  }
  right_child->left = node;
  node->parent = right_child;
}

static void iree_hal_hip_util_tree_rotate_right(
    iree_hal_hip_util_tree_t* tree, iree_hal_hip_util_tree_node_t* node) {
  iree_hal_hip_util_tree_node_t* left_child = node->left;
  node->left = left_child->right;
  if (left_child->right != &tree->nil) {
    left_child->right->parent = node;
  }
  left_child->parent = node->parent;
  if (node->parent == NULL) {
    tree->root = left_child;
  } else if (node == node->parent->right) {
    node->parent->right = left_child;
  } else {
    node->parent->left = left_child;
  }
  left_child->right = node;
  node->parent = left_child;
}

static iree_status_t iree_hal_hip_util_tree_insert_internal(
    iree_hal_hip_util_tree_t* tree, iree_host_size_t key,
    iree_hal_hip_util_tree_node_t* node) {
  node->left = &tree->nil;
  node->right = &tree->nil;
  node->key = key;
  node->red = true;  // red
  node->parent = NULL;

  iree_hal_hip_util_tree_node_t* search_position = tree->root;
  iree_hal_hip_util_tree_node_t* target_parent = NULL;
  while (search_position != &tree->nil) {
    target_parent = search_position;
    if (node->key < search_position->key) {
      search_position = search_position->left;
    } else if (node->key > search_position->key) {
      search_position = search_position->right;
    } else {
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "trying to insert a duplicate key");
    }
  }
  node->parent = target_parent;

  if (!target_parent) {
    tree->root = node;
  } else if (node->key < target_parent->key) {
    target_parent->left = node;
  } else {
    target_parent->right = node;
  }

  if (node->parent == NULL) {
    node->red = false;
    return iree_ok_status();
  }

  if (node->parent == tree->root) {
    return iree_ok_status();
  }

  while (node->parent->red) {
    if (node->parent == node->parent->parent->right) {
      iree_hal_hip_util_tree_node_t* uncle = node->parent->parent->left;
      if (uncle->red) {
        uncle->red = false;
        node->parent->red = false;
        node->parent->parent->red = true;
        node = node->parent->parent;
      } else {
        if (node == node->parent->left) {
          node = node->parent;
          iree_hal_hip_util_tree_rotate_right(tree, node);
        }
        node->parent->red = false;
        node->parent->parent->red = true;
        iree_hal_hip_util_tree_rotate_left(tree, node->parent->parent);
      }
    } else {
      iree_hal_hip_util_tree_node_t* uncle = node->parent->parent->right;
      if (uncle && uncle->red) {
        uncle->red = false;
        node->parent->red = false;
        node->parent->parent->red = true;
        node = node->parent->parent;
      } else {
        if (node == node->parent->right) {
          node = node->parent;
          iree_hal_hip_util_tree_rotate_left(tree, node);
        }
        node->parent->red = false;
        node->parent->parent->red = true;
        iree_hal_hip_util_tree_rotate_right(tree, node->parent->parent);
      }
    }
    if (node == tree->root) {
      break;
    }
  }
  tree->root->red = false;

  return iree_ok_status();
}

static bool iree_hal_hip_util_tree_walk_helper(
    iree_hal_hip_util_tree_node_t* node,
    iree_hal_hip_util_tree_walk_type_t walk_type,
    iree_hal_hip_util_tree_walk_callback_fn_t callback, void* user_data) {
  IREE_ASSERT_LE(walk_type, IREE_TREE_WALK_POSTORDER);
  if (!node || node->is_sentinel) {
    return true;
  }
  switch (walk_type) {
    case IREE_TREE_WALK_PREORDER:
      if (!callback(node, user_data)) {
        return false;
      }
      if (!iree_hal_hip_util_tree_walk_helper(node->left, walk_type, callback,
                                              user_data)) {
        return false;
      }
      return iree_hal_hip_util_tree_walk_helper(node->right, walk_type,
                                                callback, user_data);
    case IREE_TREE_WALK_INORDER:
      if (!iree_hal_hip_util_tree_walk_helper(node->left, walk_type, callback,
                                              user_data)) {
        return false;
      }
      if (!callback(node, user_data)) {
        return false;
      }
      return iree_hal_hip_util_tree_walk_helper(node->right, walk_type,
                                                callback, user_data);
    case IREE_TREE_WALK_POSTORDER:
      if (!iree_hal_hip_util_tree_walk_helper(node->left, walk_type, callback,
                                              user_data)) {
        return false;
      }
      if (!iree_hal_hip_util_tree_walk_helper(node->right, walk_type, callback,
                                              user_data)) {
        return false;
      }
      return callback(node, user_data);
  }
  return false;
}

static void iree_hal_hip_util_tree_replace(iree_hal_hip_util_tree_t* tree,
                                           iree_hal_hip_util_tree_node_t* dst,
                                           iree_hal_hip_util_tree_node_t* src) {
  if (!dst->parent) {
    tree->root = src;
  } else if (dst == dst->parent->left) {
    dst->parent->left = src;
  } else {
    dst->parent->right = src;
  }
  src->parent = dst->parent;
}

static void iree_hal_hip_util_tree_remove(
    iree_hal_hip_util_tree_t* tree, iree_hal_hip_util_tree_node_t* to_remove) {
  iree_hal_hip_util_tree_node_t* replacement = NULL;
  iree_hal_hip_util_tree_node_t* next = to_remove;

  bool initial_red = next->red;
  if (to_remove->left == &tree->nil) {
    replacement = to_remove->right;
    iree_hal_hip_util_tree_replace(tree, to_remove, to_remove->right);
  } else if (to_remove->right == &tree->nil) {
    replacement = to_remove->left;
    iree_hal_hip_util_tree_replace(tree, to_remove, to_remove->left);
  } else {
    next = iree_hal_hip_util_tree_node_next(to_remove);
    initial_red = next->red;
    replacement = next->right;
    if (next->parent == to_remove) {
      replacement->parent = next;
    } else {
      iree_hal_hip_util_tree_replace(tree, next, next->right);
      next->right = to_remove->right;
      next->right->parent = next;
    }

    iree_hal_hip_util_tree_replace(tree, to_remove, next);
    next->left = to_remove->left;
    next->left->parent = next;
    next->red = to_remove->red;
  }
  if (initial_red) {
    return;
  }
  while (replacement != tree->root && !replacement->red) {
    if (replacement == replacement->parent->left) {
      iree_hal_hip_util_tree_node_t* sibling = replacement->parent->right;
      if (sibling->red) {
        sibling->red = false;
        replacement->parent->red = true;
        iree_hal_hip_util_tree_rotate_left(tree, replacement->parent);
        sibling = replacement->parent->right;
      }

      if (!sibling->left->red && !sibling->right->red) {
        sibling->red = true;
        replacement = replacement->parent;
      } else {
        if (!sibling->right->red) {
          sibling->left->red = false;
          sibling->red = true;
          iree_hal_hip_util_tree_rotate_right(tree, sibling);
          sibling = replacement->parent->right;
        }
        sibling->red = replacement->parent->red;
        replacement->parent->red = false;
        sibling->right->red = false;
        iree_hal_hip_util_tree_rotate_left(tree, replacement->parent);
        replacement = tree->root;
      }
    } else {
      iree_hal_hip_util_tree_node_t* sibling = replacement->parent->left;
      if (sibling->red) {
        sibling->red = false;
        replacement->parent->red = true;
        iree_hal_hip_util_tree_rotate_right(tree, replacement->parent);
        sibling = replacement->parent->left;
      }

      if (!sibling->left->red && !sibling->right->red) {
        sibling->red = true;
        replacement = replacement->parent;
      } else {
        if (!sibling->left->red) {
          sibling->right->red = false;
          sibling->red = true;
          iree_hal_hip_util_tree_rotate_left(tree, sibling);
          sibling = replacement->parent->left;
        }

        sibling->red = replacement->parent->red;
        replacement->parent->red = false;
        sibling->left->red = false;
        iree_hal_hip_util_tree_rotate_right(tree, replacement->parent);
        replacement = tree->root;
      }
    }
  }
  replacement->red = false;
}

//===----------------------------------------------------------------------===//
// iree_hal_hip_util_tree_node_t
//===----------------------------------------------------------------------===//

void* iree_hal_hip_util_tree_node_get_value(
    const iree_hal_hip_util_tree_node_t* node) {
  IREE_ASSERT_ARGUMENT(node);
  return node->data;
}

iree_host_size_t iree_hal_hip_util_tree_node_get_key(
    const iree_hal_hip_util_tree_node_t* node) {
  IREE_ASSERT_ARGUMENT(node);
  return node->key;
}

iree_hal_hip_util_tree_node_t* iree_hal_hip_util_tree_node_next(
    iree_hal_hip_util_tree_node_t* node) {
  IREE_ASSERT_ARGUMENT(node != NULL);
  // 1. Find the smallest thing on our right hand side.
  if (!node->right->is_sentinel) {
    node = node->right;
    while (!node->left->is_sentinel) {
      node = node->left;
    }
    return node;
  }

  // 2. Find the parent who is not on the right
  iree_hal_hip_util_tree_node_t* parent = node->parent;
  while (parent && node == parent->right) {
    node = parent;
    parent = node->parent;
  }
  return parent;
}

iree_hal_hip_util_tree_node_t* iree_hal_hip_util_tree_node_prev(
    iree_hal_hip_util_tree_node_t* node) {
  IREE_ASSERT_ARGUMENT(node);
  // 1. to find the largest thing on our left hand side.
  if (!node->left->is_sentinel) {
    node = node->left;
    while (!node->right->is_sentinel) {
      node = node->right;
    }
    return node;
  }

  // 2. Find the parent who is not on the left
  iree_hal_hip_util_tree_node_t* parent = node->parent;
  while (parent && node == parent->left) {
    node = parent;
    parent = node->parent;
  }
  return parent;
}

//===----------------------------------------------------------------------===//
// iree_hal_hip_util_tree_t
//===----------------------------------------------------------------------===//

void iree_hal_hip_util_tree_initialize(iree_allocator_t allocator,
                                       iree_host_size_t element_size,
                                       void* initial_node_cache,
                                       iree_host_size_t initial_node_cache_size,
                                       iree_hal_hip_util_tree_t* out_tree) {
  out_tree->element_size = element_size;
  out_tree->allocator = allocator;
  out_tree->root = &out_tree->nil;
  out_tree->size = 0;
  out_tree->cache = NULL;  // Initialize cache
  memset(&out_tree->nil, 0x00, sizeof(out_tree->nil));
  out_tree->nil.is_sentinel = true;
  out_tree->initial_node_cache = initial_node_cache;
  out_tree->initial_node_cache_size = initial_node_cache_size;
  if (initial_node_cache) {
    memset(initial_node_cache, 0, initial_node_cache_size);
    iree_host_size_t node_size =
        iree_host_align(sizeof(out_tree->nil) + element_size, 16);

    iree_hal_hip_util_tree_node_t* node =
        (iree_hal_hip_util_tree_node_t*)initial_node_cache;
    for (iree_host_size_t i = 0; i < initial_node_cache_size / node_size; ++i) {
      node->data = (uint8_t*)node + sizeof(*node);
      iree_hal_hip_util_tree_add_node_to_cache(out_tree, node);
      node = (iree_hal_hip_util_tree_node_t*)((uint8_t*)node + node_size);
    }
  }
}

void iree_hal_hip_util_tree_deinitialize(iree_hal_hip_util_tree_t* tree) {
  iree_hal_hip_util_tree_walk(tree, IREE_TREE_WALK_POSTORDER,
                              iree_hal_hip_util_tree_free_node, tree);

  // Free cache nodes
  iree_hal_hip_util_tree_node_t* node = tree->cache;
  while (node) {
    iree_hal_hip_util_tree_node_t* next = node->right;
    if ((uint8_t*)node < tree->initial_node_cache ||
        (uint8_t*)node >
            tree->initial_node_cache + tree->initial_node_cache_size) {
      iree_allocator_free(tree->allocator, node);
    }
    node = next;
  }

  // Reset the tree structure.
  tree->root = &tree->nil;
  memset(&tree->nil, 0, sizeof(tree->nil));
  tree->nil.is_sentinel = true;
  tree->size = 0;
  tree->cache = NULL;
}

iree_host_size_t iree_hal_hip_util_tree_element_size(
    const iree_hal_hip_util_tree_t* tree) {
  return tree->element_size;
}

iree_status_t iree_hal_hip_util_tree_insert(
    iree_hal_hip_util_tree_t* tree, iree_host_size_t key,
    iree_hal_hip_util_tree_node_t** out_data) {
  *out_data = NULL;
  iree_hal_hip_util_tree_node_t* t = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_hip_util_tree_allocate_node(tree, &t));

  iree_status_t status = iree_hal_hip_util_tree_insert_internal(tree, key, t);
  if (!iree_status_is_ok(status)) {
    iree_hal_hip_util_tree_delete_node(tree, t);
    return status;
  }
  ++tree->size;
  *out_data = t;
  return status;
}

iree_host_size_t iree_hal_hip_util_tree_size(
    const iree_hal_hip_util_tree_t* tree) {
  return tree->size;
}

iree_status_t iree_hal_hip_util_tree_move_node(
    iree_hal_hip_util_tree_t* tree, iree_hal_hip_util_tree_node_t* node,
    iree_host_size_t new_key) {
  iree_hal_hip_util_tree_node_t* next = iree_hal_hip_util_tree_node_next(node);
  iree_hal_hip_util_tree_node_t* prev = iree_hal_hip_util_tree_node_prev(node);
  if ((!next || next->key > new_key) && (!prev || prev->key < new_key)) {
    // This node isn't going to move, just update its value.
    node->key = new_key;
    return iree_ok_status();
  }
  iree_hal_hip_util_tree_remove(tree, node);
  return iree_hal_hip_util_tree_insert_internal(tree, new_key, node);
}

iree_hal_hip_util_tree_node_t* iree_hal_hip_util_tree_get(
    const iree_hal_hip_util_tree_t* tree, iree_host_size_t key) {
  iree_hal_hip_util_tree_node_t* node = tree->root;
  while (node->is_sentinel == false) {
    if (key == node->key) {
      return node;
    } else if (key < node->key) {
      node = node->left;
    } else {
      node = node->right;
    }
  }
  return NULL;
}

iree_hal_hip_util_tree_node_t* iree_hal_hip_util_tree_lower_bound(
    const iree_hal_hip_util_tree_t* tree, iree_host_size_t key) {
  iree_hal_hip_util_tree_node_t* node = tree->root;
  iree_hal_hip_util_tree_node_t* last = NULL;
  while (node->is_sentinel == false) {
    last = node;
    if (key == node->key) {
      return node;
    } else if (key < node->key) {
      node = node->left;
    } else {
      node = node->right;
    }
  }
  if (!last || last->key > key) {
    return last;
  }
  return iree_hal_hip_util_tree_node_next(last);
}

iree_hal_hip_util_tree_node_t* iree_hal_hip_util_tree_upper_bound(
    const iree_hal_hip_util_tree_t* tree, iree_host_size_t key) {
  iree_hal_hip_util_tree_node_t* node = tree->root;
  iree_hal_hip_util_tree_node_t* last = NULL;
  while (node->is_sentinel == false) {
    last = node;
    if (key == node->key) {
      return node;
    } else if (key < node->key) {
      node = node->left;
    } else {
      node = node->right;
    }
  }
  if (!last || last->key > key) {
    return last;
  }
  while (last && last->key <= key) {
    last = iree_hal_hip_util_tree_node_next(last);
  }
  return last;
}

iree_hal_hip_util_tree_node_t* iree_hal_hip_util_tree_first(
    const iree_hal_hip_util_tree_t* tree) {
  if (!tree->root || tree->root->is_sentinel) {
    return NULL;
  }
  iree_hal_hip_util_tree_node_t* val = tree->root;
  while (!val->left->is_sentinel) {
    val = val->left;
  }
  return val;
}

iree_hal_hip_util_tree_node_t* iree_hal_hip_util_tree_last(
    const iree_hal_hip_util_tree_t* tree) {
  if (!tree->root || tree->root->is_sentinel) {
    return NULL;
  }
  iree_hal_hip_util_tree_node_t* val = tree->root;
  while (!val->right->is_sentinel) {
    val = val->right;
  }
  return val;
}

void iree_hal_hip_util_tree_erase(iree_hal_hip_util_tree_t* tree,
                                  iree_hal_hip_util_tree_node_t* node) {
  iree_hal_hip_util_tree_remove(tree, node);
  iree_hal_hip_util_tree_delete_node(tree, node);
  --tree->size;
}

void iree_hal_hip_util_tree_walk(
    const iree_hal_hip_util_tree_t* tree,
    iree_hal_hip_util_tree_walk_type_t walk_type,
    iree_hal_hip_util_tree_walk_callback_fn_t callback, void* user_data) {
  iree_hal_hip_util_tree_walk_helper(tree->root, walk_type, callback,
                                     user_data);
}
