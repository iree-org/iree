// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/vocab_trie.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Internal Types
//===----------------------------------------------------------------------===//

// Compact trie node stored in contiguous array.
typedef struct iree_tokenizer_trie_node_t {
  int32_t token_id;       // Token ID if word ends here, -1 otherwise.
  uint16_t child_count;   // Number of children (0-256).
  uint32_t child_offset;  // Offset into edges array.
} iree_tokenizer_trie_node_t;

// Edge connecting parent to child node.
// Edges for each node are stored contiguously and sorted by byte value.
typedef struct iree_tokenizer_trie_edge_t {
  uint8_t byte;         // Byte value for this edge.
  uint8_t padding[3];   // Alignment padding.
  uint32_t node_index;  // Index of child node.
} iree_tokenizer_trie_edge_t;

// Main trie structure.
struct iree_tokenizer_vocab_trie_t {
  iree_allocator_t allocator;

  // Node storage (contiguous array, index 0 is root).
  iree_tokenizer_trie_node_t* nodes;
  iree_host_size_t node_count;
  iree_host_size_t node_capacity;

  // Edge storage (contiguous array, sorted by byte within each node's range).
  iree_tokenizer_trie_edge_t* edges;
  iree_host_size_t edge_count;
  iree_host_size_t edge_capacity;

  // Maximum depth (longest token length).
  iree_host_size_t max_depth;
};

//===----------------------------------------------------------------------===//
// Builder Helpers
//===----------------------------------------------------------------------===//

// Growth factor for dynamic arrays.
#define IREE_TRIE_GROWTH_FACTOR 2

// Ensures node array has capacity for at least one more node.
static iree_status_t iree_tokenizer_trie_ensure_node_capacity(
    iree_tokenizer_vocab_trie_t* trie) {
  if (trie->node_count < trie->node_capacity) {
    return iree_ok_status();
  }

  iree_host_size_t new_capacity = trie->node_capacity * IREE_TRIE_GROWTH_FACTOR;
  if (new_capacity < 64) new_capacity = 64;

  iree_tokenizer_trie_node_t* new_nodes = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      trie->allocator, new_capacity * sizeof(iree_tokenizer_trie_node_t),
      (void**)&new_nodes));

  if (trie->nodes) {
    memcpy(new_nodes, trie->nodes,
           trie->node_count * sizeof(iree_tokenizer_trie_node_t));
    iree_allocator_free(trie->allocator, trie->nodes);
  }

  trie->nodes = new_nodes;
  trie->node_capacity = new_capacity;
  return iree_ok_status();
}

// Ensures edge array has capacity for at least `min_count` more edges.
// RVW: edge_count + min_count can overflow; while loop can overflow new_capacity
// RVW: Consider adding: if (trie->edge_count > IREE_HOST_SIZE_MAX - min_count)
// RVW: and overflow check in the while loop like vocab_builder.c does.
static iree_status_t iree_tokenizer_trie_ensure_edge_capacity(
    iree_tokenizer_vocab_trie_t* trie, iree_host_size_t min_count) {
  iree_host_size_t required = trie->edge_count + min_count;
  if (required <= trie->edge_capacity) {
    return iree_ok_status();
  }

  iree_host_size_t new_capacity = trie->edge_capacity * IREE_TRIE_GROWTH_FACTOR;
  if (new_capacity < 64) new_capacity = 64;
  while (new_capacity < required) {
    new_capacity *= IREE_TRIE_GROWTH_FACTOR;
  }

  iree_tokenizer_trie_edge_t* new_edges = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      trie->allocator, new_capacity * sizeof(iree_tokenizer_trie_edge_t),
      (void**)&new_edges));

  if (trie->edges) {
    memcpy(new_edges, trie->edges,
           trie->edge_count * sizeof(iree_tokenizer_trie_edge_t));
    iree_allocator_free(trie->allocator, trie->edges);
  }

  trie->edges = new_edges;
  trie->edge_capacity = new_capacity;
  return iree_ok_status();
}

// Allocates a new node and returns its index.
// RVW: node_count is iree_host_size_t (64-bit) but cast to uint32_t truncates.
// RVW: If node_count > UINT32_MAX, index wraps causing memory corruption.
// RVW: Consider adding: if (trie->node_count > UINT32_MAX) return error;
static iree_status_t iree_tokenizer_trie_allocate_node(
    iree_tokenizer_vocab_trie_t* trie, uint32_t* out_index) {
  IREE_RETURN_IF_ERROR(iree_tokenizer_trie_ensure_node_capacity(trie));

  uint32_t index = (uint32_t)trie->node_count++;
  trie->nodes[index].token_id = -1;
  trie->nodes[index].child_count = 0;
  trie->nodes[index].child_offset = 0;

  *out_index = index;
  return iree_ok_status();
}

// Binary search for an edge with the given byte in a node's edge range.
// Returns the edge index if found, or (iree_host_size_t)-1 if not found.
static iree_host_size_t iree_tokenizer_trie_find_edge(
    const iree_tokenizer_vocab_trie_t* trie, uint32_t node_index,
    uint8_t byte) {
  const iree_tokenizer_trie_node_t* node = &trie->nodes[node_index];
  if (node->child_count == 0) return (iree_host_size_t)-1;

  const iree_tokenizer_trie_edge_t* edges = &trie->edges[node->child_offset];
  iree_host_size_t left = 0;
  iree_host_size_t right = node->child_count;

  while (left < right) {
    iree_host_size_t mid = left + (right - left) / 2;
    if (edges[mid].byte < byte) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  if (left < node->child_count && edges[left].byte == byte) {
    return node->child_offset + left;
  }
  return (iree_host_size_t)-1;
}

// Adds an edge from parent to a new child node for the given byte.
// Maintains sorted order of edges by byte value.
// Returns the index of the new child node.
static iree_status_t iree_tokenizer_trie_add_edge(
    iree_tokenizer_vocab_trie_t* trie, uint32_t parent_index, uint8_t byte,
    uint32_t* out_child_index) {
  IREE_RETURN_IF_ERROR(iree_tokenizer_trie_ensure_edge_capacity(trie, 1));

  // Allocate new child node.
  uint32_t child_index = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_trie_allocate_node(trie, &child_index));

  iree_tokenizer_trie_node_t* parent = &trie->nodes[parent_index];

  if (parent->child_count == 0) {
    // First child: edge goes at current edge_count position.
    parent->child_offset = (uint32_t)trie->edge_count;
    trie->edges[trie->edge_count].byte = byte;
    trie->edges[trie->edge_count].node_index = child_index;
    trie->edge_count++;
    parent->child_count = 1;
  } else {
    // Find insertion position to maintain sorted order.
    iree_host_size_t insert_pos = 0;
    for (iree_host_size_t i = 0; i < parent->child_count; ++i) {
      if (trie->edges[parent->child_offset + i].byte < byte) {
        insert_pos = i + 1;
      } else {
        break;
      }
    }

    // Check if we're inserting at the end of the node's edges AND at the
    // end of the global edge array. Only then can we simply append.
    iree_host_size_t node_edge_end = parent->child_offset + parent->child_count;
    if (insert_pos == parent->child_count &&
        node_edge_end == trie->edge_count) {
      // Append at end.
      trie->edges[trie->edge_count].byte = byte;
      trie->edges[trie->edge_count].node_index = child_index;
      trie->edge_count++;
      parent->child_count++;
    } else {
      // Need to insert in the middle or the node's edges aren't at the end.
      // This is complex because we need to maintain contiguity.
      // For simplicity, we rebuild the edges for this node at the end.
      iree_host_size_t old_offset = parent->child_offset;
      iree_host_size_t old_count = parent->child_count;
      iree_host_size_t new_count = old_count + 1;

      // Ensure capacity for new_count edges.
      IREE_RETURN_IF_ERROR(
          iree_tokenizer_trie_ensure_edge_capacity(trie, new_count));

      // Copy edges to new location at end, inserting new edge in sorted order.
      iree_host_size_t new_offset = trie->edge_count;
      iree_host_size_t src = 0;
      for (iree_host_size_t dst = 0; dst < new_count; ++dst) {
        if (dst == insert_pos) {
          trie->edges[new_offset + dst].byte = byte;
          trie->edges[new_offset + dst].node_index = child_index;
        } else {
          trie->edges[new_offset + dst] = trie->edges[old_offset + src];
          src++;
        }
      }

      parent->child_offset = (uint32_t)new_offset;
      parent->child_count = (uint16_t)new_count;
      trie->edge_count += new_count;

      // Note: Old edges at old_offset are now orphaned but we don't reclaim
      // them during build. The final edge array may have some unused slots,
      // but this is acceptable for build simplicity.
    }
  }

  *out_child_index = child_index;
  return iree_ok_status();
}

// Inserts a token into the trie.
static iree_status_t iree_tokenizer_trie_insert_token(
    iree_tokenizer_vocab_trie_t* trie, int32_t token_id,
    const uint8_t* string_data, iree_host_size_t string_length) {
  uint32_t current_node = 0;  // Start at root.

  for (iree_host_size_t i = 0; i < string_length; ++i) {
    uint8_t byte = string_data[i];

    // Look for existing edge.
    iree_host_size_t edge_index =
        iree_tokenizer_trie_find_edge(trie, current_node, byte);

    if (edge_index != (iree_host_size_t)-1) {
      // Edge exists, follow it.
      current_node = trie->edges[edge_index].node_index;
    } else {
      // No edge, create new node and edge.
      uint32_t new_node = 0;
      IREE_RETURN_IF_ERROR(
          iree_tokenizer_trie_add_edge(trie, current_node, byte, &new_node));
      current_node = new_node;
    }
  }

  // Mark end of token.
  trie->nodes[current_node].token_id = token_id;

  // Update max depth.
  if (string_length > trie->max_depth) {
    trie->max_depth = string_length;
  }

  return iree_ok_status();
}

// Gets token string from token entry and string table.
// RVW: offset + length can overflow, bypassing the bounds check.
// RVW: Consider: if (token->string_offset > string_table.data_length ||
// RVW:              token->string_length > string_table.data_length - token->string_offset)
static iree_string_view_t iree_tokenizer_trie_token_string(
    const iree_tokenizer_token_t* token, iree_const_byte_span_t string_table) {
  if (token->string_offset + token->string_length > string_table.data_length) {
    return iree_string_view_empty();
  }
  return iree_make_string_view(
      (const char*)string_table.data + token->string_offset,
      token->string_length);
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_vocab_trie_build(
    const iree_tokenizer_token_t* tokens, iree_host_size_t token_count,
    iree_const_byte_span_t string_table, iree_allocator_t allocator,
    iree_tokenizer_vocab_trie_t** out_trie) {
  IREE_ASSERT_ARGUMENT(out_trie);
  *out_trie = NULL;

  // Allocate trie structure.
  iree_tokenizer_vocab_trie_t* trie = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*trie), (void**)&trie));
  memset(trie, 0, sizeof(*trie));
  trie->allocator = allocator;

  // Allocate root node.
  iree_status_t status =
      iree_tokenizer_trie_allocate_node(trie, &(uint32_t){0});

  // Insert all tokens.
  for (iree_host_size_t i = 0; i < token_count && iree_status_is_ok(status);
       ++i) {
    const iree_tokenizer_token_t* token = &tokens[i];

    // Skip unused/gap tokens.
    if (token->attributes & IREE_TOKENIZER_TOKEN_ATTR_UNUSED) {
      continue;
    }

    iree_string_view_t text =
        iree_tokenizer_trie_token_string(token, string_table);

    if (text.size > 0) {
      status = iree_tokenizer_trie_insert_token(
          trie, (int32_t)i, (const uint8_t*)text.data, text.size);
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_tokenizer_vocab_trie_free(trie);
    return status;
  }

  *out_trie = trie;
  return iree_ok_status();
}

void iree_tokenizer_vocab_trie_free(iree_tokenizer_vocab_trie_t* trie) {
  if (!trie) return;

  iree_allocator_t allocator = trie->allocator;

  if (trie->nodes) {
    iree_allocator_free(allocator, trie->nodes);
  }
  if (trie->edges) {
    iree_allocator_free(allocator, trie->edges);
  }
  iree_allocator_free(allocator, trie);
}

iree_host_size_t iree_tokenizer_vocab_trie_max_depth(
    const iree_tokenizer_vocab_trie_t* trie) {
  return trie ? trie->max_depth : 0;
}

iree_host_size_t iree_tokenizer_vocab_trie_node_count(
    const iree_tokenizer_vocab_trie_t* trie) {
  return trie ? trie->node_count : 0;
}

//===----------------------------------------------------------------------===//
// Cursor Operations
//===----------------------------------------------------------------------===//

void iree_tokenizer_trie_cursor_reset(iree_tokenizer_trie_cursor_t* cursor,
                                      const iree_tokenizer_vocab_trie_t* trie) {
  cursor->trie = trie;
  cursor->node_index = 0;  // Root node.
  cursor->depth = 0;
}

bool iree_tokenizer_trie_cursor_advance(iree_tokenizer_trie_cursor_t* cursor,
                                        uint8_t byte) {
  if (!cursor->trie || cursor->node_index >= cursor->trie->node_count) {
    return false;
  }

  iree_host_size_t edge_index =
      iree_tokenizer_trie_find_edge(cursor->trie, cursor->node_index, byte);

  if (edge_index == (iree_host_size_t)-1) {
    return false;  // No edge for this byte.
  }

  cursor->node_index = cursor->trie->edges[edge_index].node_index;
  cursor->depth++;
  return true;
}

int32_t iree_tokenizer_trie_cursor_token_id(
    const iree_tokenizer_trie_cursor_t* cursor) {
  if (!cursor->trie || cursor->node_index >= cursor->trie->node_count) {
    return -1;
  }
  return cursor->trie->nodes[cursor->node_index].token_id;
}
