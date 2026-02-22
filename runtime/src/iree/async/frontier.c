// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/frontier.h"

#include "iree/base/api.h"

iree_status_t iree_async_frontier_validate(
    const iree_async_frontier_t* frontier) {
  // Check strict ascending axis order (implies no duplicates).
  for (uint8_t i = 1; i < frontier->entry_count; ++i) {
    if (frontier->entries[i - 1].axis >= frontier->entries[i].axis) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "frontier entries not in strictly ascending axis order: "
          "entries[%" PRIu8 "].axis (0x%016" PRIX64 ") >= entries[%" PRIu8
          "].axis (0x%016" PRIX64 ")",
          (uint8_t)(i - 1), frontier->entries[i - 1].axis, i,
          frontier->entries[i].axis);
    }
  }
  // Check for zero-epoch entries (meaningless — axis hasn't advanced).
  for (uint8_t i = 0; i < frontier->entry_count; ++i) {
    if (frontier->entries[i].epoch == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "frontier entries[%" PRIu8
                              "] has epoch 0 (axis 0x%016" PRIX64
                              "); zero-epoch entries are meaningless",
                              i, frontier->entries[i].axis);
    }
  }
  return iree_ok_status();
}

iree_async_frontier_comparison_t iree_async_frontier_compare(
    const iree_async_frontier_t* a, const iree_async_frontier_t* b) {
  bool has_less = false;
  bool has_greater = false;

  // Fast path: identical axis sets (common in steady-state).
  // When both frontiers track the same set of timelines, we skip the merge-scan
  // branching and do a straight epoch comparison loop with early exit.
  if (a->entry_count == b->entry_count) {
    bool axes_match = true;
    for (uint8_t k = 0; k < a->entry_count; ++k) {
      if (a->entries[k].axis != b->entries[k].axis) {
        axes_match = false;
        break;
      }
    }
    if (axes_match) {
      for (uint8_t k = 0; k < a->entry_count; ++k) {
        if (a->entries[k].epoch < b->entries[k].epoch) has_less = true;
        if (a->entries[k].epoch > b->entries[k].epoch) has_greater = true;
        if (has_less && has_greater) return IREE_ASYNC_FRONTIER_CONCURRENT;
      }
      if (has_less) return IREE_ASYNC_FRONTIER_BEFORE;
      if (has_greater) return IREE_ASYNC_FRONTIER_AFTER;
      return IREE_ASYNC_FRONTIER_EQUAL;
    }
  }

  // Slow path: merge-scan of two sorted arrays.
  uint8_t i = 0, j = 0;
  while (i < a->entry_count && j < b->entry_count) {
    if (a->entries[i].axis == b->entries[j].axis) {
      if (a->entries[i].epoch < b->entries[j].epoch) has_less = true;
      if (a->entries[i].epoch > b->entries[j].epoch) has_greater = true;
      ++i;
      ++j;
    } else if (a->entries[i].axis < b->entries[j].axis) {
      // a has an axis that b doesn't — a's epoch > 0, b's implicit epoch is 0.
      has_greater = true;
      ++i;
    } else {
      // b has an axis that a doesn't — b's epoch > 0, a's implicit epoch is 0.
      has_less = true;
      ++j;
    }
  }
  // Remaining entries in a: a has axes b doesn't.
  if (i < a->entry_count) has_greater = true;
  // Remaining entries in b: b has axes a doesn't.
  if (j < b->entry_count) has_less = true;

  if (has_less && has_greater) return IREE_ASYNC_FRONTIER_CONCURRENT;
  if (has_less) return IREE_ASYNC_FRONTIER_BEFORE;
  if (has_greater) return IREE_ASYNC_FRONTIER_AFTER;
  return IREE_ASYNC_FRONTIER_EQUAL;
}

iree_status_t iree_async_frontier_merge(iree_async_frontier_t* target,
                                        uint8_t target_capacity,
                                        const iree_async_frontier_t* source) {
  // Path 1: source empty — nothing to merge.
  if (source->entry_count == 0) {
    return iree_ok_status();
  }

  // Path 2: same axis set — epoch-max in place, zero entry movement.
  // This is the steady-state hot path: same GPU topology, advancing epochs.
  if (target->entry_count == source->entry_count) {
    bool axes_match = true;
    for (uint8_t k = 0; k < target->entry_count; ++k) {
      if (target->entries[k].axis != source->entries[k].axis) {
        axes_match = false;
        break;
      }
    }
    if (axes_match) {
      for (uint8_t k = 0; k < target->entry_count; ++k) {
        if (source->entries[k].epoch > target->entries[k].epoch) {
          target->entries[k].epoch = source->entries[k].epoch;
        }
      }
      return iree_ok_status();
    }
  }

  // Path 3: different axis sets — in-place right-to-left merge.
  //
  // First pass: count merged size without mutation (fail-fast on overflow).
  // Use uint16_t because the sum of two uint8_t entry counts can reach 510.
  uint16_t merged_count = 0;
  {
    uint8_t ti = 0, si = 0;
    while (ti < target->entry_count && si < source->entry_count) {
      if (target->entries[ti].axis == source->entries[si].axis) {
        ++ti;
        ++si;
      } else if (target->entries[ti].axis < source->entries[si].axis) {
        ++ti;
      } else {
        ++si;
      }
      ++merged_count;
    }
    merged_count += (target->entry_count - ti) + (source->entry_count - si);
  }

  if (merged_count > target_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "frontier merge would produce %" PRIu16
                            " entries but target capacity is %" PRIu8,
                            merged_count, target_capacity);
  }

  // Second pass: merge right-to-left (in-place, no scratch buffer).
  //
  // The write pointer starts at merged_count-1. The target read pointer starts
  // at entry_count-1. Since merged_count >= entry_count, wi >= ti always holds.
  // Each step decrements wi by 1 and ti by at most 1, so we never write to a
  // position we haven't yet read from in the target array.
  int16_t ti = (int16_t)target->entry_count - 1;
  int16_t si = (int16_t)source->entry_count - 1;
  int16_t wi = (int16_t)merged_count - 1;

  while (ti >= 0 && si >= 0) {
    if (target->entries[ti].axis > source->entries[si].axis) {
      target->entries[wi] = target->entries[ti];
      --ti;
    } else if (target->entries[ti].axis < source->entries[si].axis) {
      target->entries[wi] = source->entries[si];
      --si;
    } else {
      // Same axis: take the max epoch.
      target->entries[wi].axis = target->entries[ti].axis;
      target->entries[wi].epoch =
          iree_max(target->entries[ti].epoch, source->entries[si].epoch);
      --ti;
      --si;
    }
    --wi;
  }
  // Remaining source entries go at the front.
  while (si >= 0) {
    target->entries[wi] = source->entries[si];
    --si;
    --wi;
  }
  // Remaining target entries (ti >= 0) are already at positions 0..ti,
  // which equals 0..wi at this point — no movement needed.

  target->entry_count = (uint8_t)merged_count;
  return iree_ok_status();
}

bool iree_async_frontier_is_satisfied(
    const iree_async_frontier_t* frontier,
    const iree_async_frontier_entry_t* current_epochs,
    iree_host_size_t current_epochs_count) {
  if (frontier->entry_count == 0) return true;

  // Merge-scan with early exit on first unsatisfied entry.
  iree_host_size_t i = 0, j = 0;
  while (i < frontier->entry_count) {
    // Advance past current_epochs entries with smaller axes.
    while (j < current_epochs_count &&
           current_epochs[j].axis < frontier->entries[i].axis) {
      ++j;
    }
    // No more current epochs — remaining frontier entries are unsatisfied.
    if (j >= current_epochs_count) return false;
    // Frontier axis not present in current_epochs — epoch is implicitly 0.
    if (current_epochs[j].axis != frontier->entries[i].axis) return false;
    // Axis matched — check that current epoch has reached the target.
    if (current_epochs[j].epoch < frontier->entries[i].epoch) return false;
    ++i;
  }
  return true;
}
