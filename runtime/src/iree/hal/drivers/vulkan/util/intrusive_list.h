// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Doubly linked list using element interior storage.
// This has the performance of std::list (that means O(1) on insert and remove)
// but performs no allocations and has better caching behavior.
//
// Elements are maintained in lists by way of IntrusiveListLinks, with each link
// allowing the element to exist in one list simultaneously. In the most simple
// case subclassing IntrusiveLinkBase will let the type be added to a list with
// little boilerplate. If an element must be in more than one list
// simultaneously IntrusiveListLinks can be added as members.
//
// Usage (simple):
//   class MySimpleElement : public IntrusiveLinkBase {};
//   IntrusiveList<MySimpleElement> list;
//   list.push_back(new MySimpleElement());
//   for (auto element : list) { ... }
//
// Usage (multiple lists):
//   class MultiElement {
//    public:
//     IntrusiveListLink list_link_a;
//     IntrusiveListLink list_link_b;
//   };
//   IntrusiveList<MultiElement, offsetof(MultiElement, list_link_a)> list_a;
//   IntrusiveList<MultiElement, offsetof(MultiElement, list_link_b)> list_b;
//
// By default elements in the list are not retained and must be kept alive
// externally. For automatic memory management there are specializations for
// std::unique_ptr.
//
// Usage (unique_ptr):
//   IntrusiveList<std::unique_ptr<MyElement>> list;
//   list.push_back(std::make_unique<MyElement>());
//   std::unique_ptr<MyElement> elm = list.take(list.front());
//
// This type is thread-unsafe.

#ifndef IREE_HAL_DRIVERS_VULKAN_UTIL_INTRUSIVE_LIST_H_
#define IREE_HAL_DRIVERS_VULKAN_UTIL_INTRUSIVE_LIST_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <utility>

#include "iree/base/assert.h"

namespace iree {

// Define to enable extensive checks after each mutation of the intrusive list.
// #define IREE_PARANOID_INTRUSIVE_LIST

// Storage for the doubly-linked list.
// This is embedded within all elements in an intrusive list.
struct IntrusiveListLink {
  IntrusiveListLink* prev = nullptr;
  IntrusiveListLink* next = nullptr;

  IntrusiveListLink() = default;

  // Prevent copies.
  IntrusiveListLink(const IntrusiveListLink&) = delete;
  IntrusiveListLink& operator=(const IntrusiveListLink&) = delete;
};

template <class T>
struct IntrusiveLinkBase : public T {
 public:
  IntrusiveListLink link;
};

template <>
struct IntrusiveLinkBase<void> {
 public:
  IntrusiveListLink link;
};

// Base type for intrusive lists.
// This is either used directly when the list is on naked pointers or
// specialized to std::unique_ptr.
template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
class IntrusiveListBase {
 public:
  using self_type = IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>;

  IntrusiveListBase() = default;
  virtual ~IntrusiveListBase() { clear(); }

  // Prevent copies.
  IntrusiveListBase(const IntrusiveListBase&) = delete;
  IntrusiveListBase& operator=(const IntrusiveListBase&) = delete;

  // Returns true if the list is empty.
  // Performance: O(1)
  constexpr bool empty() const { return head_ == nullptr; }

  // Returns the total number of items in the list.
  // Performance: O(1)
  constexpr size_t size() const { return count_; }

  // Returns true if the given item is contained within the list.
  // Performance: O(n)
  bool contains(T* value) const;

  // Appends the contents of the given list to this one.
  // The |other_list| is cleared.
  // Performance: O(1)
  void merge_from(self_type* other_list);

  // Removes all items from the list.
  // Performance: O(n)
  void clear();

  IteratorT begin() const { return IteratorT(head_); }
  IteratorT end() const { return IteratorT(nullptr); }
  ReverseIteratorT rbegin() const { return ReverseIteratorT(tail_); }
  ReverseIteratorT rend() const { return ReverseIteratorT(nullptr); }

  // Returns the next item in the list relative to the given item.
  // |value| must exist in the list.
  // Performance: O(1)
  T* next(T* value) const;

  // Returns the previous item in the list relative to the given item.
  // |value| must exist in the list.
  // Performance: O(1)
  T* previous(T* value) const;

  // Returns the item at the front of the list, if any.
  // Performance: O(1)
  T* front() const;

  // Inserts an item at the front of the list.
  // Performance: O(1)
  void push_front(T* value);

  // Removes the item at the front of the list.
  // Performance: O(1)
  void pop_front();

  // Returns the item at the back of the list, if any.
  // Performance: O(1)
  T* back() const;

  // Inserts an item at the back of the list.
  // Performance: O(1)
  void push_back(T* value);

  // Removes the item at the back of the list.
  // Performance: O(1)
  void pop_back();

  // Inserts an item into the list before the given iterator.
  // Performance: O(1)
  void insert(const IteratorT& it, T* value) { return insert(*it, value); }
  void insert(T* position, T* value);

  // Erases the given item from the list.
  // Returns the item following the erased item, if any.
  // Performance: O(1)
  T* erase(T* value);

  // Erases the item from the list at the given iterator.
  // Performance: O(1)
  IteratorT erase(const IteratorT& it);
  ReverseIteratorT erase(const ReverseIteratorT& it);

  // Replaces the item with a new item at the same position.
  // |new_value| must not be contained in any list.
  // Performance: O(1)
  void replace(T* old_value, T* new_value);

  // Sorts the list with the given comparison function.
  // The sort function is the same as used by std::sort.
  //
  // Uses merge sort O(N log N) using the algorithm described here:
  // http://www.chiark.greenend.org.uk/~sgtatham/algorithms/listsort.html
  void sort(bool (*compare_fn)(T* a, T* b));

 protected:
  // Called when an item is added to the list.
  virtual void OnAdd(T* value) {}
  // Called when an item is removed from the list.
  virtual void OnRemove(T* value) {}
  // Called when an item is removed and deallocated.
  virtual void OnDeallocate(T* value) {}

  // Performs expensive correctness checks on the list structure. It's too slow
  // to use in normal builds (even dbg), so it should only be used when there's
  // a suspected issue with an intrusive list. Define
  // IREE_PARANOID_INTRUSIVE_LIST to enable.
  void CheckCorrectness() const;

  IntrusiveListLink* head_ = nullptr;
  IntrusiveListLink* tail_ = nullptr;
  size_t count_ = 0;
};

namespace impl {

// std::iterator has been deprecated in C++17, therefore we create our own
// definition of the required types for iterators.
template <typename Category, typename T, typename Distance = std::ptrdiff_t,
          typename Pointer = T*, typename Reference = T&>
struct iterator {
  using iterator_category = Category;
  using value_type = T;
  using difference_type = Distance;
  using pointer = Pointer;
  using reference = Reference;
};

}  // namespace impl

// Basic iterator for an IntrusiveList.
template <typename T, size_t kOffset, bool kForward>
class IntrusiveListIterator
    : public impl::iterator<std::input_iterator_tag, int> {
 public:
  using self_type = IntrusiveListIterator<T, kOffset, kForward>;

  explicit IntrusiveListIterator(IntrusiveListLink* current)
      : current_(current) {}
  IntrusiveListIterator& operator++();
  self_type operator++(int);
  self_type& operator--();
  self_type operator--(int);
  bool operator==(const self_type& rhs) const;
  bool operator!=(const self_type& rhs) const;
  T* operator*() const;

 protected:
  IntrusiveListLink* current_;
};

// Specialized IntrusiveListBase used for unreferenced naked pointers.
// This very thinly wraps the base type and does no special memory management.
template <typename T, size_t kOffset>
class IntrusiveListUnrefBase
    : public IntrusiveListBase<T, IntrusiveListIterator<T, kOffset, true>,
                               IntrusiveListIterator<T, kOffset, false>,
                               kOffset> {
 public:
  using IteratorT = IntrusiveListIterator<T, kOffset, true>;
  using ReverseIteratorT = IntrusiveListIterator<T, kOffset, false>;
  using base_list = IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>;

  using base_list::clear;

  // Removes all items from the list and calls the given deleter function for
  // each of them. The built-in OnDeallocate will not be used.
  // Performance: O(n)
  void clear(const std::function<void(T*)>& deleter);

 private:
  using base_list::count_;
  using base_list::head_;
  using base_list::tail_;
};

constexpr size_t kUseDefaultLinkOffset = std::numeric_limits<size_t>::max();

// IntrusiveList for raw pointers with a specified offset.
// Use this if there are multiple links within a type.
//
// Usage:
//  struct MyType {
//   IntrusiveListLink link_a;
//   IntrusiveListLink link_b;
//  };
//  IntrusiveList<MyType, offsetof(MyType, link_a)> list_a;
//  IntrusiveList<MyType, offsetof(MyType, link_b)> list_b;
template <typename T, size_t kOffset = kUseDefaultLinkOffset>
class IntrusiveList : public IntrusiveListUnrefBase<T, kOffset> {};

// IntrusiveList for raw pointers.
// Items added to the list will not be owned by the list and must be freed by
// the caller.
//
// Usage:
//  struct MyType : public IntrusiveListBase<void> {};
//  IntrusiveList<MyType> list;
//  auto* p = new MyType();
//  list.push_back(p);  // p is not retained and won't be freed!
//  delete p;
template <typename T>
class IntrusiveList<T, kUseDefaultLinkOffset>
    : public IntrusiveListUnrefBase<T, offsetof(T, link)> {};

// -- implementation --

namespace impl {

// Maps an IntrusiveListLink to its containing type T.
template <typename T, size_t kOffset>
static inline T* LinkToT(IntrusiveListLink* link) {
  if (link) {
    return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(link) - kOffset);
  } else {
    return nullptr;
  }
}

// Maps a containing type T to its IntrusiveListLink.
template <typename T, size_t kOffset>
static inline IntrusiveListLink* TToLink(T* value) {
  if (value) {
    return reinterpret_cast<IntrusiveListLink*>(
        reinterpret_cast<uintptr_t>(value) + kOffset);
  } else {
    return nullptr;
  }
}

}  // namespace impl

template <typename T, size_t kOffset, bool kForward>
IntrusiveListIterator<T, kOffset, kForward>&
IntrusiveListIterator<T, kOffset, kForward>::operator++() {
  if (current_) {
    current_ = kForward ? current_->next : current_->prev;
  }
  return *this;
}

template <typename T, size_t kOffset, bool kForward>
IntrusiveListIterator<T, kOffset, kForward>
IntrusiveListIterator<T, kOffset, kForward>::operator++(int) {
  self_type tmp(current_);
  operator++();
  return tmp;
}

template <typename T, size_t kOffset, bool kForward>
IntrusiveListIterator<T, kOffset, kForward>&
IntrusiveListIterator<T, kOffset, kForward>::operator--() {
  if (current_) {
    current_ = kForward ? current_->prev : current_->next;
  }
  return *this;
}

template <typename T, size_t kOffset, bool kForward>
IntrusiveListIterator<T, kOffset, kForward>
IntrusiveListIterator<T, kOffset, kForward>::operator--(int) {
  self_type tmp(current_);
  operator--();
  return tmp;
}

template <typename T, size_t kOffset, bool kForward>
bool IntrusiveListIterator<T, kOffset, kForward>::operator==(
    const self_type& rhs) const {
  return rhs.current_ == current_;
}

template <typename T, size_t kOffset, bool kForward>
bool IntrusiveListIterator<T, kOffset, kForward>::operator!=(
    const self_type& rhs) const {
  return !operator==(rhs);
}

template <typename T, size_t kOffset, bool kForward>
T* IntrusiveListIterator<T, kOffset, kForward>::operator*() const {
  return impl::LinkToT<T, kOffset>(current_);
}

template <typename T, size_t kOffset>
void IntrusiveListUnrefBase<T, kOffset>::clear(
    const std::function<void(T*)>& deleter) {
  auto* link = head_;
  while (link) {
    auto* next = link->next;
    link->prev = link->next = nullptr;
    deleter(impl::LinkToT<T, kOffset>(link));
    link = next;
  }
  head_ = tail_ = nullptr;
  count_ = 0;
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
void IntrusiveListBase<T, IteratorT, ReverseIteratorT,
                       kOffset>::CheckCorrectness() const {
#if defined(IREE_PARANOID_INTRUSIVE_LIST)
  auto* link = head_;
  IntrusiveListLink* previous = nullptr;
  size_t actual_count = 0;
  while (link) {
    ++actual_count;
    if (!link->prev) {
      IREE_ASSERT_EQ(link, head_);
    }
    if (!link->next) {
      IREE_ASSERT_EQ(link, tail_);
    }
    IREE_ASSERT_EQ(link->prev, previous);
    previous = link;
    link = link->next;
  }
  IREE_ASSERT_EQ(actual_count, count_);
#endif  // IREE_PARANOID_INTRUSIVE_LIST
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
bool IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>::contains(
    T* value) const {
  if (!value) return false;
  // TODO(benvanik): faster way of checking? requires list ptr in link?
  auto* needle = impl::TToLink<T, kOffset>(value);
  auto* link = head_;
  while (link) {
    if (link == needle) {
      return true;
    }
    link = link->next;
  }
  return false;
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
void IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>::merge_from(
    self_type* other_list) {
  if (tail_) {
    tail_->next = other_list->head_;
  }
  if (other_list->head_) {
    other_list->head_->prev = tail_;
  }
  if (!head_) {
    head_ = other_list->head_;
  }
  tail_ = other_list->tail_;

  other_list->head_ = nullptr;
  other_list->tail_ = nullptr;

  count_ += other_list->count_;
  other_list->count_ = 0;
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
void IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>::clear() {
  auto* link = head_;
  while (link) {
    auto* next = link->next;
    link->prev = link->next = nullptr;
    OnDeallocate(impl::LinkToT<T, kOffset>(link));
    link = next;
  }
  head_ = tail_ = nullptr;
  count_ = 0;
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
inline T* IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>::next(
    T* value) const {
  if (!value) {
    return nullptr;
  }
  auto* link = impl::TToLink<T, kOffset>(value);
  return impl::LinkToT<T, kOffset>(link->next);
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
inline T* IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>::previous(
    T* value) const {
  if (!value) {
    return nullptr;
  }
  auto* link = impl::TToLink<T, kOffset>(value);
  return impl::LinkToT<T, kOffset>(link->prev);
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
inline T* IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>::front()
    const {
  return impl::LinkToT<T, kOffset>(head_);
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
void IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>::push_front(
    T* value) {
  IREE_ASSERT(value);
  auto* link = impl::TToLink<T, kOffset>(value);
  IREE_ASSERT(!link->next);
  IREE_ASSERT(!link->prev);
  link->next = head_;
  link->prev = nullptr;
  head_ = link;
  if (link->next) {
    link->next->prev = link;
  }
  if (!tail_) {
    tail_ = link;
  }
  ++count_;
  OnAdd(value);
  CheckCorrectness();
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
void IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>::pop_front() {
  IREE_ASSERT(head_);
  auto* link = head_;
  if (link) {
    head_ = head_->next;
    link->next = link->prev = nullptr;
    if (head_) {
      head_->prev = nullptr;
    }
    if (link == tail_) {
      tail_ = nullptr;
    }
    --count_;
    OnDeallocate(impl::LinkToT<T, kOffset>(link));
  }
  CheckCorrectness();
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
inline T* IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>::back()
    const {
  return impl::LinkToT<T, kOffset>(tail_);
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
void IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>::push_back(
    T* value) {
  IREE_ASSERT(value);
  auto* link = impl::TToLink<T, kOffset>(value);
  IREE_ASSERT(!link->next);
  IREE_ASSERT(!link->prev);
  link->prev = tail_;
  link->next = nullptr;
  tail_ = link;
  if (link->prev) {
    link->prev->next = link;
  }
  if (!head_) {
    head_ = link;
  }
  ++count_;
  OnAdd(value);
  CheckCorrectness();
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
void IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>::pop_back() {
  IREE_ASSERT(tail_);
  auto* link = tail_;
  if (link) {
    tail_ = tail_->prev;
    link->next = link->prev = nullptr;
    if (tail_) {
      tail_->next = nullptr;
    }
    if (link == head_) {
      head_ = nullptr;
    }
    --count_;
    OnDeallocate(impl::LinkToT<T, kOffset>(link));
  }
  CheckCorrectness();
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
void IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>::insert(
    T* position, T* value) {
  IREE_ASSERT(value);
  auto* link = impl::TToLink<T, kOffset>(value);
  auto* position_link = impl::TToLink<T, kOffset>(position);
  IREE_ASSERT(!link->next);
  IREE_ASSERT(!link->prev);

  if (position_link == head_) {
    push_front(value);
  } else if (position_link == nullptr) {
    push_back(value);
  } else {
    link->next = position_link;
    link->prev = position_link->prev;
    position_link->prev->next = link;
    position_link->prev = link;
    ++count_;
    OnAdd(value);
  }
  CheckCorrectness();
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
T* IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>::erase(T* value) {
  if (!value) {
    return nullptr;
  }
  auto* link = impl::TToLink<T, kOffset>(value);
  if (link->prev) {
    IREE_ASSERT_NE(link, head_);
    link->prev->next = link->next;
  } else {
    IREE_ASSERT_EQ(link, head_);
    head_ = link->next;
  }
  if (link->next) {
    IREE_ASSERT_NE(link, tail_);
    link->next->prev = link->prev;
  } else {
    IREE_ASSERT_EQ(link, tail_);
    tail_ = link->prev;
  }
  auto* next = link->next;
  link->next = link->prev = nullptr;
  --count_;
  OnDeallocate(value);
  CheckCorrectness();
  return impl::LinkToT<T, kOffset>(next);
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
IteratorT IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>::erase(
    const IteratorT& it) {
  return IteratorT(impl::TToLink<T, kOffset>(erase(*it)));
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
ReverseIteratorT IntrusiveListBase<T, IteratorT, ReverseIteratorT,
                                   kOffset>::erase(const ReverseIteratorT& it) {
  return ReverseIteratorT(impl::TToLink<T, kOffset>(erase(*it)));
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
void IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>::replace(
    T* old_value, T* new_value) {
  IREE_ASSERT(old_value);
  IREE_ASSERT(new_value);
  IREE_ASSERT_NE(old_value, new_value);
  auto* old_link = impl::TToLink<T, kOffset>(old_value);
  auto* new_link = impl::TToLink<T, kOffset>(new_value);
  new_link->next = old_link->next;
  new_link->prev = old_link->prev;
  if (new_link->prev) {
    new_link->prev->next = new_link;
  } else {
    head_ = new_link;
  }
  if (new_link->next) {
    new_link->next->prev = new_link;
  } else {
    tail_ = new_link;
  }
  old_link->next = old_link->prev = nullptr;
  OnAdd(new_value);
  OnDeallocate(old_value);
  CheckCorrectness();
}

template <typename T, typename IteratorT, typename ReverseIteratorT,
          size_t kOffset>
void IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>::sort(
    bool (*compare_fn)(T* a, T* b)) {
  if (empty()) {
    // Empty list no-op.
    return;
  }
  // Repeatedly run until the list is sorted.
  int in_size = 1;
  while (true) {
    IntrusiveListLink* p = head_;
    IntrusiveListLink* q = nullptr;
    IntrusiveListLink* e = nullptr;
    IntrusiveListLink* tail = nullptr;
    head_ = nullptr;
    tail_ = nullptr;
    // Repeatedly merge sublists.
    int merge_count = 0;
    do {
      ++merge_count;
      q = p;
      // Determine the size of the first part and find the second.
      int p_size = 0;
      for (int i = 0; i < in_size; ++i) {
        ++p_size;
        q = q->next;
        if (!q) {
          break;
        }
      }
      // Merge the two lists (if we have two).
      int q_size = in_size;
      while (p_size > 0 || (q_size > 0 && q)) {
        if (p_size == 0) {
          // p is empty; e must come from q.
          e = q;
          q = q->next;
          --q_size;
        } else if (q_size == 0 || !q) {
          // q is empty; e must come from p.
          e = p;
          p = p->next;
          --p_size;
        } else if (compare_fn(impl::LinkToT<T, kOffset>(p),
                              impl::LinkToT<T, kOffset>(q))) {
          // p <= q; e must come from p.
          e = p;
          p = p->next;
          --p_size;
        } else {
          // q < p; e must come from q.
          e = q;
          q = q->next;
          --q_size;
        }
        // Append e to the merged list.
        if (tail) {
          tail->next = e;
        } else {
          head_ = e;
        }
        e->prev = tail;
        tail = e;
      }
      p = q;
    } while (p);
    tail->next = nullptr;
    if (merge_count <= 1) {
      // List is now sorted; stash and return.
      tail_ = tail;
      CheckCorrectness();
      return;
    }
    // Run merge again with larger lists.
    in_size *= 2;
  }
}

}  // namespace iree

// Specializations:
#include "iree/hal/drivers/vulkan/util/intrusive_list_unique_ptr.inc"

#endif  // IREE_HAL_DRIVERS_VULKAN_UTIL_INTRUSIVE_LIST_H_
