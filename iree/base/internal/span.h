// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_SPAN_H_
#define IREE_BASE_INTERNAL_SPAN_H_
#ifdef __cplusplus

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>

// std::span is available starting in C++20.
// Prior to that we fall back to our simplified implementation below.
#if defined(__has_include)
#if __has_include(<span>) && __cplusplus >= 202002L
#define IREE_HAVE_STD_SPAN 1
#include <span>
#endif  // __has_include(<span>)
#endif  // __has_include

#ifndef IREE_HAVE_STD_SPAN
#include <limits>
#endif

namespace iree {

#if defined(IREE_HAVE_STD_SPAN)

// Alias. Once we bump up our minimum C++ version we can drop this entire file.
template <typename T>
using span = std::span<T>;

#else

constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

// A pared down version of std::span doing just enough for our uses in IREE.
// Most of the IREE code started using absl::Span which while close to std::span
// has some additional functionality of its own and is missing some from std.
// The benefit here is that means we only need to implement the intersection of
// the two as none of our code uses those newer std features.
//
// https://en.cppreference.com/w/cpp/container/span/subspan
template <typename T>
class span {
 private:
  template <typename V>
  using remove_cv_t = typename std::remove_cv<V>::type;
  template <typename V>
  using decay_t = typename std::decay<V>::type;

  template <typename C>
  static constexpr auto GetDataImpl(C& c, char) noexcept -> decltype(c.data()) {
    return c.data();
  }
  static inline char* GetDataImpl(std::string& s, int) noexcept {
    return &s[0];
  }
  template <typename C>
  static constexpr auto GetData(C& c) noexcept -> decltype(GetDataImpl(c, 0)) {
    return GetDataImpl(c, 0);
  }

  template <typename C>
  using HasSize =
      std::is_integral<decay_t<decltype(std::declval<C&>().size())> >;

  template <typename V, typename C>
  using HasData =
      std::is_convertible<decay_t<decltype(GetData(std::declval<C&>()))>*,
                          V* const*>;

  template <typename C>
  using EnableIfConvertibleFrom =
      typename std::enable_if<HasData<T, C>::value && HasSize<C>::value>::type;

  template <typename U>
  using EnableIfConstView =
      typename std::enable_if<std::is_const<T>::value, U>::type;

  template <typename U>
  using EnableIfMutableView =
      typename std::enable_if<!std::is_const<T>::value, U>::type;

 public:
  using value_type = remove_cv_t<T>;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator = pointer;
  using const_iterator = const_pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  constexpr span() noexcept : span(nullptr, 0) {}
  constexpr span(pointer array, size_type length) noexcept
      : ptr_(array), len_(length) {}

  template <size_type N>
  constexpr span(T (&a)[N]) noexcept : span(a, N) {}

  template <typename V, typename = EnableIfConvertibleFrom<V>,
            typename = EnableIfMutableView<V> >
  explicit span(V& v) noexcept : span(GetData(v), v.size()) {}

  template <typename V, typename = EnableIfConvertibleFrom<V>,
            typename = EnableIfConstView<V> >
  constexpr span(const V& v) noexcept : span(GetData(v), v.size()) {}

  template <typename LazyT = T, typename = EnableIfConstView<LazyT> >
  span(std::initializer_list<value_type> v) noexcept
      : span(v.begin(), v.size()) {}

  constexpr pointer data() const noexcept { return ptr_; }

  constexpr size_type size() const noexcept { return len_; }

  constexpr size_type length() const noexcept { return size(); }

  constexpr bool empty() const noexcept { return size() == 0; }

  constexpr reference operator[](size_type i) const noexcept {
    // MSVC 2015 accepts this as constexpr, but not ptr_[i]
    assert(i < size());
    return *(data() + i);
  }

  constexpr reference at(size_type i) const {
    return i < size() ? *(data() + i) : (std::abort(), *(data() + i));
  }

  constexpr reference front() const noexcept {
    assert(size() > 0);
    return *data();
  }
  constexpr reference back() const noexcept {
    assert(size() > 0);
    return *(data() + size() - 1);
  }

  constexpr iterator begin() const noexcept { return data(); }
  constexpr iterator end() const noexcept { return data() + size(); }

  constexpr reverse_iterator rbegin() const noexcept {
    return reverse_iterator(end());
  }
  constexpr reverse_iterator rend() const noexcept {
    return reverse_iterator(begin());
  }

  constexpr span subspan(size_type pos = 0,
                         size_type len = iree::dynamic_extent) const {
    return (pos <= size()) ? span(data() + pos, std::min(size() - pos, len))
                           : (std::abort(), span());
  }

  constexpr span first(size_type len) const {
    return (len <= size()) ? span(data(), len) : (std::abort(), span());
  }

  constexpr span last(size_type len) const {
    return (len <= size()) ? span(size() - len + data(), len)
                           : (std::abort(), span());
  }

 private:
  pointer ptr_;
  size_type len_;
};

#endif  // IREE_HAVE_STD_SPAN

}  // namespace iree

#endif  // __cplusplus
#endif  // IREE_BASE_INTERNAL_SPAN_H_
