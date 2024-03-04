/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef _LIBCUDACXX___MDSPAN_STANDARD_LAYOUT_STATIC_ARRAY_HPP
#define _LIBCUDACXX___MDSPAN_STANDARD_LAYOUT_STATIC_ARRAY_HPP

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__mdspan/compressed_pair.h"
#include "../__mdspan/dynamic_extent.h"
#include "../__mdspan/macros.h"
#if !defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
#include "../__mdspan/no_unique_address.h"
#endif
#include "../__type_traits/enable_if.h"
#include "../__utility/integer_sequence.h"
#include "../array"
#include "../cstddef"
#include "../span"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 11

namespace __detail {

//==============================================================================

_LIBCUDACXX_INLINE_VAR constexpr struct
    __construct_psa_from_dynamic_exts_values_tag_t {
} __construct_psa_from_dynamic_exts_values_tag = {};

_LIBCUDACXX_INLINE_VAR constexpr struct
    __construct_psa_from_all_exts_values_tag_t {
} __construct_psa_from_all_exts_values_tag = {};

struct __construct_psa_from_all_exts_array_tag_t {};
template <size_t _Np = 0> struct __construct_psa_from_dynamic_exts_array_tag_t {};

//==============================================================================

template <size_t _Ip, class _Tp> using __repeated_with_idxs = _Tp;

//==============================================================================

#if __MDSPAN_PRESERVE_STANDARD_LAYOUT

/**
 *  PSA = "partially static array"
 *
 * @tparam _Tp
 * @tparam _ValsSeq
 * @tparam __sentinal
 */
template <class _Tag, class _Tp, class _static_t, class _ValsSeq, _static_t __sentinal = static_cast<_static_t>(dynamic_extent),
          class _IdxsSeq = _CUDA_VSTD::make_index_sequence<_ValsSeq::size()>>
struct __standard_layout_psa;

//==============================================================================
// Static case
template <class _Tag, class _Tp, class _static_t, _static_t __value, _static_t... __values_or_sentinals,
          _static_t __sentinal, size_t _Idx, size_t... _Idxs>
struct __standard_layout_psa<
    _Tag, _Tp, _static_t, _CUDA_VSTD::integer_sequence<_static_t, __value, __values_or_sentinals...>,
    __sentinal, _CUDA_VSTD::integer_sequence<size_t, _Idx, _Idxs...>>
#if !defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
    : private __no_unique_address_emulation<__standard_layout_psa<
          _Tag, _Tp, _static_t, _CUDA_VSTD::integer_sequence<_static_t, __values_or_sentinals...>, __sentinal,
          _CUDA_VSTD::integer_sequence<size_t, _Idxs...>>>
#endif
{

  //--------------------------------------------------------------------------

  using __next_t =
      __standard_layout_psa<_Tag, _Tp, _static_t,
                            _CUDA_VSTD::integer_sequence<_static_t, __values_or_sentinals...>,
                            __sentinal, _CUDA_VSTD::integer_sequence<size_t, _Idxs...>>;

#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
  __MDSPAN_NO_UNIQUE_ADDRESS __next_t __next_;
#else
  using __base_t = __no_unique_address_emulation<__next_t>;
#endif

  __MDSPAN_FORCE_INLINE_FUNCTION constexpr __next_t &__next() noexcept {
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
    return __next_;
#else
    return this->__base_t::__ref();
#endif
  }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr __next_t const &__next() const noexcept {
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
    return __next_;
#else
    return this->__base_t::__ref();
#endif
  }

  static constexpr auto __size = sizeof...(_Idxs) + 1;
  static constexpr auto __size_dynamic = __next_t::__size_dynamic;

  //--------------------------------------------------------------------------

  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __standard_layout_psa() noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __standard_layout_psa(__standard_layout_psa const &) noexcept =
      default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __standard_layout_psa(__standard_layout_psa &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __standard_layout_psa &
  operator=(__standard_layout_psa const &) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __standard_layout_psa &
  operator=(__standard_layout_psa &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  ~__standard_layout_psa() noexcept = default;

  //--------------------------------------------------------------------------

  __MDSPAN_INLINE_FUNCTION
  constexpr __standard_layout_psa(
      __construct_psa_from_all_exts_values_tag_t, _Tp const & /*__val*/,
      __repeated_with_idxs<_Idxs, _Tp> const &... __vals) noexcept
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      : __next_{
#else
      : __base_t(__base_t{__next_t(
#endif
          __construct_psa_from_all_exts_values_tag, __vals...
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
#else
        )})
#endif
  { }

  template <class... _Ts>
  __MDSPAN_INLINE_FUNCTION constexpr __standard_layout_psa(
      __construct_psa_from_dynamic_exts_values_tag_t,
      _Ts const &... __vals) noexcept
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      : __next_{
#else
      : __base_t(__base_t{__next_t(
#endif
          __construct_psa_from_dynamic_exts_values_tag, __vals...
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
#else
        )})
#endif
  { }

  template <class _Up, size_t _Np>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      array<_Up, _Np> const &__vals) noexcept
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      : __next_{
#else
      : __base_t(__base_t{__next_t(
#endif
          __vals
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
#else
        )})
#endif
  { }

  template <class _Up, size_t _NStatic>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      __construct_psa_from_all_exts_array_tag_t const & __tag,
      array<_Up, _NStatic> const &__vals) noexcept
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      : __next_{
#else
      : __base_t(__base_t{__next_t(
#endif
          __tag, __vals
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
#else
        )})
#endif
  { }

  template <class _Up, size_t _IDynamic, size_t _NDynamic>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      __construct_psa_from_dynamic_exts_array_tag_t<_IDynamic> __tag,
      array<_Up, _NDynamic> const &__vals) noexcept
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      : __next_{
#else
      : __base_t(__base_t{__next_t(
#endif
          __tag, __vals
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
#else
        )})
#endif
  { }

  template <class _Up, size_t _Np>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      _CUDA_VSTD::span<_Up, _Np> const &__vals) noexcept
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      : __next_{
#else
      : __base_t(__base_t{__next_t(
#endif
          __vals
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
#else
        )})
#endif
  { }

  template <class _Up, size_t _NStatic>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      __construct_psa_from_all_exts_array_tag_t const & __tag,
      _CUDA_VSTD::span<_Up, _NStatic> const &__vals) noexcept
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      : __next_{
#else
      : __base_t(__base_t{__next_t(
#endif
          __tag, __vals
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
#else
        )})
#endif
  { }

  template <class _Up, size_t _IDynamic, size_t _NDynamic>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      __construct_psa_from_dynamic_exts_array_tag_t<_IDynamic> __tag,
      _CUDA_VSTD::span<_Up, _NDynamic> const &__vals) noexcept
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      : __next_{
#else
      : __base_t(__base_t{__next_t(
#endif
          __tag, __vals
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
#else
        )})
#endif
  { }

  template <class _UTag, class _Up, class _static_U, class _UValsSeq, _static_U __u_sentinal,
            class _IdxsSeq>
  __MDSPAN_INLINE_FUNCTION constexpr __standard_layout_psa(
      __standard_layout_psa<_UTag, _Up, _static_U, _UValsSeq, __u_sentinal, _IdxsSeq> const
          &__rhs) noexcept
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      : __next_{
#else
      : __base_t(__base_t{__next_t(
#endif
          __rhs.__next()
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
#else
        )})
#endif
  { }

  //--------------------------------------------------------------------------

  // See https://godbolt.org/z/_KSDNX for a summary-by-example of why this is
  // necessary. We're using inheritance here instead of an alias template
  // because we have to deduce __values_or_sentinals in several places, and
  // alias templates don't permit that in this context.
  __MDSPAN_FORCE_INLINE_FUNCTION
  constexpr __standard_layout_psa const &__enable_psa_conversion() const
      noexcept {
    return *this;
  }

  template <size_t _Ip, _CUDA_VSTD::enable_if_t<_Ip != _Idx, int> = 0>
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Tp __get_n() const noexcept {
    return __next().template __get_n<_Ip>();
  }
  template <size_t _Ip, _CUDA_VSTD::enable_if_t<_Ip == _Idx, int> = 1>
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Tp __get_n() const noexcept {
    return __value;
  }
  template <size_t _Ip, _CUDA_VSTD::enable_if_t<_Ip != _Idx, int> = 0>
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr void
  __set_n(_Tp const &__rhs) noexcept {
    __next().__set_value(__rhs);
  }
  template <size_t _Ip, _CUDA_VSTD::enable_if_t<_Ip == _Idx, int> = 1>
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr void
  __set_n(_Tp const &) noexcept {
    // Don't assert here because that would break constexpr. This better
    // not change anything, though
  }
  template <size_t _Ip, _CUDA_VSTD::enable_if_t<_Ip == _Idx, _static_t> = __sentinal>
  __MDSPAN_FORCE_INLINE_FUNCTION static constexpr _static_t __get_static_n() noexcept {
    return __value;
  }
  template <size_t _Ip, _CUDA_VSTD::enable_if_t<_Ip != _Idx, _static_t> __default = __sentinal>
  __MDSPAN_FORCE_INLINE_FUNCTION static constexpr _static_t __get_static_n() noexcept {
    return __next_t::template __get_static_n<_Ip, __default>();
  }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Tp __get(size_t __n) const noexcept {
    return __value * (_Tp(_Idx == __n)) + __next().__get(__n);
  }

  //--------------------------------------------------------------------------
};

//==============================================================================

// Dynamic case, __next_t may or may not be empty
template <class _Tag, class _Tp, class _static_t, _static_t __sentinal, _static_t... __values_or_sentinals,
          size_t _Idx, size_t... _Idxs>
struct __standard_layout_psa<
    _Tag, _Tp, _static_t, _CUDA_VSTD::integer_sequence<_static_t, __sentinal, __values_or_sentinals...>,
    __sentinal, _CUDA_VSTD::integer_sequence<size_t, _Idx, _Idxs...>> {
  //--------------------------------------------------------------------------

  using __next_t =
      __standard_layout_psa<_Tag, _Tp, _static_t,
                            _CUDA_VSTD::integer_sequence<_static_t, __values_or_sentinals...>,
                            __sentinal, _CUDA_VSTD::integer_sequence<size_t, _Idxs...>>;

  using __value_pair_t = __compressed_pair<_Tp, __next_t>;
  __value_pair_t __value_pair;
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr __next_t &__next() noexcept {
    return __value_pair.__second();
  }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr __next_t const &__next() const noexcept {
    return __value_pair.__second();
  }

  static constexpr auto __size = sizeof...(_Idxs) + 1;
  static constexpr auto __size_dynamic = 1 + __next_t::__size_dynamic;

  //--------------------------------------------------------------------------

  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __standard_layout_psa() noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __standard_layout_psa(__standard_layout_psa const &) noexcept =
      default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __standard_layout_psa(__standard_layout_psa &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __standard_layout_psa &
  operator=(__standard_layout_psa const &) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __standard_layout_psa &
  operator=(__standard_layout_psa &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  ~__standard_layout_psa() noexcept = default;

  //--------------------------------------------------------------------------

  __MDSPAN_INLINE_FUNCTION
  constexpr __standard_layout_psa(
      __construct_psa_from_all_exts_values_tag_t, _Tp const &__val,
      __repeated_with_idxs<_Idxs, _Tp> const &... __vals) noexcept
      : __value_pair(__val,
                     __next_t(__construct_psa_from_all_exts_values_tag,
                              __vals...)) {}

  template <class... _Ts>
  __MDSPAN_INLINE_FUNCTION constexpr __standard_layout_psa(
      __construct_psa_from_dynamic_exts_values_tag_t, _Tp const &__val,
      _Ts const &... __vals) noexcept
      : __value_pair(__val,
                     __next_t(__construct_psa_from_dynamic_exts_values_tag,
                              __vals...)) {}

  template <class _Up, size_t _Np>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      array<_Up, _Np> const &__vals) noexcept
      : __value_pair(_CUDA_VSTD::get<_Idx>(__vals), __vals) {}

  template <class _Up, size_t _NStatic>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      __construct_psa_from_all_exts_array_tag_t __tag,
      array<_Up, _NStatic> const &__vals) noexcept
      : __value_pair(
            _CUDA_VSTD::get<_Idx>(__vals),
            __next_t(__tag,
                     __vals)) {}

  template <class _Up, size_t _IDynamic, size_t _NDynamic>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      __construct_psa_from_dynamic_exts_array_tag_t<_IDynamic>,
      array<_Up, _NDynamic> const &__vals) noexcept
      : __value_pair(
            _CUDA_VSTD::get<_IDynamic>(__vals),
            __next_t(__construct_psa_from_dynamic_exts_array_tag_t<_IDynamic + 1>{},
                     __vals)) {}

  template <class _Up, size_t _Np>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      _CUDA_VSTD::span<_Up, _Np> const &__vals) noexcept
      : __value_pair(__vals[_Idx], __vals) {}

  template <class _Up, size_t _NStatic>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      __construct_psa_from_all_exts_array_tag_t __tag,
      _CUDA_VSTD::span<_Up, _NStatic> const &__vals) noexcept
      : __value_pair(
            __vals[_Idx],
            __next_t(__tag,
                     __vals)) {}

  template <class _Up, size_t _IDynamic, size_t _NDynamic>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      __construct_psa_from_dynamic_exts_array_tag_t<_IDynamic>,
      _CUDA_VSTD::span<_Up, _NDynamic> const &__vals) noexcept
      : __value_pair(
            __vals[_IDynamic],
            __next_t(__construct_psa_from_dynamic_exts_array_tag_t<_IDynamic + 1>{},
                     __vals)) {}

  template <class _UTag, class _Up, class _static_U, class _UValsSeq, _static_U __u_sentinal,
            class _UIdxsSeq>
  __MDSPAN_INLINE_FUNCTION constexpr __standard_layout_psa(
      __standard_layout_psa<_UTag, _Up, _static_U, _UValsSeq, __u_sentinal, _UIdxsSeq> const
          &__rhs) noexcept
      : __value_pair(__rhs.template __get_n<_Idx>(), __rhs.__next()) {}

  //--------------------------------------------------------------------------

  // See comment in the previous partial specialization for why this is
  // necessary.  Or just trust me that it's messy.
  __MDSPAN_FORCE_INLINE_FUNCTION
  constexpr __standard_layout_psa const &__enable_psa_conversion() const
      noexcept {
    return *this;
  }

  template <size_t _Ip, _CUDA_VSTD::enable_if_t<_Ip != _Idx, int> = 0>
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Tp __get_n() const noexcept {
    return __next().template __get_n<_Ip>();
  }
  template <size_t _Ip, _CUDA_VSTD::enable_if_t<_Ip == _Idx, int> = 1>
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Tp __get_n() const noexcept {
    return __value_pair.__first();
  }
  template <size_t _Ip, _CUDA_VSTD::enable_if_t<_Ip != _Idx, int> = 0>
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr void
  __set_n(_Tp const &__rhs) noexcept {
    __next().__set_value(__rhs);
  }
  template <size_t _Ip, _CUDA_VSTD::enable_if_t<_Ip == _Idx, int> = 1>
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr void
  __set_n(_Tp const &__rhs) noexcept {
    __value_pair.__first() = __rhs;
  }
  template <size_t _Ip, _CUDA_VSTD::enable_if_t<_Ip == _Idx, _static_t> __default = __sentinal>
  __MDSPAN_FORCE_INLINE_FUNCTION static constexpr _static_t __get_static_n() noexcept {
    return __default;
  }
  template <size_t _Ip, _CUDA_VSTD::enable_if_t<_Ip != _Idx, _static_t> __default = __sentinal>
  __MDSPAN_FORCE_INLINE_FUNCTION static constexpr _static_t __get_static_n() noexcept {
    return __next_t::template __get_static_n<_Ip, __default>();
  }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Tp __get(size_t __n) const noexcept {
    return __value_pair.__first() * (_Tp(_Idx == __n)) + __next().__get(__n);
  }

  //--------------------------------------------------------------------------
};

// empty/terminal case
template <class _Tag, class _Tp, class _static_t, _static_t __sentinal>
struct __standard_layout_psa<_Tag, _Tp, _static_t, _CUDA_VSTD::integer_sequence<_static_t>, __sentinal,
                             _CUDA_VSTD::integer_sequence<size_t>> {
  //--------------------------------------------------------------------------

  static constexpr auto __size = 0;
  static constexpr auto __size_dynamic = 0;

  //--------------------------------------------------------------------------

  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __standard_layout_psa() noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __standard_layout_psa(__standard_layout_psa const &) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __standard_layout_psa(__standard_layout_psa &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __standard_layout_psa &
  operator=(__standard_layout_psa const &) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __standard_layout_psa &
  operator=(__standard_layout_psa &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  ~__standard_layout_psa() noexcept = default;

  __MDSPAN_INLINE_FUNCTION
  constexpr __standard_layout_psa(
      __construct_psa_from_all_exts_values_tag_t) noexcept {}

  template <class... _Ts>
  __MDSPAN_INLINE_FUNCTION constexpr __standard_layout_psa(
      __construct_psa_from_dynamic_exts_values_tag_t) noexcept {}

  template <class _Up, size_t _Np>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      array<_Up, _Np> const &) noexcept {}

  template <class _Up, size_t _NStatic>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      __construct_psa_from_all_exts_array_tag_t,
      array<_Up, _NStatic> const &) noexcept {}

  template <class _Up, size_t _IDynamic, size_t _NDynamic>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      __construct_psa_from_dynamic_exts_array_tag_t<_IDynamic>,
      array<_Up, _NDynamic> const &) noexcept {}

  template <class _Up, size_t _Np>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      _CUDA_VSTD::span<_Up, _Np> const &) noexcept {}

  template <class _Up, size_t _NStatic>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      __construct_psa_from_all_exts_array_tag_t,
      _CUDA_VSTD::span<_Up, _NStatic> const &) noexcept {}

  template <class _Up, size_t _IDynamic, size_t _NDynamic>
  __MDSPAN_INLINE_FUNCTION constexpr explicit __standard_layout_psa(
      __construct_psa_from_dynamic_exts_array_tag_t<_IDynamic>,
      _CUDA_VSTD::span<_Up, _NDynamic> const &) noexcept {}

  template <class _UTag, class _Up, class _static_U, class _UValsSeq, _static_U __u_sentinal,
            class _UIdxsSeq>
  __MDSPAN_INLINE_FUNCTION constexpr __standard_layout_psa(
      __standard_layout_psa<_UTag, _Up, _static_U, _UValsSeq, __u_sentinal, _UIdxsSeq> const&) noexcept {}

  // See comment in the previous partial specialization for why this is
  // necessary.  Or just trust me that it's messy.
  __MDSPAN_FORCE_INLINE_FUNCTION
  constexpr __standard_layout_psa const &__enable_psa_conversion() const
      noexcept {
    return *this;
  }

  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Tp __get(size_t /*n*/) const noexcept {
    return 0;
  }
};

// Same thing, but with a disambiguator so that same-base issues doesn't cause
// a loss of standard-layout-ness.
template <class _Tag, class T, class _static_t, _static_t... __values_or_sentinals>
struct __partially_static_sizes_tagged
    : __standard_layout_psa<
          _Tag, T, _static_t,
          _CUDA_VSTD::integer_sequence<_static_t, __values_or_sentinals...>> {
  using __tag_t = _Tag;
  using __psa_impl_t = __standard_layout_psa<
      _Tag, T, _static_t, _CUDA_VSTD::integer_sequence<_static_t, __values_or_sentinals...>>;
  using __psa_impl_t::__psa_impl_t;
#ifdef __MDSPAN_DEFAULTED_CONSTRUCTORS_INHERITANCE_WORKAROUND
  __MDSPAN_INLINE_FUNCTION
#endif
  constexpr __partially_static_sizes_tagged() noexcept
#ifdef __MDSPAN_DEFAULTED_CONSTRUCTORS_INHERITANCE_WORKAROUND
    : __psa_impl_t() { }
#else
    = default;
#endif
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __partially_static_sizes_tagged(
      __partially_static_sizes_tagged const &) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __partially_static_sizes_tagged(
      __partially_static_sizes_tagged &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __partially_static_sizes_tagged &
  operator=(__partially_static_sizes_tagged const &) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __partially_static_sizes_tagged &
  operator=(__partially_static_sizes_tagged &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  ~__partially_static_sizes_tagged() noexcept = default;

  template <class _UTag>
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr explicit __partially_static_sizes_tagged(
    __partially_static_sizes_tagged<_UTag, T, _static_t, __values_or_sentinals...> const& __vals
  ) noexcept : __psa_impl_t(__vals.__enable_psa_conversion()) { }
};

struct __no_tag {};
template <class T, class _static_t, _static_t... __values_or_sentinals>
struct __partially_static_sizes
    : __partially_static_sizes_tagged<__no_tag, T, _static_t, __values_or_sentinals...> {
private:
  using __base_t =
      __partially_static_sizes_tagged<__no_tag, T, _static_t, __values_or_sentinals...>;
  template <class _UTag>
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr __partially_static_sizes(
    __partially_static_sizes_tagged<_UTag, T, _static_t, __values_or_sentinals...>&& __vals
  ) noexcept : __base_t(_CUDA_VSTD::move(__vals)) { }
public:
  using __base_t::__base_t;

#ifdef __MDSPAN_DEFAULTED_CONSTRUCTORS_INHERITANCE_WORKAROUND
  __MDSPAN_INLINE_FUNCTION
  constexpr __partially_static_sizes() noexcept : __base_t() { }
#endif
  template <class _UTag>
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr __partially_static_sizes_tagged<
      _UTag, T, _static_t, __values_or_sentinals...>
  __with_tag() const noexcept {
    return __partially_static_sizes_tagged<_UTag, T, _static_t, __values_or_sentinals...>(*this);
  }
};

#endif // __MDSPAN_PRESERVE_STATIC_LAYOUT

} // end namespace __detail

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MDSPAN_STANDARD_LAYOUT_STATIC_ARRAY_HPP
