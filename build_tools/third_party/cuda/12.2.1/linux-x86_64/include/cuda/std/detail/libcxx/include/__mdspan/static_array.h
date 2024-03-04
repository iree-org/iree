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

#ifndef _LIBCUDACXX___MDSPAN_STATIC_ARRAY_HPP
#define _LIBCUDACXX___MDSPAN_STATIC_ARRAY_HPP

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__fwd/span.h" // dynamic_extent
#include "../__mdspan/dynamic_extent.h"
#include "../__mdspan/macros.h"
#include "../__mdspan/maybe_static_value.h"
#include "../__mdspan/standard_layout_static_array.h"
#include "../__mdspan/type_list.h"
#include "../__utility/integer_sequence.h"
#include "../array"
#include "../cstddef"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#if !__MDSPAN_PRESERVE_STANDARD_LAYOUT

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 11

namespace __detail {

//==============================================================================

template <class _Tp, _Tp _Val, bool _Mask> struct __mask_element {};

template <class _Tp, _Tp... _Result>
struct __mask_sequence_assign_op {
  template <_Tp _V>
  __mask_sequence_assign_op<_Tp, _Result..., _V>
  operator=(__mask_element<_Tp, _V, true>&&);
  template <_Tp _V>
  __mask_sequence_assign_op<_Tp, _Result...>
  operator=(__mask_element<_Tp, _V, false>&&);
  using __result = _CUDA_VSTD::integer_sequence<_Tp, _Result...>;
};

template <class _Seq, class _Mask>
struct __mask_sequence;

template <class _Tp, _Tp... _Vals, bool... _Masks>
struct __mask_sequence<_CUDA_VSTD::integer_sequence<_Tp, _Vals...>, _CUDA_VSTD::integer_sequence<bool, _Masks...>>
{
  using type = typename decltype(
    __MDSPAN_FOLD_ASSIGN_LEFT(
      __mask_sequence_assign_op<_Tp>{}, /* = ... = */ __mask_element<_Tp, _Vals, _Masks>{}
    )
  )::__result;
};

//==============================================================================

template <class _Tp, class _static_t, class _Vals, _static_t __sentinal,
          class _Idxs, class _IdxsDynamic, class _IdxsDynamicIdxs>
class __partially_static_array_impl;

template <
  class _Tp, class _static_t,
  _static_t... __values_or_sentinals, _static_t __sentinal,
  size_t... _Idxs,
  size_t... _IdxsDynamic,
  size_t... _IdxsDynamicIdxs
>
class __partially_static_array_impl<
  _Tp,
  _static_t,
  _CUDA_VSTD::integer_sequence<_static_t, __values_or_sentinals...>,
  __sentinal,
  _CUDA_VSTD::integer_sequence<size_t, _Idxs...>,
  _CUDA_VSTD::integer_sequence<size_t, _IdxsDynamic...>,
  _CUDA_VSTD::integer_sequence<size_t, _IdxsDynamicIdxs...>
>
    : private __maybe_static_value<_Tp, _static_t, __values_or_sentinals, __sentinal,
                                   _Idxs>... {
private:

  template <size_t _Np>
  using __base_n = typename __type_at<_Np,
    __type_list<__maybe_static_value<_Tp, _static_t, __values_or_sentinals, __sentinal, _Idxs>...>
  >::type;

public:

  static constexpr auto __size = sizeof...(_Idxs);
  static constexpr auto __size_dynamic =
    __MDSPAN_FOLD_PLUS_RIGHT(static_cast<int>((__values_or_sentinals == __sentinal)), /* + ... + */ 0);

  //--------------------------------------------------------------------------

  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __partially_static_array_impl() = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __partially_static_array_impl(
      __partially_static_array_impl const &) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __partially_static_array_impl(
      __partially_static_array_impl &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __partially_static_array_impl &
  operator=(__partially_static_array_impl const &) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __partially_static_array_impl &
  operator=(__partially_static_array_impl &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  ~__partially_static_array_impl() noexcept = default;

  __MDSPAN_INLINE_FUNCTION
  constexpr __partially_static_array_impl(
      __construct_psa_from_all_exts_values_tag_t,
      __repeated_with_idxs<_Idxs, _Tp> const &... __vals) noexcept
      : __base_n<_Idxs>(__base_n<_Idxs>{{__vals}})... {}

  __MDSPAN_INLINE_FUNCTION
  constexpr __partially_static_array_impl(
      __construct_psa_from_dynamic_exts_values_tag_t,
      __repeated_with_idxs<_IdxsDynamicIdxs, _Tp> const &... __vals) noexcept
      : __base_n<_IdxsDynamic>(__base_n<_IdxsDynamic>{{__vals}})... {}

  __MDSPAN_INLINE_FUNCTION constexpr explicit __partially_static_array_impl(
    _CUDA_VSTD::array<_Tp, sizeof...(_Idxs)> const& __vals) noexcept
    : __partially_static_array_impl(
        __construct_psa_from_all_exts_values_tag,
        _CUDA_VSTD::get<_Idxs>(__vals)...) {}

  // clang-format off
  __MDSPAN_FUNCTION_REQUIRES(
    (__MDSPAN_INLINE_FUNCTION constexpr explicit),
    __partially_static_array_impl,
    (_CUDA_VSTD::array<_Tp, __size_dynamic> const &__vals), noexcept,
    /* requires */
      (sizeof...(_Idxs) != __size_dynamic)
  ): __partially_static_array_impl(
       __construct_psa_from_dynamic_exts_values_tag,
       _CUDA_VSTD::get<_IdxsDynamicIdxs>(__vals)...) {}
  // clang-format on

  template <class _Up, class _static_u, class _UValsSeq, _static_u __u_sentinal, class _UIdxsSeq,
            class _UIdxsDynamicSeq, class _UIdxsDynamicIdxsSeq>
  __MDSPAN_INLINE_FUNCTION constexpr __partially_static_array_impl(
    __partially_static_array_impl<
      _Up, _static_u, _UValsSeq, __u_sentinal, _UIdxsSeq,
     _UIdxsDynamicSeq, _UIdxsDynamicIdxsSeq> const &__rhs) noexcept
    : __partially_static_array_impl(
        __construct_psa_from_all_exts_values_tag,
        __rhs.template __get_n<_Idxs>()...) {}

  //--------------------------------------------------------------------------

  // See comment in the previous partial specialization for why this is
  // necessary.  Or just trust me that it's messy.
  __MDSPAN_FORCE_INLINE_FUNCTION
  constexpr __partially_static_array_impl const &__enable_psa_conversion() const
  noexcept {
      return *this;
  }

  template <size_t _Ip>
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Tp __get_n() const noexcept {
    return static_cast<__base_n<_Ip> const*>(this)->__value();
  }

  template <class _Up, size_t _Ip>
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr void __set_n(_Up&& __rhs) noexcept {
    static_cast<__base_n<_Ip>*>(this)->__set_value((_Up&&)__rhs);
  }

  template <size_t _Ip, _static_t __default = __sentinal>
  __MDSPAN_FORCE_INLINE_FUNCTION static constexpr _static_t
  __get_static_n() noexcept {
    return __base_n<_Ip>::__static_value == __sentinal ?
      __default : __base_n<_Ip>::__static_value;
  }

  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Tp
  __get(size_t __n) const noexcept {
    return __MDSPAN_FOLD_PLUS_RIGHT(
      (_Tp(_Idxs == __n) * __get_n<_Idxs>()), /* + ... + */ _Tp(0)
    );
  }

};

//==============================================================================

template <class _Tp, class _static_t, class _ValSeq, _static_t __sentinal, class _Idxs = _CUDA_VSTD::make_index_sequence<_ValSeq::size()>>
struct __partially_static_array_impl_maker;

template <
  class _Tp, class _static_t,  _static_t... _Vals, _static_t __sentinal, size_t... _Idxs
>
struct __partially_static_array_impl_maker<
  _Tp, _static_t, _CUDA_VSTD::integer_sequence<_static_t, _Vals...>, __sentinal, _CUDA_VSTD::integer_sequence<size_t, _Idxs...>
>
{
  using __dynamic_idxs = typename __mask_sequence<
    _CUDA_VSTD::integer_sequence<size_t, _Idxs...>,
    _CUDA_VSTD::integer_sequence<bool, (_Vals == __sentinal)...>
  >::type;
  using __impl_base =
    __partially_static_array_impl<_Tp, _static_t,
      _CUDA_VSTD::integer_sequence<_static_t, _Vals...>,
      __sentinal, _CUDA_VSTD::integer_sequence<size_t, _Idxs...>,
      __dynamic_idxs,
      _CUDA_VSTD::make_index_sequence<__dynamic_idxs::size()>
    >;
};

template <class _Tp, class _static_t, class _ValsSeq, _static_t __sentinal = dynamic_extent>
class __partially_static_array_with_sentinal
  : public __partially_static_array_impl_maker<_Tp, _static_t, _ValsSeq, __sentinal>::__impl_base
{
private:
  using __base_t = typename __partially_static_array_impl_maker<_Tp, _static_t, _ValsSeq, __sentinal>::__impl_base;
public:
  using __base_t::__base_t;
};

//==============================================================================

template <class T, class _static_t, _static_t... __values_or_sentinals>
struct __partially_static_sizes :
  __partially_static_array_with_sentinal<
    T, _static_t, _CUDA_VSTD::integer_sequence<_static_t, __values_or_sentinals...>>
{
private:
  using __base_t = __partially_static_array_with_sentinal<
    T, _static_t, _CUDA_VSTD::integer_sequence<_static_t, __values_or_sentinals...>>;
public:
  using __base_t::__base_t;
  template <class _UTag>
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr __partially_static_sizes<T, _static_t, __values_or_sentinals...>
  __with_tag() const noexcept {
    return *this;
  }
};

// Tags are needed for the standard layout version, but not here
template <class T, class _static_t, _static_t... __values_or_sentinals>
using __partially_static_sizes_tagged = __partially_static_sizes<T, _static_t, __values_or_sentinals...>;

} // end namespace __detail

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // !__MDSPAN_PRESERVE_STANDARD_LAYOUT

#endif // _LIBCUDACXX___MDSPAN_STATIC_ARRAY_HPP
