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

#ifndef _LIBCUDACXX___MDSPAN_EXTENTS_HPP
#define _LIBCUDACXX___MDSPAN_EXTENTS_HPP

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__mdspan/macros.h"
#if !defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
#include "../__mdspan/no_unique_address.h"
#endif
#include "../__mdspan/static_array.h"
#include "../__mdspan/standard_layout_static_array.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_convertible.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__type_traits/make_unsigned.h"
#include "../__utility/integer_sequence.h"
#include "../array"
#include "../cstddef"
#include "../span"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#if defined(_LIBCUDACXX_PUSH_MACROS)
  _LIBCUDACXX_PUSH_MACROS
#endif
#include "../__undef_macros"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 11

namespace __detail {

template<size_t ... _Extents>
struct __count_dynamic_extents;

template<size_t _Ep, size_t ... _Extents>
struct __count_dynamic_extents<_Ep,_Extents...> {
  static constexpr size_t val = (_Ep==dynamic_extent?1:0) + __count_dynamic_extents<_Extents...>::val;
};

template<>
struct __count_dynamic_extents<> {
  static constexpr size_t val = 0;
};

template <size_t... _Extents, size_t... _OtherExtents>
__MDSPAN_HOST_DEVICE
static constexpr false_type __check_compatible_extents(
  false_type, _CUDA_VSTD::integer_sequence<size_t, _Extents...>, _CUDA_VSTD::integer_sequence<size_t, _OtherExtents...>
) noexcept { return { }; }

// This helper prevents ICE's on MSVC.
template <size_t _Lhs, size_t _Rhs>
struct __compare_extent_compatible : integral_constant<bool,
    _Lhs == dynamic_extent ||
    _Rhs == dynamic_extent ||
    _Lhs == _Rhs>
{};

template <size_t... _Extents, size_t... _OtherExtents>
static integral_constant<
  bool,
  __MDSPAN_FOLD_AND(
    (
      __compare_extent_compatible<_Extents, _OtherExtents>::value
    ) /* && ... */
  )
>
__MDSPAN_HOST_DEVICE
__check_compatible_extents(
  true_type, _CUDA_VSTD::integer_sequence<size_t, _Extents...>, _CUDA_VSTD::integer_sequence<size_t, _OtherExtents...>
) noexcept { return { }; }

struct __extents_tag { };

} // end namespace __detail

template <class _ThisIndexType, size_t... _Extents>
class extents
#if !defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
  : private __detail::__no_unique_address_emulation<
      __detail::__partially_static_sizes_tagged<__detail::__extents_tag, _ThisIndexType , size_t, _Extents...>>
#endif
{
public:

  using rank_type = size_t;
  using index_type = _ThisIndexType;
  using size_type = make_unsigned_t<index_type>;

// internal typedefs which for technical reasons are public
  using __storage_t = __detail::__partially_static_sizes_tagged<__detail::__extents_tag, index_type, size_t, _Extents...>;

#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
  __MDSPAN_NO_UNIQUE_ADDRESS __storage_t __storage_;
#else
  using __base_t = __detail::__no_unique_address_emulation<__storage_t>;
#endif

// private members dealing with the way we internally store dynamic extents
 private:

  __MDSPAN_FORCE_INLINE_FUNCTION constexpr
  __storage_t& __storage() noexcept {
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
    return __storage_;
#else
    return this->__base_t::__ref();
#endif
  }
  __MDSPAN_FORCE_INLINE_FUNCTION
  constexpr __storage_t const& __storage() const noexcept {
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
    return __storage_;
#else
    return this->__base_t::__ref();
#endif
  }

  template <size_t... _Idxs>
  __MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr
  index_type _static_extent_impl(size_t __n, _CUDA_VSTD::integer_sequence<size_t, _Idxs...>) noexcept {
    return __MDSPAN_FOLD_PLUS_RIGHT(((_Idxs == __n) ? _Extents : 0), /* + ... + */ 0);
  }

  template <class, size_t...>
  friend class extents;

  template <class _OtherIndexType, size_t... _OtherExtents, size_t... _Idxs>
  __MDSPAN_INLINE_FUNCTION
  constexpr bool _eq_impl(_CUDA_VSTD::extents<_OtherIndexType, _OtherExtents...>, false_type, _CUDA_VSTD::index_sequence<_Idxs...>) const noexcept { return false; }
  template <class _OtherIndexType, size_t... _OtherExtents, size_t... _Idxs>
  __MDSPAN_INLINE_FUNCTION
  constexpr bool _eq_impl(
    _CUDA_VSTD::extents<_OtherIndexType, _OtherExtents...> __other,
    true_type, _CUDA_VSTD::index_sequence<_Idxs...>
  ) const noexcept {
    return __MDSPAN_FOLD_AND(
      (__storage().template __get_n<_Idxs>() == __other.__storage().template __get_n<_Idxs>()) /* && ... */
    );
  }

  template <class _OtherIndexType, size_t... _OtherExtents, size_t... _Idxs>
  __MDSPAN_INLINE_FUNCTION
  constexpr bool _not_eq_impl(_CUDA_VSTD::extents<_OtherIndexType, _OtherExtents...>, false_type, _CUDA_VSTD::index_sequence<_Idxs...>) const noexcept { return true; }
  template <class _OtherIndexType, size_t... _OtherExtents, size_t... _Idxs>
  __MDSPAN_INLINE_FUNCTION
  constexpr bool _not_eq_impl(
    _CUDA_VSTD::extents<_OtherIndexType, _OtherExtents...> __other,
    true_type, _CUDA_VSTD::index_sequence<_Idxs...>
  ) const noexcept {
    return __MDSPAN_FOLD_OR(
      (__storage().template __get_n<_Idxs>() != __other.__storage().template __get_n<_Idxs>()) /* || ... */
    );
  }

#if !defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
  __MDSPAN_INLINE_FUNCTION constexpr explicit
  extents(__base_t&& __b) noexcept
    : __base_t(_CUDA_VSTD::move(__b))
  { }
#endif


// public interface:
public:
  /* Defined above for use in the private code
  using rank_type = size_t;
  using index_type = _ThisIndexType;
  */

  __MDSPAN_INLINE_FUNCTION
  static constexpr rank_type rank() noexcept { return sizeof...(_Extents); }
  __MDSPAN_INLINE_FUNCTION
  static constexpr rank_type rank_dynamic() noexcept { return __MDSPAN_FOLD_PLUS_RIGHT((rank_type(_Extents == dynamic_extent)), /* + ... + */ 0); }

  //--------------------------------------------------------------------------------
  // Constructors, Destructors, and Assignment

  // Default constructor
  __MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr extents() noexcept = default;

  // Converting constructor
  __MDSPAN_TEMPLATE_REQUIRES(
    class _OtherIndexType, size_t... _OtherExtents,
    /* requires */ (
      /* multi-stage check to protect from invalid pack expansion when sizes don't match? */
      decltype(__detail::__check_compatible_extents(
        integral_constant<bool, sizeof...(_Extents) == sizeof...(_OtherExtents)>{},
        _CUDA_VSTD::integer_sequence<size_t, _Extents...>{},
        _CUDA_VSTD::integer_sequence<size_t, _OtherExtents...>{}
      ))::value
    )
  )
  __MDSPAN_INLINE_FUNCTION
  __MDSPAN_CONDITIONAL_EXPLICIT(
    (((_Extents != dynamic_extent) && (_OtherExtents == dynamic_extent)) || ...) ||
    (_CUDA_VSTD::numeric_limits<index_type>::max() < _CUDA_VSTD::numeric_limits<_OtherIndexType>::max()))
  constexpr extents(const extents<_OtherIndexType, _OtherExtents...>& __other)
    noexcept
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
    : __storage_{
#else
    : __base_t(__base_t{__storage_t{
#endif
        __other.__storage().__enable_psa_conversion()
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      }
#else
      }})
#endif
  {
    /* TODO: precondition check
     * __other.extent(r) equals Er for each r for which Er is a static extent, and
     * either
     *   - sizeof...(_OtherExtents) is zero, or
     *   - __other.extent(r) is a representable value of type index_type for all rank index r of __other
     */
  }

#ifdef __NVCC__
    __MDSPAN_TEMPLATE_REQUIRES(
    class... _Integral,
    /* requires */ (
      // TODO: check whether the other version works with newest NVCC, doesn't with 11.4
      // NVCC seems to pick up rank_dynamic from the wrong extents type???
      __MDSPAN_FOLD_AND(_LIBCUDACXX_TRAIT(_CUDA_VSTD::is_convertible, _Integral, index_type) /* && ... */) &&
      __MDSPAN_FOLD_AND(_LIBCUDACXX_TRAIT(_CUDA_VSTD::is_nothrow_constructible, index_type, _Integral) /* && ... */) &&
      // NVCC chokes on the fold thingy here so wrote the workaround
      ((sizeof...(_Integral) == __detail::__count_dynamic_extents<_Extents...>::val) ||
       (sizeof...(_Integral) == sizeof...(_Extents)))
      )
    )
#else
    __MDSPAN_TEMPLATE_REQUIRES(
    class... _Integral,
    /* requires */ (
       __MDSPAN_FOLD_AND(_LIBCUDACXX_TRAIT(_CUDA_VSTD::is_convertible, _Integral, index_type) /* && ... */) &&
       __MDSPAN_FOLD_AND(_LIBCUDACXX_TRAIT(_CUDA_VSTD::is_nothrow_constructible, index_type, _Integral) /* && ... */) &&
       ((sizeof...(_Integral) == rank_dynamic()) || (sizeof...(_Integral) == rank()))
      )
    )
#endif
  __MDSPAN_INLINE_FUNCTION
  explicit constexpr extents(_Integral... __exts) noexcept
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
    : __storage_{
#else
    : __base_t(__base_t{typename __base_t::__stored_type{
#endif
      _CUDA_VSTD::conditional_t<sizeof...(_Integral)==rank_dynamic(),
        __detail::__construct_psa_from_dynamic_exts_values_tag_t,
        __detail::__construct_psa_from_all_exts_values_tag_t>(),
        static_cast<index_type>(__exts)...
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      }
#else
      }})
#endif
  {
    /* TODO: precondition check
     * If sizeof...(_IndexTypes) != rank_dynamic() is true, exts_arr[r] equals Er for each r for which Er is a static extent, and
     * either
     *   - sizeof...(__exts) == 0 is true, or
     *   - each element of __exts is nonnegative and is a representable value of type index_type.
     */
  }

    // TODO: check whether this works with newest NVCC, doesn't with 11.4
#ifdef __NVCC__
  // NVCC seems to pick up rank_dynamic from the wrong extents type???
  // NVCC chokes on the fold thingy here so wrote the workaround
  __MDSPAN_TEMPLATE_REQUIRES(
    class _IndexType, size_t _Np,
    /* requires */ (
      _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_convertible, _IndexType, index_type) &&
      _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_nothrow_constructible, index_type, _IndexType) &&
      ((_Np == __detail::__count_dynamic_extents<_Extents...>::val) ||
       (_Np == sizeof...(_Extents)))
    )
  )
#else
    __MDSPAN_TEMPLATE_REQUIRES(
        class _IndexType, size_t _Np,
        /* requires */ (
          _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_convertible, _IndexType, index_type) &&
          _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_nothrow_constructible, index_type, _IndexType) &&
          (_Np == rank() || _Np == rank_dynamic())
    )
  )
#endif
  __MDSPAN_CONDITIONAL_EXPLICIT(_Np != rank_dynamic())
  __MDSPAN_INLINE_FUNCTION
  constexpr
  extents(_CUDA_VSTD::array<_IndexType, _Np> const& __exts) noexcept
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
    : __storage_{
#else
    : __base_t(__base_t{typename __base_t::__stored_type{
#endif
      _CUDA_VSTD::conditional_t<_Np==rank_dynamic(),
        __detail::__construct_psa_from_dynamic_exts_array_tag_t<0>,
        __detail::__construct_psa_from_all_exts_array_tag_t>(),
      _CUDA_VSTD::array<_IndexType,_Np>{__exts}
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      }
#else
      }})
#endif
  {
    /* TODO: precondition check
     * If _Np != rank_dynamic() is true, __exts[r] equals Er for each r for which Er is a static extent, and
     * either
     *   - _Np is zero, or
     *   - __exts[r] is nonnegative and is a representable value of type index_type for all rank index r
     */
  }

  // TODO: check whether the below works with newest NVCC, doesn't with 11.4
#ifdef __NVCC__
  // NVCC seems to pick up rank_dynamic from the wrong extents type???
  // NVCC chokes on the fold thingy here so wrote the workaround
  __MDSPAN_TEMPLATE_REQUIRES(
    class _IndexType, size_t _Np,
    /* requires */ (
      _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_convertible, _IndexType, index_type) &&
      _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_nothrow_constructible, index_type, _IndexType) &&
      ((_Np == __detail::__count_dynamic_extents<_Extents...>::val) ||
       (_Np == sizeof...(_Extents)))
    )
  )
#else
    __MDSPAN_TEMPLATE_REQUIRES(
        class _IndexType, size_t _Np,
        /* requires */ (
          _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_convertible, _IndexType, index_type) &&
          _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_nothrow_constructible, index_type, _IndexType) &&
          (_Np == rank() || _Np == rank_dynamic())
    )
  )
#endif
  __MDSPAN_CONDITIONAL_EXPLICIT(_Np != rank_dynamic())
  __MDSPAN_INLINE_FUNCTION
  constexpr
  extents(_CUDA_VSTD::span<_IndexType, _Np> __exts) noexcept
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
    : __storage_{
#else
    : __base_t(__base_t{typename __base_t::__stored_type{
#endif
      _CUDA_VSTD::conditional_t<_Np==rank_dynamic(),
        __detail::__construct_psa_from_dynamic_exts_array_tag_t<0>,
        __detail::__construct_psa_from_all_exts_array_tag_t>(),
      __exts
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      }
#else
      }})
#endif
  {
    /* TODO: precondition check
     * If _Np != rank_dynamic() is true, __exts[r] equals Er for each r for which Er is a static extent, and
     * either
     *   - _Np is zero, or
     *   - __exts[r] is nonnegative and is a representable value of type index_type for all rank index r
     */
  }

  // Need this constructor for some submdspan implementation stuff
  // for the layout_stride case where I use an extents object for strides
  __MDSPAN_INLINE_FUNCTION
  constexpr explicit
  extents(__storage_t const& __sto ) noexcept
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
    : __storage_{
#else
    : __base_t(__base_t{
#endif
        __sto
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      }
#else
      })
#endif
  { }

  //--------------------------------------------------------------------------------

  __MDSPAN_INLINE_FUNCTION
  static constexpr
  size_t static_extent(size_t __n) noexcept {
    // Can't do assert here since that breaks true constexpr ness
    // assert(__n<rank());
    return _static_extent_impl(__n, _CUDA_VSTD::make_integer_sequence<size_t, sizeof...(_Extents)>{});
  }

  __MDSPAN_INLINE_FUNCTION
  constexpr
  index_type extent(size_t __n) const noexcept {
    // Can't do assert here since that breaks true constexpr ness
    // assert(__n<rank());
    return __storage().__get(__n);
  }

  //--------------------------------------------------------------------------------

  template<class _OtherIndexType, size_t... _RHS>
  __MDSPAN_INLINE_FUNCTION
  friend constexpr bool operator==(extents const& lhs, extents<_OtherIndexType, _RHS...> const& __rhs) noexcept {
    return lhs._eq_impl(
      __rhs, integral_constant<bool, (sizeof...(_RHS) == rank())>{},
      _CUDA_VSTD::make_index_sequence<sizeof...(_RHS)>{}
    );
  }

#if !(__MDSPAN_HAS_CXX_20)
  template<class _OtherIndexType, size_t... _RHS>
  __MDSPAN_INLINE_FUNCTION
  friend constexpr bool operator!=(extents const& lhs, extents<_OtherIndexType, _RHS...> const& __rhs) noexcept {
    return lhs._not_eq_impl(
      __rhs, integral_constant<bool, (sizeof...(_RHS) == rank())>{},
      _CUDA_VSTD::make_index_sequence<sizeof...(_RHS)>{}
    );
  }
#endif

  // End of public interface

public:  // (but not really)

  __MDSPAN_INLINE_FUNCTION static constexpr
  extents __make_extents_impl(__detail::__partially_static_sizes<index_type, size_t,_Extents...>&& __bs) noexcept {
    // This effectively amounts to a sideways cast that can be done in a constexpr
    // context, but we have to do it to handle the case where the extents and the
    // strides could accidentally end up with the same types in their hierarchies
    // somehow (which would cause layout_stride::mapping to not be standard_layout)
    return extents(
#if !defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      __base_t{
#endif
        _CUDA_VSTD::move(__bs.template __with_tag<__detail::__extents_tag>())
#if !defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      }
#endif
    );
  }

  template <size_t _Np>
  __MDSPAN_FORCE_INLINE_FUNCTION
  constexpr
  index_type __extent() const noexcept {
    return __storage().template __get_n<_Np>();
  }

  template <size_t _Np, size_t _Default=dynamic_extent>
  __MDSPAN_INLINE_FUNCTION
  static constexpr
  index_type __static_extent() noexcept {
    return __storage_t::template __get_static_n<_Np, _Default>();
  }

};

namespace __detail {

template <class _IndexType, size_t _Rank, class _Extents = _CUDA_VSTD::extents<_IndexType>>
struct __make_dextents;

template <class _IndexType, size_t _Rank, size_t... _ExtentsPack>
struct __make_dextents<_IndexType, _Rank, _CUDA_VSTD::extents<_IndexType, _ExtentsPack...>> {
  using type = typename __make_dextents<_IndexType, _Rank - 1,
    _CUDA_VSTD::extents<_IndexType, _CUDA_VSTD::dynamic_extent, _ExtentsPack...>>::type;
};

template <class _IndexType, size_t... _ExtentsPack>
struct __make_dextents<_IndexType, 0, _CUDA_VSTD::extents<_IndexType, _ExtentsPack...>> {
  using type = _CUDA_VSTD::extents<_IndexType, _ExtentsPack...>;
};

} // end namespace __detail

template <class _IndexType, size_t _Rank>
using dextents = typename __detail::__make_dextents<_IndexType, _Rank>::type;

#if defined(__MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
template <class... _IndexTypes>
extents(_IndexTypes...)
  // Workaround for nvcc
  //-> extents<size_t, __detail::__make_dynamic_extent<_IndexTypes>()...>;
  // Adding "(void)" so that clang doesn't complain this is unused
  -> extents<size_t, size_t(((void)_IndexTypes(), -1))...>;
#endif

namespace __detail {

template <class _Tp>
struct __is_extents : false_type {};

template <class _IndexType, size_t... _ExtentsPack>
struct __is_extents<_CUDA_VSTD::extents<_IndexType, _ExtentsPack...>> : true_type {};

template <class _Tp>
static constexpr bool __is_extents_v = __is_extents<_Tp>::value;


template <typename _Extents>
struct __extents_to_partially_static_sizes;

template <class _IndexType, size_t... _ExtentsPack>
struct __extents_to_partially_static_sizes<_CUDA_VSTD::extents<_IndexType, _ExtentsPack...>> {
  using type = __detail::__partially_static_sizes<
          typename _CUDA_VSTD::extents<_IndexType, _ExtentsPack...>::index_type, size_t,
          _ExtentsPack...>;
};

template <typename _Extents>
using __extents_to_partially_static_sizes_t = typename __extents_to_partially_static_sizes<_Extents>::type;

} // end namespace __detail

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#if defined(_LIBCUDACXX_POP_MACROS)
  _LIBCUDACXX_POP_MACROS
#endif

#endif // _LIBCUDACXX___MDSPAN_EXTENTS_HPP
