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

#ifndef _LIBCUDACXX___MDSPAN_LAYOUT_RIGHT_HPP
#define _LIBCUDACXX___MDSPAN_LAYOUT_RIGHT_HPP

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__assert"
#include "../__mdspan/extents.h"
#include "../__mdspan/layout_stride.h"
#include "../__mdspan/macros.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_convertible.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__utility/integer_sequence.h"
#include "../cstddef"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 11

//==============================================================================
template <class _Extents>
class layout_right::mapping {
  public:
    using extents_type = _Extents;
    using index_type = typename extents_type::index_type;
    using size_type = typename extents_type::size_type;
    using rank_type = typename extents_type::rank_type;
    using layout_type = layout_right;
  private:

    static_assert(__detail::__is_extents_v<extents_type>, "layout_right::mapping must be instantiated with a specialization of _CUDA_VSTD::extents.");

    template <class>
    friend class mapping;

    // i0+(i1 + E(1)*(i2 + E(2)*i3))
    template <size_t _r, size_t _Rank>
    struct __rank_count {};

    template <size_t _r, size_t _Rank, class _Ip, class... _Indices>
    __MDSPAN_HOST_DEVICE
    constexpr index_type __compute_offset(
      index_type __offset, __rank_count<_r,_Rank>, const _Ip& __i, _Indices... __idx) const {
      return __compute_offset(__offset * __extents.template __extent<_r>() + __i,__rank_count<_r+1,_Rank>(),  __idx...);
    }

    template<class _Ip, class ... _Indices>
    __MDSPAN_HOST_DEVICE
    constexpr index_type __compute_offset(
      __rank_count<0,extents_type::rank()>, const _Ip& __i, _Indices... __idx) const {
      return __compute_offset(__i,__rank_count<1,extents_type::rank()>(),__idx...);
    }

    __MDSPAN_HOST_DEVICE
    constexpr index_type __compute_offset(size_t __offset, __rank_count<extents_type::rank(), extents_type::rank()>) const {
      return static_cast<index_type>(__offset);
    }

    __MDSPAN_HOST_DEVICE
    constexpr index_type __compute_offset(__rank_count<0,0>) const { return 0; }

  public:

    //--------------------------------------------------------------------------------

    __MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping() noexcept = default;
    __MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping(mapping const&) noexcept = default;

    __MDSPAN_HOST_DEVICE
    constexpr mapping(extents_type const& __exts) noexcept
      :__extents(__exts)
    { }

    __MDSPAN_TEMPLATE_REQUIRES(
      class _OtherExtents,
      /* requires */ (
        _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_constructible, extents_type, _OtherExtents)
      )
    )
    __MDSPAN_CONDITIONAL_EXPLICIT((!_CUDA_VSTD::is_convertible<_OtherExtents, extents_type>::value)) // needs two () due to comma
    __MDSPAN_INLINE_FUNCTION constexpr
    mapping(mapping<_OtherExtents> const& __other) noexcept // NOLINT(google-explicit-constructor)
      :__extents(__other.extents())
    {
       /*
        * TODO: check precondition
        * __other.required_span_size() is a representable value of type index_type
        */
    }

    __MDSPAN_TEMPLATE_REQUIRES(
      class _OtherExtents,
      /* requires */ (
        _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_constructible, extents_type, _OtherExtents) &&
        (extents_type::rank() <= 1)
      )
    )
    __MDSPAN_CONDITIONAL_EXPLICIT((!_CUDA_VSTD::is_convertible<_OtherExtents, extents_type>::value)) // needs two () due to comma
    __MDSPAN_INLINE_FUNCTION constexpr
    mapping(layout_left::mapping<_OtherExtents> const& __other) noexcept // NOLINT(google-explicit-constructor)
      :__extents(__other.extents())
    {
       /*
        * TODO: check precondition
        * __other.required_span_size() is a representable value of type index_type
        */
    }

    __MDSPAN_TEMPLATE_REQUIRES(
      class _OtherExtents,
      /* requires */ (
        _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_constructible, extents_type, _OtherExtents)
      )
    )
    __MDSPAN_CONDITIONAL_EXPLICIT((extents_type::rank() > 0))
    __MDSPAN_INLINE_FUNCTION constexpr
    mapping(layout_stride::mapping<_OtherExtents> const& __other) // NOLINT(google-explicit-constructor)
      :__extents(__other.extents())
    {
       /*
        * TODO: check precondition
        * __other.required_span_size() is a representable value of type index_type
        */
       #ifndef __CUDA_ARCH__
       size_t __stride = 1;
       for(rank_type __r=__extents.rank(); __r>0; __r--) {
         #ifndef _LIBCUDACXX_NO_EXCEPTIONS
         if(__stride != static_cast<size_t>(__other.stride(__r-1)))
           __throw_runtime_error("Assigning layout_stride to layout_right with invalid strides.");
         #else
         _LIBCUDACXX_ASSERT(__stride == static_cast<size_t>(__other.stride(__r-1)), "");
         #endif
         __stride *= __extents.extent(__r-1);
       }
       #endif
    }

    __MDSPAN_INLINE_FUNCTION_DEFAULTED __MDSPAN_CONSTEXPR_14_DEFAULTED mapping& operator=(mapping const&) noexcept = default;

    __MDSPAN_INLINE_FUNCTION
    constexpr const extents_type& extents() const noexcept {
      return __extents;
    }

    __MDSPAN_INLINE_FUNCTION
    constexpr index_type required_span_size() const noexcept {
      index_type __value = 1;
      for(rank_type __r=0; __r != extents_type::rank(); ++__r) __value*=__extents.extent(__r);
      return __value;
    }

    //--------------------------------------------------------------------------------

    __MDSPAN_TEMPLATE_REQUIRES(
      class... _Indices,
      /* requires */ (
        (sizeof...(_Indices) == extents_type::rank()) &&
        __MDSPAN_FOLD_AND(
           (_LIBCUDACXX_TRAIT(_CUDA_VSTD::is_convertible, _Indices, index_type) &&
            _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_nothrow_constructible, index_type, _Indices))
        )
      )
    )
    __MDSPAN_HOST_DEVICE
    constexpr index_type operator()(_Indices... __idxs) const noexcept {
      return __compute_offset(__rank_count<0, extents_type::rank()>(), static_cast<index_type>(__idxs)...);
    }

    __MDSPAN_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept { return true; }
    __MDSPAN_INLINE_FUNCTION static constexpr bool is_always_exhaustive() noexcept { return true; }
    __MDSPAN_INLINE_FUNCTION static constexpr bool is_always_strided() noexcept { return true; }
    __MDSPAN_INLINE_FUNCTION constexpr bool is_unique() const noexcept { return true; }
    __MDSPAN_INLINE_FUNCTION constexpr bool is_exhaustive() const noexcept { return true; }
    __MDSPAN_INLINE_FUNCTION constexpr bool is_strided() const noexcept { return true; }

    __MDSPAN_TEMPLATE_REQUIRES(
      class _Ext = _Extents,
      /* requires */ (
        _Ext::rank() > 0
      )
    )
    __MDSPAN_INLINE_FUNCTION
    constexpr index_type stride(rank_type __i) const noexcept {
      index_type __value = 1;
      for(rank_type __r=extents_type::rank()-1; __r>__i; __r--) __value*=__extents.extent(__r);
      return __value;
    }

    template<class _OtherExtents>
    __MDSPAN_INLINE_FUNCTION
    friend constexpr bool operator==(mapping const& __lhs, mapping<_OtherExtents> const& __rhs) noexcept {
      return __lhs.extents() == __rhs.extents();
    }

    // In C++ 20 the not equal exists if equal is found
#if !(__MDSPAN_HAS_CXX_20)
    template<class _OtherExtents>
    __MDSPAN_INLINE_FUNCTION
    friend constexpr bool operator!=(mapping const& __lhs, mapping<_OtherExtents> const& __rhs) noexcept {
      return __lhs.extents() != __rhs.extents();
    }
#endif

    // Not really public, but currently needed to implement fully constexpr useable submdspan:
    template<size_t _Np, class _SizeType, size_t ... _Ep, size_t ... _Idx>
    __MDSPAN_HOST_DEVICE
    constexpr index_type __get_stride(_CUDA_VSTD::extents<_SizeType, _Ep...>,_CUDA_VSTD::integer_sequence<size_t, _Idx...>) const {
      return __MDSPAN_FOLD_TIMES_RIGHT((_Idx>_Np? __extents.template __extent<_Idx>():1),1);
    }
    template<size_t _Np>
    __MDSPAN_HOST_DEVICE
    constexpr index_type __stride() const noexcept {
      return __get_stride<_Np>(__extents, _CUDA_VSTD::make_index_sequence<extents_type::rank()>());
    }

private:
   __MDSPAN_NO_UNIQUE_ADDRESS extents_type __extents{};

};

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MDSPAN_LAYOUT_RIGHT_HPP
