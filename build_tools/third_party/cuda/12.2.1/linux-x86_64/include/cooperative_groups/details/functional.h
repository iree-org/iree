 /* Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to NVIDIA intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  * conditions of a form of NVIDIA software license agreement by and
  * between NVIDIA and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of NVIDIA is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  * OF THESE LICENSED DELIVERABLES.
  *
  * U.S. Government End Users.  These Licensed Deliverables are a
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  * 1995), consisting of "commercial computer software" and "commercial
  * computer software documentation" as such terms are used in 48
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  * U.S. Government End Users acquire the Licensed Deliverables with
  * only those rights set forth herein.
  *
  * Any use of the Licensed Deliverables in individual and commercial
  * software must include, in the user documentation and internal
  * comments to the code, the above Disclaimer and U.S. Government End
  * Users Notice.
  */

#ifndef _CG_FUNCTIONAL_H
#define _CG_FUNCTIONAL_H

#include "info.h"
#include "helpers.h"

#ifdef _CG_CPP11_FEATURES
#ifdef _CG_USE_CUDA_STL
# include <cuda/std/functional>
#endif

_CG_BEGIN_NAMESPACE

namespace details {
#ifdef _CG_USE_CUDA_STL
    using cuda::std::plus;
    using cuda::std::bit_and;
    using cuda::std::bit_xor;
    using cuda::std::bit_or;
#else
    template <typename Ty> struct plus {__device__ __forceinline__ Ty operator()(Ty arg1, Ty arg2) const {return arg1 + arg2;}};
    template <typename Ty> struct bit_and {__device__ __forceinline__ Ty operator()(Ty arg1, Ty arg2) const {return arg1 & arg2;}};
    template <typename Ty> struct bit_xor {__device__ __forceinline__ Ty operator()(Ty arg1, Ty arg2) const {return arg1 ^ arg2;}};
    template <typename Ty> struct bit_or {__device__ __forceinline__ Ty operator()(Ty arg1, Ty arg2) const {return arg1 | arg2;}};
#endif // _CG_USE_PLATFORM_STL
} // details

template <typename Ty>
struct plus : public details::plus<Ty> {};

template <typename Ty>
struct less {
    __device__ __forceinline__ Ty operator()(Ty arg1, Ty arg2) const {
        return (arg2 < arg1) ? arg2 : arg1;
    }
};

template <typename Ty>
struct greater {
    __device__ __forceinline__ Ty operator()(Ty arg1, Ty arg2) const {
        return (arg1 < arg2) ? arg2 : arg1;
    }
};

template <typename Ty>
struct bit_and : public details::bit_and<Ty> {};

template <typename Ty>
struct bit_xor : public details::bit_xor<Ty> {};

template <typename Ty>
struct bit_or : public details::bit_or<Ty> {};

#if defined(_CG_HAS_STL_ATOMICS)
namespace details {
    template <class Ty>
    using _atomic_is_type_supported = _CG_STL_NAMESPACE::integral_constant<bool,
            _CG_STL_NAMESPACE::is_integral<Ty>::value && (sizeof(Ty) == 4 || sizeof(Ty) == 8)>;

    template <typename TyOp> struct _atomic_op_supported                                : public _CG_STL_NAMESPACE::false_type {};
    template <typename Ty> struct _atomic_op_supported<cooperative_groups::plus<Ty>>    : public _atomic_is_type_supported<Ty> {};
    template <typename Ty> struct _atomic_op_supported<cooperative_groups::less<Ty>>    : public _atomic_is_type_supported<Ty> {};
    template <typename Ty> struct _atomic_op_supported<cooperative_groups::greater<Ty>> : public _atomic_is_type_supported<Ty> {};
    template <typename Ty> struct _atomic_op_supported<cooperative_groups::bit_and<Ty>> : public _atomic_is_type_supported<Ty> {};
    template <typename Ty> struct _atomic_op_supported<cooperative_groups::bit_or<Ty>>  : public _atomic_is_type_supported<Ty> {};
    template <typename Ty> struct _atomic_op_supported<cooperative_groups::bit_xor<Ty>> : public _atomic_is_type_supported<Ty> {};

    template<typename TyAtomic, typename TyVal, typename TyOp>
    _CG_QUALIFIER remove_qual<TyVal> atomic_cas_fallback(TyAtomic&& atomic, TyVal&& val, TyOp&& op) {
        auto old = atomic.load(cuda::std::memory_order_relaxed);
        while(!atomic.compare_exchange_weak(old, op(old, val), cuda::std::memory_order_relaxed));
        return old;
    }

    template<typename TyOp>
    struct op_picker;

    template<typename TyVal>
    struct op_picker<cooperative_groups::plus<TyVal>> {
        template<typename TyAtomic>
        _CG_STATIC_QUALIFIER TyVal atomic_update(TyAtomic& atomic, TyVal val) {
            return atomic.fetch_add(val, cuda::std::memory_order_relaxed);
        }
    };

    template<typename TyVal>
    struct op_picker<cooperative_groups::less<TyVal>> {
        template<typename TyAtomic>
        _CG_STATIC_QUALIFIER TyVal atomic_update(TyAtomic& atomic, TyVal val) {
            return atomic.fetch_min(val, cuda::std::memory_order_relaxed);
        }
    };

    template<typename TyVal>
    struct op_picker<cooperative_groups::greater<TyVal>> {
        template<typename TyAtomic>
        _CG_STATIC_QUALIFIER TyVal atomic_update(TyAtomic& atomic, TyVal val) {
            return atomic.fetch_max(val, cuda::std::memory_order_relaxed);
        }
    };

    template<typename TyVal>
    struct op_picker<cooperative_groups::bit_and<TyVal>> {
        template<typename TyAtomic>
        _CG_STATIC_QUALIFIER TyVal atomic_update(TyAtomic& atomic, TyVal val) {
            return atomic.fetch_and(val, cuda::std::memory_order_relaxed);
        }
    };

    template<typename TyVal>
    struct op_picker<cooperative_groups::bit_xor<TyVal>> {
        template<typename TyAtomic>
        _CG_STATIC_QUALIFIER TyVal atomic_update(TyAtomic& atomic, TyVal val) {
            return atomic.fetch_xor(val, cuda::std::memory_order_relaxed);
        }
    };

    template<typename TyVal>
    struct op_picker<cooperative_groups::bit_or<TyVal>> {
        template<typename TyAtomic>
        _CG_STATIC_QUALIFIER TyVal atomic_update(TyAtomic& atomic, TyVal val) {
            return atomic.fetch_or(val, cuda::std::memory_order_relaxed);
        }
    };

    template<bool atomic_supported>
    struct atomic_update_dispatch {};

    template<>
    struct atomic_update_dispatch<false> {
        template<typename TyAtomic, typename TyVal, typename TyOp>
        _CG_STATIC_QUALIFIER remove_qual<TyVal> atomic_update(TyAtomic& atomic, TyVal&& val, TyOp&& op) {
            return atomic_cas_fallback(atomic, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyOp>(op));
        }
    };

    template<>
    struct atomic_update_dispatch<true> {
        template<typename TyAtomic, typename TyVal, typename TyOp>
        _CG_STATIC_QUALIFIER TyVal atomic_update(TyAtomic& atomic, TyVal val, TyOp&& op) {
            using dispatch = op_picker<details::remove_qual<TyOp>>;

            return dispatch::atomic_update(atomic, val);
        }
    };

    template<typename TyAtomic, typename TyVal, typename TyOp>
    _CG_QUALIFIER remove_qual<TyVal> atomic_update(TyAtomic& atomic, TyVal&& val, TyOp&& op) {
        using dispatch = atomic_update_dispatch<_atomic_op_supported<details::remove_qual<TyOp>>::value>;

        return dispatch::atomic_update(atomic, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyOp>(op));
    }

    template<typename TyAtomic, typename TyVal>
    _CG_QUALIFIER void atomic_store(TyAtomic& atomic, TyVal&& val) {
        atomic.store(val, cuda::std::memory_order_relaxed);
    }
}
#endif

_CG_END_NAMESPACE

#endif
#endif //_CG_FUNCTIONAL_H
