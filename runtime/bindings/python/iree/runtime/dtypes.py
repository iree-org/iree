import dataclasses
import numpy as np
import ml_dtypes
from typing import Any

from ._binding import HalElementType


@dataclasses.dataclass(frozen=True)
class DTypeInfo:
    dtype: np.dtype
    name: str
    abi_type: str
    hal_type: HalElementType

    @property
    def is_complex(self) -> bool:
        return np.issubdtype(self.dtype, np.complexfloating)

    @property
    def is_floating_point(self) -> bool:
        return np.issubdtype(self.dtype, np.floating)

    @property
    def is_integer(self) -> bool:
        return np.issubdtype(self.dtype, np.integer)

    @property
    def is_signed(self) -> bool:
        return np.issubdtype(self.dtype, np.signedinteger)


_ML_DTYPE_TO_HAL_ELEMENT_TYPE = {
    np.dtype(ml_dtypes.bfloat16): HalElementType.BFLOAT_16,
    np.dtype(ml_dtypes.float8_e4m3fn): HalElementType.FLOAT_8_E4M3_FN,
    np.dtype(ml_dtypes.float8_e4m3fnuz): HalElementType.FLOAT_8_E4M3_FNUZ,
    np.dtype(ml_dtypes.float8_e5m2): HalElementType.FLOAT_8_E5M2,
    np.dtype(ml_dtypes.float8_e5m2fnuz): HalElementType.FLOAT_8_E5M2_FNUZ,
    np.dtype(ml_dtypes.float8_e8m0fnu): HalElementType.FLOAT_8_E8M0_FNU,
}


_DTYPE_TO_INFO = {
    info.dtype: info
    for info in [
        # Float
        DTypeInfo(np.dtype(np.float16), "float16", "f16", HalElementType.FLOAT_16),
        DTypeInfo(np.dtype(np.float32), "float32", "f32", HalElementType.FLOAT_32),
        DTypeInfo(np.dtype(np.float64), "float64", "f64", HalElementType.FLOAT_64),
        # Integer
        DTypeInfo(np.dtype(np.int8), "int8", "i8", HalElementType.SINT_8),
        DTypeInfo(np.dtype(np.uint8), "uint8", "i8", HalElementType.UINT_8),
        DTypeInfo(np.dtype(np.int16), "int16", "i16", HalElementType.SINT_16),
        DTypeInfo(np.dtype(np.uint16), "uint16", "i16", HalElementType.UINT_16),
        DTypeInfo(np.dtype(np.int32), "int32", "i32", HalElementType.SINT_32),
        DTypeInfo(np.dtype(np.uint32), "uint32", "i32", HalElementType.UINT_32),
        DTypeInfo(np.dtype(np.int64), "int64", "i64", HalElementType.SINT_64),
        DTypeInfo(np.dtype(np.uint64), "uint64", "i64", HalElementType.UINT_64),
        # Boolean
        DTypeInfo(np.dtype(np.bool_), "bool", "i1", HalElementType.BOOL_8),
        # Complex
        DTypeInfo(
            np.dtype(np.complex64),
            "complex64",
            "complex<f32>",
            HalElementType.COMPLEX_64,
        ),
        DTypeInfo(
            np.dtype(np.complex128),
            "complex128",
            "complex<f64>",
            HalElementType.COMPLEX_128,
        ),
    ]
}

_NAME_TO_INFO: dict[str, DTypeInfo] = {
    info.name: info for info in _DTYPE_TO_INFO.values()
}

_ABI_TYPE_TO_INFO: dict[str, DTypeInfo] = {
    info.abi_type: info
    for info in _DTYPE_TO_INFO.values()
    if not info.is_integer
    or info.is_signed  # Exclude unsigned integers since they share ABI types with signed integers.
}


def map_dtype_to_dtype_info(dtype: np.dtype | type) -> DTypeInfo:
    dtype = np.dtype(dtype)
    try:
        return _DTYPE_TO_INFO[dtype]
    except KeyError:
        raise KeyError(f"Numpy dtype {dtype} not found.") from None


def map_name_to_dtype_info(name: str) -> DTypeInfo:
    try:
        return _NAME_TO_INFO[name]
    except KeyError:
        raise KeyError(f"Numpy dtype name {name!r} not found.") from None


def map_abi_type_to_dtype_info(abi_type: str) -> DTypeInfo:
    try:
        return _ABI_TYPE_TO_INFO[abi_type]
    except KeyError:
        raise KeyError(f"ABI type {abi_type!r} not found.") from None


def map_dtype_to_hal_element_type(dtype: Any) -> HalElementType | None:
    try:
        normalized_dtype = np.dtype(dtype)
    except TypeError:
        return None
    elem_ty = _ML_DTYPE_TO_HAL_ELEMENT_TYPE.get(normalized_dtype)
    if elem_ty is not None:
        return elem_ty
    try:
        return map_dtype_to_dtype_info(normalized_dtype).hal_type
    except KeyError:
        return None


DTYPE_TO_ABI_TYPE: dict[np.dtype, str] = {
    info.dtype: info.abi_type for info in _DTYPE_TO_INFO.values()
}
