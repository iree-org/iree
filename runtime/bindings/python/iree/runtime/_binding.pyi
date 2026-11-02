import enum
import os
import numpy as np
from typing import (
    Any,
    Callable,
    ClassVar,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)
import asyncio

from .typing import HalModuleBufferViewTraceCallback

_T = TypeVar("_T")

def create_hal_module(
    instance: VmInstance,
    device: Optional[HalDevice] = None,
    devices: Optional[List[HalDevice]] = None,
    debug_sink: Optional[HalModuleDebugSink] = None,
) -> VmModule: ...
def create_io_parameters_module(
    instance: VmInstance, *providers: ParameterProvider
) -> VmModule: ...
def disable_leak_checker(): ...
def get_cached_hal_driver(device_uri: str) -> HalDriver: ...
def parse_flags(*flag: str): ...

class BufferCompatibility(enum.IntFlag):
    NONE = 0
    ALLOCATABLE = 1
    IMPORTABLE = 2
    EXPORTABLE = 4
    QUEUE_TRANSFER = 1024
    QUEUE_DISPATCH = 2048

class BufferUsage(enum.IntFlag):
    NONE = 0
    TRANSFER_SOURCE = 1
    TRANSFER_TARGET = 2
    TRANSFER = 3
    DISPATCH_INDIRECT_PARAMETERS = 256
    DISPATCH_UNIFORM_READ = 512
    DISPATCH_STORAGE_READ = 1024
    DISPATCH_STORAGE_WRITE = 2048
    DISPATCH_STORAGE = 3072
    DISPATCH_IMAGE_READ = 4096
    DISPATCH_IMAGE_WRITE = 8192
    DISPATCH_IMAGE = 12288
    SHARING_EXPORT = 65536
    SHARING_REPLICATE = 131072
    SHARING_CONCURRENT = 262144
    SHARING_IMMUTABLE = 524288
    MAPPING_SCOPED = 16777216
    MAPPING_PERSISTENT = 33554432
    MAPPING_OPTIONAL = 67108864
    MAPPING_ACCESS_RANDOM = 134217728
    MAPPING_ACCESS_SEQUENTIAL_WRITE = 268435456
    MAPPING = 150994944
    DEFAULT = 3075

class ExternalTimepointType(enum.IntFlag):
    NONE = 0
    ASYNC_PRIMITIVE = 1
    CUDA_EVENT = 2
    HIP_EVENT = 3

class SemaphoreCompatibility(enum.IntFlag):
    NONE = 0
    HOST_WAIT = 1
    DEVICE_WAIT = 2
    HOST_SIGNAL = 4
    DEVICE_SIGNAL = 8
    HOST_ONLY = 5
    DEVICE_ONLY = 10
    ALL = 15

class ExternalTimepointFlags(enum.IntFlag):
    NONE = 0

class _InvokeStatics: ...

_invoke_statics: _InvokeStatics

class InvokeContext:
    def __init__(self, device: HalDevice) -> None: ...

class ArgumentPacker:
    def __init__(
        self, statics: _InvokeStatics, arg_descs: Optional[list[Any]] = None
    ) -> None: ...
    def pack(
        self,
        invoke_context: InvokeContext,
        pos_args: Sequence[Any],
        kwargs: dict[str, Any],
    ) -> VmVariantList: ...

class FileHandle:
    @staticmethod
    def wrap_memory(
        host_buffer: Any, readable: bool = True, writable: bool = False
    ) -> FileHandle: ...
    @staticmethod
    def wrap_fd(
        fd: int, readable: bool = True, writable: bool = False
    ) -> FileHandle: ...
    def host_allocation(self) -> memoryview:
        """Access the raw view of the allocated host memory.

        Requires is_host_allocation.
        """
        ...

    @property
    def is_host_allocation(self) -> bool: ...
    @property
    def is_async(self) -> bool:
        """Whether the handle was opened for asynchronous I/O."""
        ...

class HalAllocator:
    def allocate_buffer(
        self,
        memory_type: Union[MemoryType, int],
        allowed_usage: Union[BufferUsage, int],
        allocation_size: int,
    ) -> HalBuffer: ...
    def allocate_buffer_copy(
        self,
        memory_type: Union[MemoryType, int],
        allowed_usage: Union[BufferUsage, int],
        device: HalDevice,
        buffer: object,
        element_type: Optional[HalElementType] = ...,
    ) -> Union[HalBuffer, HalBufferView]: ...
    def allocate_host_staging_buffer_copy(
        self, device: HalDevice, initial_contents: object
    ) -> HalBuffer: ...
    def import_host_allocation(
        self,
        buffer_obj: Any,
        memory_type: Union[MemoryType, int],
        allowed_usage: Union[BufferUsage, int],
    ) -> HalBuffer: ...
    def query_buffer_compatibility(
        self,
        memory_type: Union[MemoryType, int],
        allowed_usage: Union[BufferUsage, int],
        intended_usage: Union[BufferUsage, int],
        allocation_size: int,
    ) -> int: ...
    def trim(self) -> None: ...
    @property
    def formatted_statistics(self) -> str: ...
    @property
    def has_statistics(self) -> bool: ...
    @property
    def statistics(self) -> dict: ...

class HalBuffer:
    def allowed_usage(self) -> BufferUsage: ...
    def byte_length(self) -> int: ...
    def create_view(self, shape: Shape, element_size: int) -> HalBufferView: ...
    def fill_zero(self, byte_offset: int, byte_length: int) -> None: ...
    def map(self) -> MappedMemory: ...
    def memory_type(self) -> int: ...

class HalBufferView:
    def __init__(
        self,
        buffer: HalBuffer,
        shape: Sequence[int],
        element_type: Union[HalElementType, int],
    ) -> None: ...
    def get_buffer(self) -> HalBuffer: ...
    def map(self) -> MappedMemory: ...
    def __eq__(self, other) -> bool: ...
    @property
    def element_type(self) -> HalElementType: ...
    @property
    def ref(self) -> VmRef: ...
    @property
    def shape(self) -> list[int]: ...
    @property
    def byte_length(self) -> int: ...
    @property
    def __iree_vm_ref__(self) -> VmRef: ...

class HalCommandBuffer:
    def __init__(
        self, device: HalDevice, binding_capacity: int = 0, begin: bool = True
    ) -> None: ...
    def begin(self) -> None: ...
    def copy(
        self,
        source_buffer: HalBuffer,
        target_buffer: HalBuffer,
        source_offset: int = 0,
        target_offset: int = 0,
        length: Optional[int] = None,
        end: bool = False,
    ) -> None: ...
    def end(self) -> None: ...
    def fill(
        self,
        target_buffer: HalBuffer,
        pattern: object,
        target_offset: int = 0,
        length: Optional[int] = None,
        end: bool = False,
    ) -> None: ...
    def __eq__(self, other) -> bool: ...
    @property
    def ref(self) -> VmRef: ...
    @property
    def __iree_vm_ref__(self) -> VmRef: ...

HalSemaphoreList = List[Tuple[HalSemaphore, int]] | HalFence

class HalDevice:
    def begin_profiling(self, mode: Optional[str] = None) -> None: ...
    def begin_external_capture(
        self,
        provider: str,
        file_path: Optional[str] = None,
        label: Optional[str] = None,
    ) -> None: ...
    def create_semaphore(self, initial_value: int) -> HalSemaphore: ...
    def end_external_capture(self) -> None: ...
    def end_profiling(self) -> None: ...
    def flush_profiling(self) -> None: ...
    def queue_alloca(
        self,
        allocation_size: int,
        wait_semaphores: HalSemaphoreList,
        signal_semaphores: HalSemaphoreList,
    ) -> HalBuffer: ...
    def queue_copy(
        self,
        source_buffer: HalBuffer,
        target_buffer: HalBuffer,
        wait_semaphores: HalSemaphoreList,
        signal_semaphores: HalSemaphoreList,
    ) -> None: ...
    def queue_dealloca(
        self,
        buffer: HalBuffer,
        wait_semaphores: HalSemaphoreList,
        signal_semaphores: HalSemaphoreList,
    ) -> None: ...
    def queue_execute(
        self,
        command_buffer: HalCommandBuffer,
        wait_semaphores: HalSemaphoreList,
        signal_semaphores: HalSemaphoreList,
    ) -> None: ...
    @property
    def allocator(self) -> HalAllocator: ...
    def create_dlpack_capsule(
        self, buffer_view: HalBufferView, device_type_code: int, device_id: int
    ) -> Any:
        """Creates a DLPack capsule for the given buffer view with an explicit
        device type code and id.

        Note that IREE's HAL hierarchy does not match 1:1 with DLPack's view of
        the world, which combines synchronization, memory access, and device
        identity. As such, we presume that some higher level Python class will
        represent "framework tensors" for interop in a way that interops
        completely. Such an implementation would delegate to this facility
        to create the low level capsule, having explicitly mapped the devices
        and placed appropriate synchronization guards.
        """
        ...

    def from_dlpack_capsule(self, capsule: Any) -> HalBufferView:
        """Imports a DLPack tensor capsule and returns a corresponding buffer
        view.

        As with create_dlpack_capsule, this is just a partial implementation
        of concerns and presumes that the caller has chosen the correct
        device and placed any synchronization barriers necessary for actually
        using the result.
        """
        ...

class HalDeviceLoopBridge:
    def __init__(self, device: HalDevice, loop: asyncio.BaseEventLoop): ...
    def stop(self): ...
    def on_semaphore(
        self, semaphore: HalSemaphore, payload: int, value: Any
    ) -> asyncio.Future: ...

class HalDriver:
    @staticmethod
    def query() -> List[str]: ...
    def create_default_device(
        self, allocators: Optional[list[HalAllocator]] = None
    ) -> HalDevice: ...
    @overload
    def create_device(
        self,
        device_id: int,
        params: Optional[dict],
        allocators: Optional[list[HalAllocator]] = None,
    ) -> HalDevice: ...
    @overload
    def create_device(
        self,
        device_info: dict,
        params: Optional[dict],
        allocators: Optional[list[HalAllocator]] = None,
    ) -> HalDevice: ...
    def create_device_by_uri(
        self, device_uri: str, allocators: Optional[list[HalAllocator]] = None
    ) -> HalDevice: ...
    def query_available_devices(self) -> list[dict]: ...

class HalElementType(enum.IntEnum):
    @staticmethod
    def map_to_dtype(element_type: HalElementType) -> Any: ...
    @staticmethod
    def is_byte_aligned(element_type: HalElementType) -> bool: ...
    @staticmethod
    def dense_byte_count(element_type: HalElementType) -> int: ...

    NONE = 0
    OPAQUE_8 = 8
    OPAQUE_16 = 16
    OPAQUE_32 = 32
    OPAQUE_64 = 64
    BOOL_8 = 318767112
    INT_4 = 268435460
    INT_8 = 268435464
    INT_16 = 268435472
    INT_32 = 268435488
    INT_64 = 268435520
    SINT_4 = 285212676
    SINT_8 = 285212680
    SINT_16 = 285212688
    SINT_32 = 285212704
    SINT_64 = 285212736
    UINT_4 = 301989892
    UINT_8 = 301989896
    UINT_16 = 301989904
    UINT_32 = 301989920
    UINT_64 = 301989952
    FLOAT_16 = 553648144
    FLOAT_32 = 553648160
    FLOAT_64 = 553648192
    BFLOAT_16 = 570425360
    COMPLEX_64 = 587202624
    COMPLEX_128 = 587202688
    FLOAT_8_E4M3_FN = 620757000
    FLOAT_8_E4M3_FNUZ = 654311432
    FLOAT_8_E5M2 = 603979784
    FLOAT_8_E5M2_FNUZ = 637534216
    FLOAT_8_E8M0_FNU = 671088648

class HalFence:
    @staticmethod
    def create_at(sem: HalSemaphore, value: int) -> HalFence: ...
    @staticmethod
    def join(fences: Sequence[HalFence]) -> HalFence: ...
    def __init__(self, capacity: int) -> None: ...
    def extend(self, from_fence: HalFence) -> None: ...
    def fail(self, message: str) -> None: ...
    def insert(self, sem: HalSemaphore, value: int) -> None: ...
    def signal(self) -> None: ...
    def wait(
        self, timeout: Optional[int] = None, deadline: Optional[int] = None
    ) -> None: ...
    @property
    def ref(self) -> VmRef: ...
    @property
    def timepoint_count(self) -> int: ...
    @property
    def __iree_vm_ref__(self) -> VmRef: ...

class HalSemaphore:
    def fail(self, message: str): ...
    def query(self) -> int: ...
    def signal(self, new_value: int) -> None: ...
    def wait(
        self,
        payload: int,
        timeout: Optional[int] = None,
        deadline: Optional[int] = None,
    ) -> None: ...
    def import_timepoint(self, value: int, timepoint: HalExternalTimepoint) -> None: ...
    def export_timepoint(
        self,
        value: int,
        requested_type: Union[ExternalTimepointType, int],
        requested_flags: Union[ExternalTimepointFlags, int],
        out_external_timepoint: HalExternalTimepoint,
    ) -> None: ...

class HalModuleDebugSink:
    def __init__(
        self, buffer_view_trace_callback: Optional[HalModuleBufferViewTraceCallback]
    ):
        """The function object buffer_view_trace_callback must not include the
        corresponding HAL VmModule, VmInstance or VmContext in its closure.
        Native runtime objects are managed by reference counting and do not track
        cyclic references. This will create an uncollectible cycle.
        E.g.

        ```
        vm_context = ...
        def callback(key, hal_buffer_view):
            print(vm_context)
        hal_module = iree.runtime.create_hal_module(
            vm_instance,
            device,
            debug_sink=iree.runtime.HalModuleDebugSink(callback),
        )
        ```

        This callback will cause the VM context to never be destroyed.
        """

        ...

    @property
    def buffer_view_trace_callback(
        self,
    ) -> Optional[HalModuleBufferViewTraceCallback]: ...

class Linkage(int):
    EXPORT: ClassVar[Linkage] = ...
    IMPORT: ClassVar[Linkage] = ...
    IMPORT_OPTIONAL: ClassVar[Linkage] = ...
    INTERNAL: ClassVar[Linkage] = ...
    __name__: Any

class MappedMemory:
    def __init__(self, *args, **kwargs) -> None: ...
    def asarray(
        self, shape: Sequence[int], numpy_dtype_descr: object
    ) -> np.ndarray[Any, np.dtype[Any]]: ...

class MemoryAccess(enum.IntFlag):
    NONE = 0
    READ = 1
    WRITE = 2
    DISCARD = 4
    DISCARD_WRITE = 6
    ALL = 7

class MemoryType(enum.IntFlag):
    NONE = 0
    OPTIMAL = 1
    HOST_VISIBLE = 2
    HOST_COHERENT = 4
    HOST_CACHED = 8
    HOST_LOCAL = 70
    DEVICE_VISIBLE = 16
    DEVICE_LOCAL = 48

class ParameterIndexEntry:
    @property
    def key(self) -> str: ...
    @property
    def length(self) -> int: ...
    @property
    def metadata(self) -> bytes: ...
    @property
    def is_file(self) -> bool: ...
    @property
    def is_splat(self) -> bool: ...
    @property
    def file_storage(self) -> Tuple[FileHandle, int]:
        """Accesses the underlying storage (if is_file).

        Only valid if is_file. Returns the backing FileHandle and offset.
        """
        ...

    @property
    def file_view(self) -> memoryview:
        """Accesses a memoryview of the file contents.

        Only valid if is_file and the file has host accessible storage.
        """
        ...

    @property
    def splat_pattern(self) -> bytes:
        """Accesses the splat pattern (if is_splat)."""
        ...

class ParameterIndex:
    def __init__(self) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i) -> ParameterIndexEntry: ...
    def reserve(self, new_capacity: int) -> None: ...
    def add_splat(
        self,
        key: str,
        pattern: Any,
        total_length: int,
        *,
        metadata: Optional[Union[bytes, str]] = None,
    ) -> None: ...
    def add_from_file_handle(
        self,
        key: str,
        file_handle: FileHandle,
        length: int,
        *,
        offset: int = 0,
        metadata: Optional[Union[bytes, str]] = None,
    ) -> None: ...
    def add_buffer(
        self,
        key: str,
        buffer: Any,
        *,
        readable: bool = True,
        writable: bool = False,
        metadata: Optional[Union[bytes, str]] = None,
    ) -> None: ...
    def load_from_file_handle(self, file_handle: FileHandle, format: str) -> None: ...
    def load(
        self,
        file_path: str,
        format: Optional[str] = None,
        readable: bool = True,
        writable: bool = False,
        mmap: Optional[bool] = None,
        *,
        mode: Optional[str] = None,
    ) -> None:
        """Load an archive using mmap (default), preload (mmap=False), or
        read-only HAL streaming (mode="file_async"). Explicit mmap and mode
        arguments are mutually exclusive. The index retains the file handle.
        """
        ...

    def create_archive_file(
        self,
        file_path: str,
        file_offset: int = 0,
        target_index: Optional[ParameterIndex] = None,
    ) -> ParameterIndex: ...
    def create_provider(
        self, *, scope: str = "", max_concurrent_operations: Optional[int] = None
    ) -> ParameterProvider: ...
    def items(self) -> List[Tuple[str, ParameterIndexEntry]]:
        """Accesses the items as a tuple(str, entry).

        Note that the index may contain duplicates, so loading into a dict
        is up to the user, as only they can know if this is legal.
        """
        ...

class ParameterProvider: ...

class PyModuleInterface:
    def __init__(self, module_name: str, ctor: object) -> None: ...
    def create(self) -> VmModule: ...
    def export(self, name: str, cconv: str, callable: object) -> None: ...
    @property
    def destroyed(self) -> bool: ...
    @property
    def initialized(self) -> bool: ...

class Shape:
    def __init__(self, indices: Sequence[int]) -> None: ...

class VmBuffer:
    def __init__(
        self, length: int, alignment: int = 0, mutable: bool = True
    ) -> None: ...
    def __eq__(self, other) -> bool: ...
    @property
    def ref(self) -> VmRef: ...
    @property
    def __iree_vm_ref__(self) -> VmRef: ...

class VmContext:
    def __init__(
        self, instance: VmInstance, modules: Optional[Sequence[VmModule]] = None
    ) -> None: ...
    def invoke(
        self, function: VmFunction, inputs: VmVariantList, outputs: VmVariantList
    ) -> None: ...
    def register_modules(self, modules: Sequence[VmModule]) -> None: ...
    @property
    def context_id(self) -> int: ...

class VmFunction:
    @property
    def linkage(self) -> int: ...
    @property
    def module_name(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def ordinal(self) -> int: ...
    @property
    def reflection(self) -> dict: ...

class VmInstance:
    def __init__(self) -> None: ...

class VmModule:
    @staticmethod
    def copy_buffer(instance: VmInstance, buffer: Any) -> VmModule: ...
    @staticmethod
    def from_buffer(
        instance: VmInstance, buffer: Any, warn_if_copy: bool = True
    ) -> VmModule: ...
    @staticmethod
    def from_flatbuffer(
        instance: VmInstance, buffer: Any, warn_if_copy: bool = True
    ) -> VmModule: ...
    @staticmethod
    def mmap(
        instance: VmInstance,
        filepath: str,
        destroy_callback: Optional[Callable[[], Any]] = None,
    ) -> VmModule: ...
    @staticmethod
    def resolve_module_dependency(
        instance: VmInstance, name: str, minimum_version: int
    ) -> VmModule: ...
    @staticmethod
    def wrap_buffer(
        instance: VmInstance,
        buffer: Any,
        destroy_callback: Optional[Callable[[], Any]] = None,
        close_buffer: bool = False,
    ) -> VmModule: ...
    def lookup_function(
        self, name: str, linkage: Linkage = ...
    ) -> Optional[VmFunction]: ...
    @property
    def function_names(self) -> list[str]: ...
    @property
    def name(self) -> str: ...
    @property
    def stashed_flatbuffer_blob(self) -> bytes | None: ...
    @property
    def version(self) -> int: ...

class VmRef:
    def deref(self, value: type[_T], optional: bool = False) -> _T | None: ...
    def isinstance(self, ref_type: type) -> bool: ...
    def __eq__(self, other) -> bool: ...
    @property
    def __iree_vm_ref__(self) -> object: ...

class VmVariantList:
    def __init__(self, capacity: int) -> None: ...
    def get_as_list(self, index: int) -> VmVariantList: ...
    @overload
    def get_as_object(self, index: int) -> object: ...
    @overload
    def get_as_object(self, index: int, clazz: type[_T]) -> _T: ...
    def get_as_ref(self, index: int) -> VmRef: ...
    def get_serialized_trace_value(self, index: int) -> dict: ...
    def get_variant(self, index: int) -> Any: ...
    def push_float(self, value: float) -> None: ...
    def push_int(self, value: int) -> None: ...
    def push_list(self, value: VmVariantList) -> None: ...
    def push_ref(self, ref: VmRef) -> None: ...
    def __len__(self) -> int: ...
    @property
    def ref(self) -> VmRef: ...
    @property
    def size(self) -> int: ...
    @property
    def __iree_vm_ref__(self) -> VmRef: ...

class Tokenizer:
    @staticmethod
    def from_file(path: Union[str, os.PathLike]) -> Tokenizer: ...
    @staticmethod
    def from_huggingface_json(json: str) -> Tokenizer: ...
    @staticmethod
    def from_tiktoken(data: str, encoding: str) -> Tokenizer: ...
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        no_special_token_matching: bool = False,
    ) -> List[int]: ...
    def decode(
        self, token_ids: List[int], skip_special_tokens: bool = False
    ) -> str: ...
    def encode_stream(
        self,
        add_special_tokens: bool = False,
        no_special_token_matching: bool = False,
    ) -> TokenizerEncodeStream: ...
    def decode_stream(
        self, skip_special_tokens: bool = False
    ) -> TokenizerDecodeStream: ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def model_type(self) -> str: ...
    def id_to_token(self, token_id: int) -> Optional[str]: ...
    def token_to_id(self, token: str) -> Optional[int]: ...
    @property
    def special_ids(self) -> dict: ...

class TokenizerEncodeStream:
    def feed(self, chunk: str) -> List[int]: ...
    def finalize(self) -> List[int]: ...
    def __enter__(self) -> TokenizerEncodeStream: ...
    def __exit__(self, *args: Any) -> None: ...

class TokenizerDecodeStream:
    def feed(self, token_ids: List[int]) -> str: ...
    def feed_one(self, token_id: int) -> str:
        """Feed a single token ID. Faster than feed([id]) for per-token LLM decoding."""
        ...

    def finalize(self) -> str: ...
    def __enter__(self) -> TokenizerDecodeStream: ...
    def __exit__(self, *args: Any) -> None: ...

class HalExternalTimepoint:
    def __init__(self) -> None: ...
    @property
    def type(self) -> ExternalTimepointType: ...
    @property
    def flags(self) -> ExternalTimepointFlags: ...
    @property
    def compatibility(self) -> SemaphoreCompatibility: ...
    @property
    def cuda_event(self) -> int: ...
    @property
    def hip_event(self) -> int: ...
