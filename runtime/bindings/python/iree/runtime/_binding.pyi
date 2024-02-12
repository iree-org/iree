from typing import Any, Callable, ClassVar, List, Optional, Sequence, Tuple, Union

from typing import overload

def create_hal_module(instance: VmInstance, device: HalDevice) -> VmModule: ...
def create_io_parameters_module(
    instance: VmInstance, *providers: ParameterProvider
) -> VmModule: ...
def disable_leak_checker(): ...
def get_cached_hal_driver(device_uri: str) -> HalDriver: ...
def parse_flags(*flag: str): ...

class BufferCompatibility(int):
    ALLOCATABLE: ClassVar[BufferCompatibility] = ...
    EXPORTABLE: ClassVar[BufferCompatibility] = ...
    IMPORTABLE: ClassVar[BufferCompatibility] = ...
    NONE: ClassVar[BufferCompatibility] = ...
    QUEUE_DISPATCH: ClassVar[BufferCompatibility] = ...
    QUEUE_TRANSFER: ClassVar[BufferCompatibility] = ...
    __name__: Any
    def __and__(self, other: BufferCompatibility) -> int: ...
    def __or__(self, other: BufferCompatibility) -> int: ...

class BufferUsage(int):
    DEFAULT: ClassVar[BufferUsage] = ...
    DISPATCH_IMAGE: ClassVar[BufferUsage] = ...
    DISPATCH_IMAGE_READ: ClassVar[BufferUsage] = ...
    DISPATCH_IMAGE_WRITE: ClassVar[BufferUsage] = ...
    DISPATCH_INDIRECT_PARAMS: ClassVar[BufferUsage] = ...
    DISPATCH_STORAGE: ClassVar[BufferUsage] = ...
    DISPATCH_STORAGE_READ: ClassVar[BufferUsage] = ...
    DISPATCH_STORAGE_WRITE: ClassVar[BufferUsage] = ...
    DISPATCH_UNIFORM_READ: ClassVar[BufferUsage] = ...
    MAPPING: ClassVar[BufferUsage] = ...
    MAPPING_ACCESS_RANDOM: ClassVar[BufferUsage] = ...
    MAPPING_ACCESS_SEQUENTIAL_WRITE: ClassVar[BufferUsage] = ...
    MAPPING_OPTIONAL: ClassVar[BufferUsage] = ...
    MAPPING_PERSISTENT: ClassVar[BufferUsage] = ...
    MAPPING_SCOPED: ClassVar[BufferUsage] = ...
    NONE: ClassVar[BufferUsage] = ...
    SHARING_CONCURRENT: ClassVar[BufferUsage] = ...
    SHARING_EXPORT: ClassVar[BufferUsage] = ...
    SHARING_IMMUTABLE: ClassVar[BufferUsage] = ...
    SHARING_REPLICATE: ClassVar[BufferUsage] = ...
    TRANSFER: ClassVar[BufferUsage] = ...
    TRANSFER_SOURCE: ClassVar[BufferUsage] = ...
    TRANSFER_TARGET: ClassVar[BufferUsage] = ...
    __name__: str
    def __and__(self, other: BufferUsage) -> int: ...
    def __or__(self, other: BufferUsage) -> int: ...

class FileHandle:
    @staticmethod
    def wrap_memory(
        host_buffer: Any, readable: bool = True, writable: bool = False
    ) -> FileHandle: ...

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

HalSemaphoreList = List[Tuple[HalSemaphore, int]]

class HalDevice:
    def begin_profiling(
        self, mode: Optional[str] = None, file_path: Optional[str] = None
    ) -> None: ...
    def create_semaphore(self, initial_value: int) -> HalSemaphore: ...
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
        command_buffers: Sequence[HalCommandBuffer],
        wait_semaphores: HalSemaphoreList,
        signal_semaphores: HalSemaphoreList,
    ) -> None: ...
    @property
    def allocator(self) -> HalAllocator: ...

class HalDriver:
    @staticmethod
    def query() -> List[str]: ...
    def create_default_device(
        self, allocators: Optional[list[HalAllocator]] = None
    ) -> HalDevice: ...
    @overload
    def create_device(
        self, device_id: int, allocators: Optional[list[HalAllocator]] = None
    ) -> HalDevice: ...
    @overload
    def create_device(
        self, device_info: dict, allocators: Optional[list[HalAllocator]] = None
    ) -> HalDevice: ...
    def create_device_by_uri(
        self, device_uri: str, allocators: Optional[list[HalAllocator]] = None
    ) -> HalDevice: ...
    def query_available_devices(self) -> list[dict]: ...

class HalElementType:
    BFLOAT_16: ClassVar[HalElementType] = ...
    BOOL_8: ClassVar[HalElementType] = ...
    COMPLEX_128: ClassVar[HalElementType] = ...
    COMPLEX_64: ClassVar[HalElementType] = ...
    FLOAT_16: ClassVar[HalElementType] = ...
    FLOAT_32: ClassVar[HalElementType] = ...
    FLOAT_64: ClassVar[HalElementType] = ...
    INT_16: ClassVar[HalElementType] = ...
    INT_32: ClassVar[HalElementType] = ...
    INT_4: ClassVar[HalElementType] = ...
    INT_64: ClassVar[HalElementType] = ...
    INT_8: ClassVar[HalElementType] = ...
    NONE: ClassVar[HalElementType] = ...
    OPAQUE_16: ClassVar[HalElementType] = ...
    OPAQUE_32: ClassVar[HalElementType] = ...
    OPAQUE_64: ClassVar[HalElementType] = ...
    OPAQUE_8: ClassVar[HalElementType] = ...
    SINT_16: ClassVar[HalElementType] = ...
    SINT_32: ClassVar[HalElementType] = ...
    SINT_4: ClassVar[HalElementType] = ...
    SINT_64: ClassVar[HalElementType] = ...
    SINT_8: ClassVar[HalElementType] = ...
    UINT_16: ClassVar[HalElementType] = ...
    UINT_32: ClassVar[HalElementType] = ...
    UINT_4: ClassVar[HalElementType] = ...
    UINT_64: ClassVar[HalElementType] = ...
    UINT_8: ClassVar[HalElementType] = ...
    @staticmethod
    def map_to_dtype(element_type: HalElementType) -> Any: ...
    @staticmethod
    def is_byte_aligned(element_type: HalElementType) -> bool: ...
    @staticmethod
    def dense_byte_count(element_type: HalElementType) -> int: ...
    __name__: Any

class HalFence:
    @staticmethod
    def create_at(value: int) -> HalFence: ...
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
    def query(self) -> int: ...
    def signal(self, new_value: int) -> None: ...

class Linkage(int):
    EXPORT: ClassVar[Linkage] = ...
    IMPORT: ClassVar[Linkage] = ...
    IMPORT_OPTIONAL: ClassVar[Linkage] = ...
    INTERNAL: ClassVar[Linkage] = ...
    __name__: Any

class MappedMemory:
    def __init__(self, *args, **kwargs) -> None: ...
    def asarray(self, shape: Sequence[int], numpy_dtype_descr: object) -> object: ...

class MemoryAccess(int):
    ALL: ClassVar[MemoryAccess] = ...
    DISCARD: ClassVar[MemoryAccess] = ...
    DISCARD_WRITE: ClassVar[MemoryAccess] = ...
    NONE: ClassVar[MemoryAccess] = ...
    READ: ClassVar[MemoryAccess] = ...
    WRITE: ClassVar[MemoryAccess] = ...
    __name__: Any
    def __and__(self, other: MemoryAccess) -> int: ...
    def __or__(self, other: MemoryAccess) -> int: ...

class MemoryType(int):
    DEVICE_LOCAL: ClassVar[MemoryType] = ...
    DEVICE_VISIBLE: ClassVar[MemoryType] = ...
    HOST_CACHED: ClassVar[MemoryType] = ...
    HOST_COHERENT: ClassVar[MemoryType] = ...
    HOST_LOCAL: ClassVar[MemoryType] = ...
    HOST_VISIBLE: ClassVar[MemoryType] = ...
    NONE: ClassVar[MemoryType] = ...
    OPTIMAL: ClassVar[MemoryType] = ...
    __name__: Any
    def __and__(self, other: MemoryType) -> int: ...
    def __or__(self, other: MemoryType) -> int: ...

class ParameterIndex:
    def __init__() -> None: ...
    def __len__(self) -> int: ...
    def reserve(self, new_capacity: int) -> None: ...
    def add_splat(
        self,
        key: str,
        pattern: Any,
        total_length: int,
        *,
        metadata: Optional[Union[bytes, str]] = None
    ) -> None: ...
    def add_from_file_handle(
        self,
        key: str,
        file_handle: FileHandle,
        length: int,
        *,
        offset: int = 0,
        metadata: Optional[Union[bytes, str]] = None
    ) -> None: ...
    def add_buffer(
        self,
        key: str,
        buffer: Any,
        *,
        readable: bool = True,
        writable: bool = False,
        metadata: Optional[Union[bytes, str]] = None
    ) -> None: ...
    def load_from_file_handle(self, file_handle: FileHandle, format: str) -> None: ...
    def load(
        self,
        file_path: str,
        *,
        format: Optional[str] = None,
        readable: bool = True,
        writable: bool = False,
        mmap: bool = True
    ) -> None: ...
    def create_archive_file(
        self,
        file_path: str,
        file_offset: int = 0,
        target_index: Optional[ParameterIndex] = None,
    ) -> ParameterIndex: ...
    def create_provider(
        self, *, scope: str = "", max_concurrent_operations: Optional[int] = None
    ) -> ParameterProvider: ...

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
        self, instance: VmInstance, modules: Optional[list[VmModule]] = None
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
    def stashed_flatbuffer_blob(self) -> object: ...
    @property
    def version(self) -> int: ...

class VmRef:
    def deref(self, value: Any, optional: bool = False) -> object: ...
    def isinstance(self, ref_type: type) -> bool: ...
    def __eq__(self, other) -> bool: ...
    @property
    def __iree_vm_ref__(self) -> object: ...

class VmVariantList:
    def __init__(self, capacity: int) -> None: ...
    def get_as_list(self, index: int) -> VmVariantList: ...
    def get_as_object(self, index: int) -> object: ...
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
