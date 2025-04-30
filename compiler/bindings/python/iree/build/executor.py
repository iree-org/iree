# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Collection, IO, Type, TypeVar

import concurrent.futures
import enum
import json
import math
import multiprocessing
import os
import time
import traceback
from pathlib import Path
import threading

from iree.build.args import (
    current_args_namespace,
    expand_cl_arg_defaults,
    extract_cl_arg_defs,
)

_locals = threading.local()


class FileNamespace(enum.Enum):
    # Transient generated files go into the GEN namespace. These are typically
    # not packaged for distribution.
    GEN = "gen"

    # Distributable parameter files.
    PARAMS = "params"

    # Distributable, platform-neutral binaries.
    BIN = "bin"

    # Distributable, platform specific binaries.
    PLATFORM_BIN = "platform_bin"

    def __str__(self) -> str:
        return self.value


FileNamespaceToPath = {
    FileNamespace.GEN: lambda executor: executor.output_dir / "genfiles",
    FileNamespace.PARAMS: lambda executor: executor.output_dir / "params",
    FileNamespace.BIN: lambda executor: executor.output_dir / "bin",
    # TODO: This isn't right. Need to resolve platform dynamically.
    FileNamespace.PLATFORM_BIN: lambda executor: executor.output_dir / "platform",
}


def join_namespace(prefix: str, suffix: str) -> str:
    """Joins two namespace components, taking care of the root namespace (empty)."""
    if not prefix:
        return suffix
    return f"{prefix}/{suffix}"


class Entrypoint:
    def __init__(
        self,
        name: str,
        wrapped: Callable,
        description: str | None = None,
    ):
        self.name = name
        self.description = description
        self.cl_arg_defs = list(extract_cl_arg_defs(wrapped))
        self._wrapped = expand_cl_arg_defaults(wrapped)

    def __call__(self, *args, **kwargs):
        parent_context = BuildContext.current()
        args_ns = current_args_namespace()
        bep = BuildEntrypoint(
            join_namespace(parent_context.path, self.name),
            parent_context.executor,
            self,
        )
        parent_context.executor.entrypoints.append(bep)
        with bep:
            results = self._wrapped(*args, **kwargs)
            if results is not None:
                files = bep.files(results)
                bep.deps.update(files)
                bep.outputs.extend(files)
                return files


class ProgressReporter:
    def reset_display(self):
        ...

    def start_graph(self, all_deps: set["BuildDependency"]):
        ...

    def start_dep(self, dep: "BuildDependency"):
        ...

    def finish_dep(self, dep: "BuildDependency"):
        ...

    def end_graph(self):
        ...

    def report_failure(self, dep: "BuildDependency"):
        ...


class DependenceException(Exception):
    """Noted on a BuildDependency.failure when the dep could not be satisfied because
    of failed dependencies."""

    ...


class Executor:
    """Executor that all build contexts share."""

    def __init__(self, output_dir: Path, stderr: IO, reporter: ProgressReporter):
        self.output_dir = output_dir
        self.verbose_level = 0
        # Keyed by path
        self.all: dict[str, "BuildContext" | "BuildFile"] = {}
        self.entrypoints: list["BuildEntrypoint"] = []
        self.failed_deps: set["BuildDependency"] = set()
        self.stderr = stderr
        self.reporter = reporter
        self.metadata_lock = threading.RLock()
        BuildContext("", self)

    def check_path_not_exists(self, path: str, for_entity):
        existing = self.all.get(path)
        if existing is not None:
            formatted_stack = "".join(traceback.format_list(existing.def_stack))
            raise RuntimeError(
                f"Cannot add {for_entity} because an entity with that name was "
                f"already defined at:\n{formatted_stack}"
            )

    def get_context(self, path: str) -> "BuildContext":
        existing = self.all.get(path)
        if existing is None:
            raise RuntimeError(f"Context at path {path} not found")
        if not isinstance(existing, BuildContext):
            raise RuntimeError(
                f"Entity at path {path} is not a context. It is: {existing}"
            )
        return existing

    def get_file(self, path: str) -> "BuildFile":
        existing = self.all.get(path)
        if existing is None:
            raise RuntimeError(f"File at path {path} not found")
        if not isinstance(existing, BuildFile):
            raise RuntimeError(
                f"Entity at path {path} is not a file. It is: {existing}"
            )
        return existing

    def write_status(self, message: str):
        self.reporter.reset_display()
        print(message, file=self.stderr)

    def get_root(self, namespace: FileNamespace) -> Path:
        return FileNamespaceToPath[namespace](self)

    def analyze(self, *entrypoints: Entrypoint):
        """Analyzes all entrypoints building the graph."""
        for entrypoint in entrypoints:
            if self.verbose_level > 1:
                self.write_status(f"Analyzing entrypoint {entrypoint.name}")
            with self.get_context("") as context:
                entrypoint()

    def build(self, *initial_deps: "BuildDependency") -> bool:
        """Transitively builds the given deps."""
        scheduler = Scheduler(reporter=self.reporter)
        success = False
        started_reporter = False
        try:
            for d in initial_deps:
                scheduler.add_initial_dep(d)
                self.reporter.start_graph(set(scheduler.producer_graph.keys()))
                started_reporter = True
                scheduler.build()
        except KeyboardInterrupt:
            raise
        except:
            # This catches truly unhandled exceptions (not just build action failures,
            # which are noted in the graph). Eagerly print the exception so that it
            # doesn't get swallowed waiting for shutdown.
            self.reporter.reset_display()
            print(
                "Unhandled exception during build. Waiting for background tasks to complete...",
                file=self.stderr,
            )
            traceback.print_exc(file=self.stderr)
        finally:
            scheduler.shutdown()
            if started_reporter:
                self.reporter.end_graph()
        self.failed_deps.update(scheduler.failed_deps)


BuildMetaType = TypeVar("BuildMetaType", bound="BuildMeta")


class BuildMeta:
    """Base class for typed metadata that can be set on a BuildDependency.

    This is an open namespace where each sub-class must have a unique key as the class
    level attribute `KEY`.
    """

    def __init__(self):
        key = getattr(self, "KEY", None)
        assert isinstance(key, str), "BuildMeta.KEY must be a str"

    @classmethod
    def get(cls: Type[BuildMetaType], dep: "BuildDependency") -> BuildMetaType:
        """Gets a metadata instance of this type from a dependency.

        If it does not yet exist, returns the value of `create_default()`, which
        by default returns a new instance (which is set on the dep).
        """
        key = getattr(cls, "KEY", None)
        assert isinstance(key, str), f"{cls.__name__}.KEY must be a str"
        instance = dep._metadata.get(key)
        if instance is None:
            instance = cls.create_default()
            dep._metadata[key] = instance
        return instance

    @classmethod
    def create_default(cls) -> "BuildMeta":
        """Creates a default instance."""
        return cls()


class BuildDependency:
    """Base class of entities that can act as a build dependency."""

    def __init__(
        self, *, executor: Executor, deps: set["BuildDependency"] | None = None
    ):
        self.executor = executor
        self.deps: set[BuildDependency] = set()
        if deps:
            self.deps.update(deps)

        # Scheduling state.
        self.future: concurrent.futures.Future | None = None
        self.start_time: float | None = None  # Time the action was scheduled.
        self.invoke_time: float | None = None  # Time that invocation began.
        self.finish_time: float | None = None  # Time that finished.

        # If the dep ended in failure, there will be an exception here.
        self.failure: Exception | None = None

        # Metadata.
        self._metadata: dict[str, BuildMeta] = {}

    @property
    def is_scheduled(self) -> bool:
        return self.future is not None

    @property
    def is_dependence_failure(self) -> bool:
        return isinstance(self.failure, DependenceException)

    @property
    def execution_time(self) -> float:
        """Time from begin of invocation to present or action finish.

        This will be zero if the dependency has no invoke time. This does not
        track queued time prior to receiving a thread.
        """
        start_time = self.invoke_time
        if start_time is None:
            return 0.0
        if self.finish_time is None:
            return time.time() - start_time
        return self.finish_time - start_time

    def start(self, future: concurrent.futures.Future):
        assert not self.is_scheduled, f"Cannot start an already scheduled dep: {self}"
        self.future = future
        self.start_time = time.time()

    def finish(self):
        assert self.is_scheduled, "Cannot finish an unstarted dep"
        self.finish_time = time.time()
        self.future.set_result(self)


BuildFileMetadata = dict[str, str | int | bool | float]


class BuildFile(BuildDependency):
    """Generated file in the build tree."""

    def __init__(
        self,
        *,
        executor: Executor,
        path: str,
        namespace: FileNamespace = FileNamespace.GEN,
        deps: set[BuildDependency] | None = None,
    ):
        super().__init__(executor=executor, deps=deps)
        self.def_stack = traceback.extract_stack()[0:-2]
        self.executor = executor
        self.path = path
        self.namespace = namespace
        # Set of build files that must be made available to any transitive user
        # of this build file at runtime.
        self.runfiles: set["BuildFile"] = set()

        executor.check_path_not_exists(path, self)
        executor.all[path] = self

    def get_fs_path(self) -> Path:
        path = self.executor.get_root(self.namespace) / self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def access_metadata(
        self,
        mutation_callback: Callable[[BuildFileMetadata], bool] | None = None,
    ) -> BuildFileMetadata:
        """Accesses persistent metadata about the build file.

        This is intended for the storage of small amounts of metadata relevant to the
        build system for performing up-to-date checks and the like.

        If a `mutation_callback=` is provided, then any modifications it makes will be
        persisted prior to returning. Using a callback in this fashion holds a lock
        and avoids data races. If the callback returns True, it is persisted.
        """
        with self.executor.metadata_lock:
            metadata = _load_metadata(self.executor)
            path_metadata = metadata.get("paths")
            if path_metadata is None:
                path_metadata = {}
                metadata["paths"] = path_metadata
            file_key = f"{self.namespace}/{self.path}"
            file_metadata = path_metadata.get(file_key)
            if file_metadata is None:
                file_metadata = {}
                path_metadata[file_key] = file_metadata
            if mutation_callback:
                if mutation_callback(file_metadata):
                    _save_metadata(self.executor, metadata)
            return file_metadata

    def __repr__(self):
        return f"BuildFile[{self.namespace}]({self.path})"


class ActionConcurrency(enum.Enum):
    THREAD = "thread"
    PROCESS = "process"
    NONE = "none"

    def __str__(self) -> str:
        return self.value


class BuildAction(BuildDependency):
    """An action that must be carried out.

    This class is designed to be subclassed by concrete actions. In-process
    only actions should override `_invoke`, whereas those that can be executed
    out-of-process must override `_remotable_thunk`.

    Note that even actions that are marked for `PROCESS` concurrency will
    run on a dedicated thread within the host process. Only the `_remotable_thunk`
    result will be scheduled out of process.
    """

    def __init__(
        self,
        *,
        desc: str,
        executor: Executor,
        concurrency: ActionConcurrency = ActionConcurrency.THREAD,
        deps: set[BuildDependency] | None = None,
    ):
        super().__init__(executor=executor, deps=deps)
        self.desc = desc
        self.concurrency = concurrency

    def __str__(self):
        return self.desc

    def __repr__(self):
        return f"Action[{type(self).__name__}]('{self.desc}')"

    def invoke(self, scheduler: "Scheduler"):
        # Invoke is run within whatever in-process execution context was requested:
        #   - On the scheduler thread for NONE
        #   - On a worker thread for THREAD or PROCESS
        # For PROCESS concurrency, we have to create a compatible invocation
        # thunk, schedule that on the process pool and wait for it.
        self.invoke_time = time.time()
        try:
            if self.concurrency == ActionConcurrency.PROCESS:
                thunk = self._remotable_thunk()
                fut = scheduler.process_pool_executor.submit(thunk)
                fut.result()
            else:
                self._invoke()
        except Exception as e:
            self.failure = e

    def _invoke(self):
        self._remotable_thunk()()

    def _remotable_thunk(self) -> Callable[[], None]:
        """Creates a remotable no-arg thunk that will execute this out of process.

        This must return a no arg/result callable that can be pickled. While there
        are various ways to ensure this, here are a few guidelines:

        * Must be a type/function defined at a module level.
        * Cannot be decorated.
        * Must only contain attributes with the same constraints.
        """
        raise NotImplementedError(
            f"Action '{self}' does not implement remotable invocation"
        )


class BuildContext(BuildDependency):
    """Manages a build graph under construction."""

    def __init__(self, path: str, executor: Executor):
        super().__init__(executor=executor)
        self.def_stack = traceback.extract_stack()[0:-2]
        self.executor = executor
        self.path = path
        executor.check_path_not_exists(path, self)
        executor.all[path] = self
        self.analyzed = False

    def __repr__(self):
        return f"{type(self).__name__}(path='{self.path}')"

    def allocate_file(
        self, path: str, namespace: FileNamespace = FileNamespace.GEN
    ) -> BuildFile:
        """Allocates a file in the build tree with local path |path|.

        If |path| is absoluate (starts with '/'), then it is used as-is. Otherwise,
        it is joined with the path of this context.
        """
        if not path.startswith("/"):
            path = join_namespace(self.path, path)
        build_file = BuildFile(executor=self.executor, path=path, namespace=namespace)
        return build_file

    def file(self, file: str | BuildFile) -> BuildFile:
        """Accesses a BuildFile by either string (path) or BuildFile.

        It must already exist.
        """
        if isinstance(file, BuildFile):
            return file
        path = file
        if not path.startswith("/"):
            path = join_namespace(self.path, path)
        existing = self.executor.all.get(path)
        if not isinstance(existing, BuildFile):
            all_files = [
                f.path for f in self.executor.all.values() if isinstance(f, BuildFile)
            ]
            all_files_lines = "\n  ".join(all_files)
            raise RuntimeError(
                f"File with path '{path}' is not known in the build graph. Available:\n"
                f"  {all_files_lines}"
            )
        return existing

    def files(
        self, files: str | BuildFile | Collection[str | BuildFile]
    ) -> list[BuildFile]:
        """Accesses a collection of files (or single) as a list of BuildFiles."""
        if isinstance(files, (str, BuildFile)):
            return [self.file(files)]
        return [self.file(f) for f in files]

    @staticmethod
    def current() -> "BuildContext":
        try:
            return _locals.context_stack[-1]
        except (AttributeError, IndexError):
            raise RuntimeError(
                "The current code can only be evaluated within an active BuildContext"
            )

    def __enter__(self) -> "BuildContext":
        try:
            stack = _locals.context_stack
        except AttributeError:
            stack = _locals.context_stack = []
        stack.append(self)
        return self

    def __exit__(self, *args):
        try:
            stack = _locals.context_stack
        except AttributeError:
            raise AssertionError("BuildContext exit without enter")
        existing = stack.pop()
        assert existing is self, "Unbalanced BuildContext enter/exit"


class BuildEntrypoint(BuildContext):
    def __init__(self, path: str, executor: Executor, entrypoint: Entrypoint):
        super().__init__(path, executor)
        self.entrypoint = entrypoint
        self.outputs: list[BuildFile] = []


class Scheduler:
    """Holds resources related to scheduling."""

    def __init__(self, reporter: ProgressReporter):
        self.reporter = reporter

        # Inverted producer-consumer graph nodes mapping a producer dep to
        # all deps which directly depend on it and will be unblocked by it
        # beins satisfied.
        self.producer_graph: dict[BuildDependency, list[BuildDependency]] = {}

        # Set of build dependencies that have been scheduled. These will all
        # have a future set on them prior to adding to the set.
        self.in_flight_deps: set[BuildDependency] = set()

        # Any deps that have failed are added here.
        self.failed_deps: set[BuildDependency] = set()

        # TODO: This needs a flag or additional heuristics. Empirically, at the
        # time of writing, it was found best to limit the scheduler concurrency
        # to a bit less than half of the hardware concurrency and then letting
        # the compiler's thread pool fan out to full hardware concurrency via
        # `export IREE_COMPILER_TASK_COUNT=0`. This wasn't tested super
        # scientifically but was shown to get the best throughput on a mixed
        # torch import -> compile build of 1000 models (about 1m9s for all of it
        # on the tested configuration).
        concurrency = int(max(1, math.ceil((os.cpu_count() or 1) * 0.40)))
        mp_context = os.environ.get("IREE_BUILD_MP_CONTEXT", "spawn")
        self.thread_pool_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="iree.build"
        )
        self.process_pool_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=concurrency, mp_context=multiprocessing.get_context(mp_context)
        )

    def shutdown(self):
        self.thread_pool_executor.shutdown(cancel_futures=True)
        self.process_pool_executor.shutdown(cancel_futures=True)

    def add_initial_dep(self, initial_dep: BuildDependency):
        assert isinstance(initial_dep, BuildDependency)
        if initial_dep in self.producer_graph:
            # Already in the graph.
            return

        # At this point nothing depends on this initial dep, so just note it
        # as producing nothing.
        self.producer_graph[initial_dep] = []

        # Adds a dep requested by some top-level caller.
        stack: set[BuildDependency] = set()
        stack.add(initial_dep)
        for producer_dep in initial_dep.deps:
            self._add_dep(producer_dep, initial_dep, stack)

    def _add_dep(
        self,
        producer_dep: BuildDependency,
        consumer_dep: BuildDependency,
        stack: set[BuildDependency],
    ):
        if producer_dep in stack:
            raise RuntimeError(
                f"Circular dependency: '{producer_dep}' depends on itself: {stack}"
            )
        plist = self.producer_graph.get(producer_dep)
        if plist is None:
            plist = []
            self.producer_graph[producer_dep] = plist
        plist.append(consumer_dep)
        next_stack = set(stack)
        next_stack.add(producer_dep)
        if producer_dep.deps:
            # Intermediate dep.
            for next_dep in producer_dep.deps:
                self._add_dep(next_dep, producer_dep, next_stack)

    def build(self):
        # Build all deps until the graph is satisfied.
        # Schedule any deps that have no dependencies to start things off.
        for eligible_dep in self.producer_graph.keys():
            if len(eligible_dep.deps) == 0:
                self._schedule_action(eligible_dep)
                self.in_flight_deps.add(eligible_dep)

        while self.producer_graph:
            self._service_graph()

    def _service_graph(self):
        completed_deps: set[BuildDependency] = set()
        try:
            for completed_fut in concurrent.futures.as_completed(
                (d.future for d in self.in_flight_deps), 0
            ):
                completed_dep = completed_fut.result()
                assert isinstance(completed_dep, BuildDependency)
                if completed_dep.failure:
                    self.failed_deps.add(completed_dep)
                    self.reporter.report_failure(completed_dep)
                completed_deps.add(completed_dep)
                self.reporter.finish_dep(completed_dep)

        except TimeoutError:
            pass
        except concurrent.futures.TimeoutError:
            # In Python 3.10, future access throws concurrent.futures.TimeoutError.
            # In 3.11, that was made a subclass of TimeoutError, which is advertised
            # as thrown (and the original is marked as deprecated).
            # TODO: Remove this clause once 3.10 support is dropped.
            pass

        # Purge done from in-flight list.
        self.in_flight_deps.difference_update(completed_deps)

        # Schedule any available.
        for completed_dep in completed_deps:
            ready_list = self.producer_graph.get(completed_dep)
            if ready_list is None:
                continue
            del self.producer_graph[completed_dep]
            for ready_dep in ready_list:
                self._schedule_action(ready_dep)
                self.in_flight_deps.add(ready_dep)

        # Do a blocking wait for at least one ready.
        concurrent.futures.wait(
            (d.future for d in self.in_flight_deps),
            return_when=concurrent.futures.FIRST_COMPLETED,
        )

    def _schedule_action(self, dep: BuildDependency):
        if dep.is_scheduled:
            return

        # If any deps depended on failed, then cascade the failure.
        for dep_dep in dep.deps:
            if dep_dep.failure:
                dep.failure = DependenceException()
                dep.start(concurrent.futures.Future())
                dep.finish()
                return

        if isinstance(dep, BuildAction):

            def invoke():
                dep.invoke(self)
                return dep

            self.reporter.start_dep(dep)
            if dep.concurrency == ActionConcurrency.NONE:
                invoke()
                dep.start(concurrent.futures.Future())
                dep.finish()
            elif (
                dep.concurrency == ActionConcurrency.THREAD
                or dep.concurrency == ActionConcurrency.PROCESS
            ):
                dep.start(self.thread_pool_executor.submit(invoke))
            else:
                raise AssertionError(
                    f"Unhandled ActionConcurrency value: {dep.concurrency}"
                )
        else:
            # Not schedulable. Just mark it as done.
            dep.start(concurrent.futures.Future())
            dep.finish()


# Type aliases.
BuildFileLike = BuildFile | str

# Private utilities.
_METADATA_FILENAME = ".metadata.json"


def _load_metadata(executor: Executor) -> dict:
    path = executor.output_dir / _METADATA_FILENAME
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return json.load(f)


def _save_metadata(executor: Executor, metadata: dict):
    path = executor.output_dir / _METADATA_FILENAME
    with open(path, "wt") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)
