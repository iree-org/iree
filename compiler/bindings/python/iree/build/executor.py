# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Collection, IO, Type, TypeVar

import abc
import concurrent.futures
import enum
import multiprocessing
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


class FileNamespace(enum.StrEnum):
    # Transient generated files go into the GEN namespace. These are typically
    # not packaged for distribution.
    GEN = enum.auto()

    # Distributable parameter files.
    PARAMS = enum.auto()

    # Distributable, platform-neutral binaries.
    BIN = enum.auto()

    # Distributable, platform specific binaries.
    PLATFORM_BIN = enum.auto()


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


class Executor:
    """Executor that all build contexts share."""

    def __init__(self, output_dir: Path, stderr: IO):
        self.output_dir = output_dir
        self.verbose_level = 0
        # Keyed by path
        self.all: dict[str, "BuildContext" | "BuildFile"] = {}
        self.entrypoints: list["BuildEntrypoint"] = []
        self.stderr = stderr
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

    def build(self, *initial_deps: "BuildDependency"):
        """Transitively builds the given deps."""
        scheduler = Scheduler(stderr=self.stderr)
        success = False
        try:
            for d in initial_deps:
                scheduler.add_initial_dep(d)
                scheduler.build()
            success = True
        finally:
            if not success:
                print("Waiting for background tasks to complete...", file=self.stderr)
            scheduler.shutdown()


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
        self.start_time: float | None = None
        self.finish_time: float | None = None

        # Metadata.
        self._metadata: dict[str, BuildMeta] = {}

    @property
    def is_scheduled(self) -> bool:
        return self.future is not None

    @property
    def execution_time(self) -> float:
        if self.start_time is None:
            return 0.0
        if self.finish_time is None:
            return time.time() - self.start_time
        return self.finish_time - self.start_time

    def start(self, future: concurrent.futures.Future):
        assert not self.is_scheduled, f"Cannot start an already scheduled dep: {self}"
        self.future = future
        self.start_time = time.time()

    def finish(self):
        assert self.is_scheduled, "Cannot finish an unstarted dep"
        self.finish_time = time.time()
        self.future.set_result(self)


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

    def __repr__(self):
        return f"BuildFile[{self.namespace}]({self.path})"


class ActionConcurrency(enum.StrEnum):
    THREAD = enum.auto()
    PROCESS = enum.auto()
    NONE = enum.auto()


class BuildAction(BuildDependency, abc.ABC):
    """An action that must be carried out."""

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
        self.concurrnecy = concurrency

    def __str__(self):
        return self.desc

    def __repr__(self):
        return f"Action[{type(self).__name__}]('{self.desc}')"

    def invoke(self):
        self._invoke()

    @abc.abstractmethod
    def _invoke(self):
        ...


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

    def __init__(self, stderr: IO):
        self.stderr = stderr

        # Inverted producer-consumer graph nodes mapping a producer dep to
        # all deps which directly depend on it and will be unblocked by it
        # beins satisfied.
        self.producer_graph: dict[BuildDependency, list[BuildDependency]] = {}

        # Set of build dependencies that have been scheduled. These will all
        # have a future set on them prior to adding to the set.
        self.in_flight_deps: set[BuildDependency] = set()

        self.thread_pool_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=10, thread_name_prefix="iree.build"
        )
        self.process_pool_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=10, mp_context=multiprocessing.get_context("spawn")
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
            print(
                f"Servicing {len(self.producer_graph)} outstanding tasks",
                file=self.stderr,
            )
            self._service_graph()

    def _service_graph(self):
        completed_deps: set[BuildDependency] = set()
        try:
            for completed_fut in concurrent.futures.as_completed(
                (d.future for d in self.in_flight_deps), 0
            ):
                completed_dep = completed_fut.result()
                assert isinstance(completed_dep, BuildDependency)
                print(f"Completed {completed_dep}", file=self.stderr)
                completed_deps.add(completed_dep)
        except TimeoutError:
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
        if isinstance(dep, BuildAction):

            def invoke():
                dep.invoke()
                return dep

            print(f"Scheduling action: {dep}", file=self.stderr)
            if dep.concurrnecy == ActionConcurrency.NONE:
                invoke()
            elif dep.concurrnecy == ActionConcurrency.THREAD:
                dep.start(self.thread_pool_executor.submit(invoke))
            elif dep.concurrnecy == ActionConcurrency.PROCESS:
                dep.start(self.process_pool_executor.submit(invoke))
            else:
                raise AssertionError(
                    f"Unhandled ActionConcurrency value: {dep.concurrnecy}"
                )
        else:
            # Not schedulable. Just mark it as done.
            dep.start(concurrent.futures.Future())
            dep.finish()


# Type aliases.
BuildFileLike = BuildFile | str
