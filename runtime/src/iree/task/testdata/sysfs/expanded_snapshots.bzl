# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule expanding sysfs snapshot manifests into directory trees at build time.
"""

def _expanded_sysfs_snapshots_impl(ctx):
    directory = ctx.actions.declare_directory(ctx.attr.name)
    suffix = ".sysfs.txt"
    commands = []
    for manifest in ctx.files.manifests:
        if not manifest.basename.endswith(suffix):
            fail("manifest %s must end in %s" % (manifest.basename, suffix))
        name = manifest.basename[:-len(suffix)]
        commands.append('bash "{script}" --expand "{manifest}" "{out}/{name}" >/dev/null'.format(
            script = ctx.file.expand_tool.path,
            manifest = manifest.path,
            out = directory.path,
            name = name,
        ))
    ctx.actions.run_shell(
        inputs = ctx.files.manifests + [ctx.file.expand_tool],
        outputs = [directory],
        command = " && ".join(commands),
        mnemonic = "ExpandSysfsSnapshots",
        progress_message = "Expanding sysfs snapshot manifests",
    )
    return [DefaultInfo(
        files = depset([directory]),
        runfiles = ctx.runfiles(files = [directory]),
    )]

expanded_sysfs_snapshots = rule(
    implementation = _expanded_sysfs_snapshots_impl,
    attrs = {
        "manifests": attr.label_list(
            mandatory = True,
            allow_files = [".sysfs.txt"],
        ),
        "expand_tool": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
    },
)
