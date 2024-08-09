# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# One stop build of IREE Python packages for Windows. This presumes that
# dependencies are installed from install_windows_deps.ps1.

# Configure settings with script parameters.
param(
    [array]$python_versions=@("3.11"),
    [array]$packages=@("iree-runtime", "iree-compiler"),
    [System.String]$output_dir
)

# Also allow setting parameters via environment variables.
if ($env:override_python_versions) { $python_versions = $env:override_python_versions -split ' '};
if ($env:packages) { $packages = $env:packages -split ' '};
if ($env:output_dir) { $output_dir = $env:output_dir };
# Default output directory requires evaluating an expression.
if (!$output_dir) { $output_dir = "${PSScriptRoot}\wheelhouse" };

$repo_root = resolve-path "${PSScriptRoot}\..\.."

# Canonicalize paths.
md -Force ${output_dir} | Out-Null
$output_dir = resolve-path "${output_dir}"

function run() {
  Write-Host "Using Python versions: ${python_versions}"

  $installed_versions_output = py --list | Out-String

  # Build phase.
  for($i=0 ; $i -lt $packages.Length; $i++) {
    $package = $packages[$i]

    echo "******************** BUILDING PACKAGE ${package} ********************"
    for($j=0 ; $j -lt $python_versions.Length; $j++) {
      $python_version = $python_versions[$j]

      if (!("${installed_versions_output}" -like "*${python_version}*")) {
        Write-Host "ERROR: Could not find python version: ${python_version}"
        continue
      }

      Write-Host ":::: Version: $(py -${python_version} --version)"
      switch ($package) {
          "iree-runtime" {
            clean_wheels iree_runtime $python_version
            build_iree_runtime $python_version
          }
          "iree-compiler" {
            clean_wheels iree_compiler $python_version
            build_iree_compiler $python_version
          }
          Default {
            Write-Host "Unrecognized package '$package'"
            exit 1
          }
      }
    }
  }
}

function build_iree_runtime() {
  param($python_version)
  $env:IREE_HAL_DRIVER_VULKAN = "ON"
  & py -${python_version} -m pip wheel -v -w $output_dir $repo_root/runtime/
}

function build_iree_compiler() {
  param($python_version)
  $env:IREE_TARGET_BACKEND_CUDA = "ON"
  py -${python_version} -m pip wheel -v -w $output_dir $repo_root/compiler/
}

function clean_wheels() {
  param($wheel_basename, $python_version)
  Write-Host ":::: Clean wheels $wheel_basename $python_version"

  # python_version is something like "3.11", but we'd want something like "cp311".
  $python_version_parts = $python_version.Split(".")
  $python_version_major = $python_version_parts[0]
  $python_version_minor = $python_version_parts[1]
  $cpython_version_string = "cp${python_version_major}${python_version_minor}"

  Get-ChildItem ${output_dir} | Where{$_.Name -Match "${wheel_basename}-.*-${cpython_version_string}-.*.whl"} | Remove-Item
}

run
