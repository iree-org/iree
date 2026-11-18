# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# One stop build of IREE Python packages for Windows. This presumes that
# dependencies are installed from install_windows_deps.ps1.

# Configure settings with script parameters.
param(
    # Python versions to build. 3.12 produces an abi3 wheel (compatible with 3.12+).
    # Per-version builds are used for <3.12 and free-threaded builds.
    [array]$python_versions=@("3.11", "3.12"),
    [array]$packages=@("iree-base-runtime", "iree-base-compiler"),
    [System.String]$output_dir
)

# Also allow setting parameters via environment variables.
if ($env:override_python_versions) { $python_versions = $env:override_python_versions -split ' '};
if ($env:packages) { $packages = $env:packages -split ' '};
if ($env:output_dir) { $output_dir = $env:output_dir };
# Default output directory requires evaluating an expression.
if (!$output_dir) { $output_dir = "${PSScriptRoot}\wheelhouse" };

$repo_root = resolve-path "${PSScriptRoot}\..\.."
$orig_path = $env:PATH

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
      install_build_requirements $python_version
      switch ($package) {
          "iree-base-runtime" {
            clean_wheels iree_base_runtime $python_version
            build_iree_runtime $python_version
          }
          "iree-base-compiler" {
            clean_wheels iree_base_compiler $python_version
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
  $env:IREE_HAL_DRIVER_HIP = "ON"
  $env:IREE_HAL_DRIVER_CUDA = "ON"
  build_wheel $python_version $repo_root/runtime/
}

function build_iree_compiler() {
  param($python_version)
  $env:IREE_TARGET_BACKEND_CUDA = "ON"
  $env:IREE_TARGET_BACKEND_ROCM = "ON"
  build_wheel $python_version $repo_root/compiler/
}

# Installs the packages that build the wheels, pinned by hash, and puts the
# scripts of this Python first on the PATH, so that the build runs its cmake.
function install_build_requirements() {
  param($python_version)
  & py -${python_version} -m pip install --require-hashes --only-binary=:all: `
    -r "${repo_root}\build_tools\python_deploy\build_requirements.txt"
  if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Installing the build requirements for Python ${python_version} failed."
    exit $LASTEXITCODE
  }
  $scripts_dir = & py -${python_version} -c "import sysconfig; print(sysconfig.get_path('scripts'))"
  $env:PATH = "${scripts_dir};${orig_path}"
}

# Builds with the packages that install_build_requirements installed, instead
# of the newest ones in an isolated environment. PowerShell does not stop when
# a native command fails, and the CI step only checks the exit code of the last
# command, so exit on the first failure.
function build_wheel() {
  param($python_version, $source_dir)
  & py -${python_version} -m pip wheel --no-deps --no-build-isolation `
    --check-build-dependencies -v -w $output_dir $source_dir
  if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Building the wheel of ${source_dir} with Python ${python_version} failed."
    exit $LASTEXITCODE
  }
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
