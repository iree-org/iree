# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Test me with something like:
#
#   $Env:GITHUB_ENV="ghenv.txt"
#   $Env:GITHUB_PATH="ghpath.txt"
#   .\configure_dev_environment.ps1 -bashExePath C:\tools\msys64\usr\bin\bash.exe

param ([Parameter(Mandatory)]$bashExePath)

# Resolve the items we need from the path prior to vcvars mangling it.
$pythonPath = (Get-Command python.exe).Path | Split-Path -Parent
Write-Output "++ Found Python $pythonPath"

# Resolve bash.exe.
if (!(Test-Path $bashExePath -PathType Leaf)) {
  Write-Error "-- Could not find bash.exe at $bashExePath"
  exit 1
}
$bashPath = $bashExePath | Split-Path -Parent
Write-Output "++ Found bash path $bashPath"

# Use vswhere to find vcvarsall path.
$vsInstallPath = vswhere -property installationPath
$vcvarsFile = "$($vsInstallPath)\VC\Auxiliary\Build\vcvarsall.bat"

if (!(Test-Path $vcvarsFile -PathType Leaf)) {
  Write-Error "-- Could not find vcvarsall file: $vcvarsFile"
  exit 1
}
Write-Output "++ VCVarsAll Path: $vcvarsFile"

# Get the github environement file.
$githubEnv = $Env:GITHUB_ENV
if (!($githubEnv)) {
  Write-Error "-- Not running under GitHub Actions (no GITHUB_ENV var)"
  exit 1
}
Write-Output "++ GITHUB_ENV = $githubEnv"
$githubPath = $Env:GITHUB_PATH
if (!($githubPath)) {
  Write-Error "-- Not running under GitHub Actions (no GITHUB_PATH var)"
  exit 1
}
Write-Output "++ GITHUB_PATH = $githubPath"

# Load it in a sub-shell and dump the variables.
$vcvars = @(cmd.exe /c "call `"$vcvarsFile`" x64 > NUL && set")
foreach ($entry in $vcvars) {
  if ($entry -match "^(.*?)=(.*)$") {
    $key = "$($matches[1])"
    $value = "$($matches[2])"

    if($key -eq "Path") {
      # Accumulate the existing path.
      $existingPathArray = @()
      $env:Path.ToString().TrimEnd(';') -split ';' | ForEach-Object {
        $existingPathArray += $_
      }

      # Process the new path.
      $newPathArray = $value -split ';'
      [array]::Reverse($newPathArray)

      # In reverse order, tell GitHub to add any new entries to the path.
      # This still may get the order slightly wrong if existing things were
      # re-added, but we really have to draw the line somewhere...
      foreach ($newPathEntry in $newPathArray) {
        if (!$existingPathArray.Contains($newPathEntry)) {
          Add-Content $githubPath "$newPathEntry" -Encoding utf8
          Write-Output "++ PREPEND PATH: $newPathEntry"
        }
      }

      # Note: High priority override path changes are applied in reverse order.
      Add-Content $githubPath "$bashPath" -Encoding utf8
      Write-Output "++ PREPEND PATH: $bashPath"
      Add-Content $githubPath "$pythonPath" -Encoding utf8
      Write-Output "++ PREPEND PATH: $pythonPath"
    } else {
      Add-Content $githubEnv "$key=$value" -Encoding utf8
      Write-Output "++  $key = $value"
    }
  }
}

# Finally, emit the BAZEL_SH parameter. Because sometimes it doesn't respect
# the path. Because... awesomeness.
Add-Content $githubEnv "BAZEL_SH=$bashExePath" -Encoding utf8
