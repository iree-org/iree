# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

# Load it in a sub-shell and dump the variables.
$vcvars = @(cmd.exe /c "call `"$vcvarsFile`" x64 > NUL && set")
foreach ($entry in $vcvars) {
  if ($entry -match "^(.*?)=(.*)$") {
    $key = "$($matches[1])"
    $value = "$($matches[2])"

    if($key -eq "Path") {
      $value = "$pythonPath;$bashPath;$value"
    }

    Add-Content $githubEnv "$key=$value"
    Write-Output ":: $key = $value"
  }
}
