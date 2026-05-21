// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';

const require = createRequire(import.meta.url);

const wgslPaths = process.argv.slice(2);
if (wgslPaths.length === 0) {
  console.error('usage: node validate_wgsl.mjs <shader.wgsl>...');
  process.exit(1);
}

async function importWebGpuPackage() {
  try {
    return await import('webgpu');
  } catch (primaryError) {
    const packageRoot = process.env.IREE_WEBGPU_PACKAGE_ROOT;
    if (packageRoot) {
      try {
        const packagePath = require.resolve('webgpu', {paths: [packageRoot]});
        return await import(pathToFileURL(packagePath));
      } catch (rootError) {
        throw new Error(
            primaryError.message + '\n' +
            `Failed to import webgpu from IREE_WEBGPU_PACKAGE_ROOT=${
                packageRoot}: ${rootError.message}`);
      }
    }
    throw primaryError;
  }
}

let gpu;
try {
  const {create, globals} = await importWebGpuPackage();
  Object.assign(globalThis, globals);
  gpu = create([]);
} catch (error) {
  console.error(
      'Failed to import webgpu package: ' + error.message + '\n' +
      'Install outside the checkout with:\n' +
      '  npm install --prefix /tmp/iree-webgpu-validator webgpu\n' +
      '  IREE_WEBGPU_PACKAGE_ROOT=/tmp/iree-webgpu-validator node ' +
      'samples/webgpu/hello_world/validate_wgsl.mjs <shader.wgsl>...');
  process.exit(1);
}

const adapter = await gpu.requestAdapter();
if (!adapter) {
  console.error('WebGPU adapter not available.');
  process.exit(1);
}

const device = await adapter.requestDevice();
if (!device) {
  console.error('WebGPU device creation failed.');
  process.exit(1);
}

function getComputeEntryPoints(source) {
  const entryPoints = [];
  const entryPointExpression =
      /@compute[\s\S]*?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/g;
  for (let match = entryPointExpression.exec(source); match;
       match = entryPointExpression.exec(source)) {
    entryPoints.push(match[1]);
  }
  return entryPoints;
}

let failed = false;
for (const wgslPath of wgslPaths) {
  const source = readFileSync(wgslPath, 'utf8');
  const entryPoints = getComputeEntryPoints(source);
  if (entryPoints.length === 0) {
    console.error(`${wgslPath}: no @compute entry point found.`);
    failed = true;
    continue;
  }

  device.pushErrorScope('validation');
  const shaderModule = device.createShaderModule({
    label: wgslPath,
    code: source,
  });

  if (shaderModule.getCompilationInfo) {
    const compilationInfo = await shaderModule.getCompilationInfo();
    for (const message of compilationInfo.messages) {
      const location = `${message.lineNum}:${message.linePos}`;
      const text =
          `${wgslPath}:${location}: ${message.type}: ${message.message}`;
      if (message.type === 'error') {
        console.error(text);
        failed = true;
      } else {
        console.warn(text);
      }
    }
  }

  for (const entryPoint of entryPoints) {
    device.createComputePipeline({
      label: `${wgslPath}:${entryPoint}`,
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint,
      },
    });
  }

  const validationError = await device.popErrorScope();
  if (validationError) {
    console.error(`${wgslPath}: ${validationError.message}`);
    failed = true;
  } else {
    console.log(`${wgslPath}: validated ${entryPoints.join(', ')}`);
  }
}

device.destroy();
process.exit(failed ? 1 : 0);
