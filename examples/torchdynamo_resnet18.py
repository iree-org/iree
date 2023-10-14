# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import sys
from typing import List

from PIL import Image
import requests

import torch
import torch._dynamo as dynamo
import torchvision.models as models
from torchvision import transforms

import torch_mlir
from torch_mlir.dynamo import make_simple_dynamo_backend
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

from magic import  *
import time


def load_and_preprocess_image(url: str):
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    }
    img = Image.open(requests.get(url, headers=headers,
                                  stream=True).raw).convert("RGB")
    # preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_preprocessed = preprocess(img)
    return torch.unsqueeze(img_preprocessed, 0)

def load_and_preprocess_image_path(path: str):
    img = Image.open(path).convert("RGB")
    # preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_preprocessed = preprocess(img)
    return torch.unsqueeze(img_preprocessed, 0)

def load_labels():
    classes_text = requests.get(
        "https://raw.githubusercontent.com/cathyzhyi/ml-data/main/imagenet-classes.txt",
        stream=True,
    ).text
    labels = [line.strip() for line in classes_text.splitlines()]
    return labels


def top3_possibilities(res):
    _, indexes = torch.sort(res, descending=True)
    percentage = torch.nn.functional.softmax(res, dim=1)[0] * 100
    top3 = [(labels[idx], percentage[idx].item()) for idx in indexes[0][:3]]
    return top3


def predictions(torch_func, jit_func, net, img, labels):
    golden_prediction = top3_possibilities(torch_func(img))
    print("PyTorch prediction")
    print(golden_prediction)
    prediction = top3_possibilities(torch.from_numpy(jit_func(img.numpy())))
    print("torch-mlir prediction")
    print(prediction)
    res = Engine(net, img)
    iree_res = top3_possibilities(torch.from_numpy(res))
    print("iree prediction")
    print(iree_res)

def profiling(torch_func, jit_func, net, img, labels, times):

    start = time.perf_counter()
    for i in range(0, times):
       res = Engine(net, img)
       iree_res = top3_possibilities(torch.from_numpy(res))
    end = time.perf_counter()
    print("iree prediction")
    print('Elapsed time:', (end - start)/times)

    start = time.perf_counter()
    for i in range(0, times):
        prediction = top3_possibilities(torch.from_numpy(jit_func(img.numpy())))
    end = time.perf_counter()
    print("torch-mlir prediction")
    print('Elapsed time:', (end - start)/times)

    start = time.perf_counter()
    for i in range(0, times):
        golden_prediction = top3_possibilities(torch_func(img))
    end = time.perf_counter()
    print("PyTorch prediction")
    print('Elapsed time:', (end - start)/times)


image_url = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"

print("load image from " + image_url, file=sys.stderr)
# img = load_and_preprocess_image(image_url)
img = load_and_preprocess_image_path("./YellowLabradorLooking_new.jpg")
print("load labels from " +  
        "https://raw.githubusercontent.com/cathyzhyi/ml-data/main/imagenet-classes.txt")
labels = load_labels()

@make_simple_dynamo_backend
def refbackend_torchdynamo_backend(fx_graph: torch.fx.GraphModule,
                                   example_inputs: List[torch.Tensor]):
    mlir_module = torch_mlir.compile(
        fx_graph, example_inputs, output_type="linalg-on-tensors")
    backend = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(mlir_module)
    loaded = backend.load(compiled)

    def compiled_callable(*inputs):
        inputs = [x.numpy() for x in inputs]
        result = loaded.forward(*inputs)
        if not isinstance(result, tuple):
            result = torch.from_numpy(result)
        else:
            result = tuple(torch.from_numpy(x) for x in result)
        return result
    return compiled_callable

resnet18 = models.resnet18(pretrained=True)
resnet18.train(False)
dynamo_callable = dynamo.optimize(refbackend_torchdynamo_backend)(resnet18)

predictions(resnet18.forward, lambda x: dynamo_callable(torch.from_numpy(x)).detach().numpy(), resnet18, img, labels)
profiling(resnet18.forward, lambda x: dynamo_callable(torch.from_numpy(x)).detach().numpy(), resnet18, img, labels, times=20)

