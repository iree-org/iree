import pytest

def pytest_addoption(parser):
    parser.addoption("--goldentime-rocm-e2e-ms", action="store", type=float, help="Golden time to test benchmark")
    parser.addoption("--goldentime-rocm-unet-ms", action="store", type=float, help="Golden time to test benchmark")
    parser.addoption("--goldentime-rocm-clip-ms", action="store", type=float, help="Golden time to test benchmark")
    parser.addoption("--goldentime-rocm-vae-ms", action="store", type=float, help="Golden time to test benchmark")
    parser.addoption("--goldendispatch-rocm-unet", action="store", default=1718, type=int, help="Golden dispatch count to test benchmark")
    parser.addoption("--goldendispatch-rocm-clip", action="store", default=1571, type=int, help="Golden dispatch count to test benchmark")
    parser.addoption("--goldendispatch-rocm-vae", action="store", default=250, type=int, help="Golden dispatch count to test benchmark")
    parser.addoption("--goldensize-rocm-unet-bytes", action="store", default=2088217, type=int, help="Golden vmfb size to test benchmark")
    parser.addoption("--goldensize-rocm-clip-bytes", action="store", default=785493, type=int, help="Golden vmfb size to test benchmark")
    parser.addoption("--goldensize-rocm-vae-bytes", action="store", default=762067, type=int, help="Golden vmfb size to test benchmark")
    parser.addoption("--gpu-number", action="store", default=0, type=int, help="IREE GPU device number to test on")
    parser.addoption("--rocm-chip", action="store", default="gfx90a", type=str, help="ROCm target chip configuration of GPU")

@pytest.fixture
def goldentime_rocm_e2e(request):
    return request.config.getoption("--goldentime-rocm-e2e-ms")

@pytest.fixture
def goldentime_rocm_unet(request):
    return request.config.getoption("--goldentime-rocm-unet-ms")

@pytest.fixture
def goldentime_rocm_clip(request):
    return request.config.getoption("--goldentime-rocm-clip-ms")

@pytest.fixture
def goldentime_rocm_vae(request):
    return request.config.getoption("--goldentime-rocm-vae-ms")

@pytest.fixture
def goldendispatch_rocm_unet(request):
    return request.config.getoption("--goldendispatch-rocm-unet")

@pytest.fixture
def goldendispatch_rocm_clip(request):
    return request.config.getoption("--goldendispatch-rocm-clip")

@pytest.fixture
def goldendispatch_rocm_vae(request):
    return request.config.getoption("--goldendispatch-rocm-vae")

@pytest.fixture
def goldensize_rocm_unet(request):
    return request.config.getoption("--goldensize-rocm-unet-bytes")

@pytest.fixture
def goldensize_rocm_clip(request):
    return request.config.getoption("--goldensize-rocm-clip-bytes")

@pytest.fixture
def goldensize_rocm_vae(request):
    return request.config.getoption("--goldensize-rocm-vae-bytes")

@pytest.fixture
def rocm_chip(request):
    return request.config.getoption("--rocm-chip")

@pytest.fixture
def gpu_number(request):
    return request.config.getoption("--gpu-number")
