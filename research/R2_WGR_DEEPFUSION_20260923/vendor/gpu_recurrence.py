"""Exact-order inference SSM recurrence in one CUDA kernel, compiled by NVRTC.

Original normalization, rate/proposal/mix projections, nonlinearities, output
projection and dropout stay in PyTorch on the original module/weights. Only
the time recurrence is fused. No extension build, CuPy, training, parameter
change, reassociation, parallel scan or lower-precision accumulator is used.

The mixed-dtype contract is deliberately strict: normalized/alpha FP32,
candidate/mix BF16. In particular (1 - mix) is evaluated by PyTorch in BF16
before conversion to FP32; doing that subtraction after promotion changes the
registered recurrence. Unsupported dtypes raise instead of silently changing
precision or falling back to an unreported algorithm.
"""
from __future__ import annotations

from pathlib import Path
import ctypes as C
import hashlib
import os
import threading

import torch
import torch.nn.functional as F

CUDA_ROOT = Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"))
CUDA_SOURCE = r'''
#include <cuda_runtime.h>

__device__ __forceinline__ float from_bf16(const unsigned short value) {
    return __uint_as_float(((unsigned int)value) << 16);
}

extern "C" __global__ void ssm_ordered_bf16_fp32(
    const float* __restrict__ normalized,
    const float* __restrict__ alpha,
    const unsigned short* __restrict__ candidate,
    const unsigned short* __restrict__ mix,
    const float* __restrict__ one_minus_mix,
    float* __restrict__ sequence,
    const int batch,
    const int time,
    const int features
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= batch * features) return;
    const int b = lane / features;
    const int feature = lane - b * features;
    float state = 0.0f;
    for (int t = 0; t < time; ++t) {
        const int index = (b * time + t) * features + feature;
        const float a = alpha[index];
        const float a_complement = __fsub_rn(1.0f, a);
        const float keep = __fmul_rn(a_complement, state);
        const float proposal = __fmul_rn(a, from_bf16(candidate[index]));
        state = __fadd_rn(keep, proposal);
        const float recurrent = __fmul_rn(from_bf16(mix[index]), state);
        const float direct = __fmul_rn(one_minus_mix[index], normalized[index]);
        sequence[index] = __fadd_rn(recurrent, direct);
    }
}
'''


class CUDARecurrenceError(RuntimeError): pass


def _bind(library, name, argtypes, restype=C.c_int):
    fn=getattr(library,name)
    fn.argtypes=argtypes
    fn.restype=restype
    return fn


class _Kernel:
    """One compiled CUDA module owned by one current CUDA device context."""
    def __init__(self,device):
        if os.name!="nt":raise CUDARecurrenceError("This NVRTC loader targets the verified Windows CUDA environment")
        self.device=torch.device(device)
        self.dll_directory=os.add_dll_directory(str(CUDA_ROOT/"bin"))
        self.nvrtc=C.WinDLL(str(CUDA_ROOT/"bin/nvrtc64_120_0.dll"))
        self.driver=C.WinDLL("nvcuda.dll")
        self.create=_bind(self.nvrtc,"nvrtcCreateProgram",[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p])
        self.compile=_bind(self.nvrtc,"nvrtcCompileProgram",[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)])
        self.log_size=_bind(self.nvrtc,"nvrtcGetProgramLogSize",[C.c_void_p,C.POINTER(C.c_size_t)])
        self.log=_bind(self.nvrtc,"nvrtcGetProgramLog",[C.c_void_p,C.c_void_p])
        self.ptx_size=_bind(self.nvrtc,"nvrtcGetPTXSize",[C.c_void_p,C.POINTER(C.c_size_t)])
        self.ptx=_bind(self.nvrtc,"nvrtcGetPTX",[C.c_void_p,C.c_void_p])
        self.destroy=_bind(self.nvrtc,"nvrtcDestroyProgram",[C.POINTER(C.c_void_p)])
        self.load=_bind(self.driver,"cuModuleLoadData",[C.POINTER(C.c_void_p),C.c_void_p])
        self.function=_bind(self.driver,"cuModuleGetFunction",[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self.launch=_bind(self.driver,"cuLaunchKernel",[C.c_void_p,
            C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,
            C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p])
        self.error_name=_bind(self.driver,"cuGetErrorName",[C.c_int,C.POINTER(C.c_char_p)])
        self.module=C.c_void_p()
        self.handle=C.c_void_p()
        capability=torch.cuda.get_device_capability(self.device)
        self.options=[f"--gpu-architecture=compute_{capability[0]}{capability[1]}",
            "--std=c++14","--fmad=false","--ftz=false","--prec-div=true","--prec-sqrt=true",
            "--include-path="+str(CUDA_ROOT/"include")]
        program=C.c_void_p()
        code=self.create(C.byref(program),CUDA_SOURCE.encode(),b"ordered_ssm.cu",0,None,None)
        if code:raise CUDARecurrenceError(f"nvrtcCreateProgram failed: {code}")
        try:
            opts=(C.c_char_p*len(self.options))(*[s.encode() for s in self.options])
            result=self.compile(program,len(self.options),opts)
            size=C.c_size_t()
            self.log_size(program,C.byref(size))
            buffer=C.create_string_buffer(max(1,size.value))
            self.log(program,buffer)
            self.compiler_log=buffer.value.decode(errors="replace")
            if result:raise CUDARecurrenceError(f"NVRTC compilation failed ({result}): {self.compiler_log}")
            self._check_nvrtc(self.ptx_size(program,C.byref(size)),"nvrtcGetPTXSize")
            self.ptx_buffer=C.create_string_buffer(size.value)
            self._check_nvrtc(self.ptx(program,self.ptx_buffer),"nvrtcGetPTX")
        finally:
            self.destroy(C.byref(program))
        self._check(self.load(C.byref(self.module),self.ptx_buffer),"cuModuleLoadData")
        self._check(self.function(C.byref(self.handle),self.module,b"ssm_ordered_bf16_fp32"),"cuModuleGetFunction")
        self.launch_count=0

    @staticmethod
    def _check_nvrtc(code,label):
        if code:raise CUDARecurrenceError(f"{label} failed: {code}")

    def _check(self,code,label):
        if code:
            name=C.c_char_p()
            self.error_name(code,C.byref(name))
            raise CUDARecurrenceError(f"{label}: {(name.value or b'unknown').decode()} ({code})")

    def __call__(self,normalized,alpha,candidate,mix,one_minus_mix):
        output=torch.empty_like(normalized,dtype=torch.float32)
        batch,time,slots,width=normalized.shape
        features=slots*width
        stream=torch.cuda.current_stream(normalized.device)
        arguments=[C.c_void_p(t.data_ptr()) for t in
                   (normalized,alpha,candidate,mix,one_minus_mix,output)]
        arguments += [C.c_int(batch),C.c_int(time),C.c_int(features)]
        pointer_array=(C.c_void_p*len(arguments))(*[C.cast(C.pointer(arg),C.c_void_p) for arg in arguments])
        block=256
        grid=(batch*features+block-1)//block
        self._check(self.launch(self.handle,grid,1,1,block,1,1,0,
            C.c_void_p(stream.cuda_stream),pointer_array,None),"cuLaunchKernel")
        for tensor in (normalized,alpha,candidate,mix,one_minus_mix,output):tensor.record_stream(stream)
        self.launch_count+=1
        return output


_KERNELS={}
_COMPILE_LOCK=threading.Lock()


def _get_kernel(device):
    device=torch.device(device)
    index=torch.cuda.current_device() if device.index is None else device.index
    with _COMPILE_LOCK:
        if index not in _KERNELS:
            with torch.cuda.device(index):_KERNELS[index]=_Kernel(torch.device("cuda",index))
        return _KERNELS[index]


def ordered_recurrence(normalized,alpha,candidate,mix):
    """Run the exact sequential update for contiguous B×T×slot×feature inputs."""
    if torch.is_grad_enabled():raise CUDARecurrenceError("Fused recurrence is inference-only; use torch.inference_mode()")
    values=(normalized,alpha,candidate,mix)
    expected=(torch.float32,torch.float32,torch.bfloat16,torch.bfloat16)
    if normalized.ndim!=4 or normalized.shape[1]<1:raise CUDARecurrenceError("Expected nonempty B×T×slot×feature tensors")
    if any(t.device.type!="cuda" or t.device!=normalized.device for t in values):
        raise CUDARecurrenceError("All recurrence operands must share one CUDA device")
    if any(t.shape!=normalized.shape for t in values):raise CUDARecurrenceError("Recurrence operand shapes differ")
    if tuple(t.dtype for t in values)!=expected:
        raise CUDARecurrenceError(f"Unsupported recurrence dtypes: {[str(t.dtype) for t in values]}; expected FP32/FP32/BF16/BF16")
    # This subtraction intentionally precedes FP32 promotion. It matches the
    # original Tensor reverse-subtraction in BF16, including its rounding.
    one_minus_mix=(1.0-mix).float().contiguous()
    normalized,alpha,candidate,mix=[t.contiguous() for t in values]
    return _get_kernel(normalized.device)(normalized,alpha,candidate,mix,one_minus_mix)


class FusedTemporalSelectiveStateSpace:
    def __init__(self,module):
        self.module=module
        self.original_forward=module.forward
        self.calls=0
        self.last_dtypes=None

    def __call__(self,value,dt_hours):
        if self.module.training or torch.is_grad_enabled():
            raise CUDARecurrenceError("Fused original SSM supports frozen inference only")
        normalized=self.module.norm(value)
        rate=F.softplus(self.module.rate(normalized)).clamp_max(20.0)
        dt=dt_hours[:,:,None,None].to(normalized.dtype)
        alpha=-torch.expm1(-(rate*dt).clamp_max(20.0))
        candidate=torch.tanh(self.module.proposal(normalized))
        mix=torch.sigmoid(self.module.mix(normalized))
        self.last_dtypes={name:str(t.dtype) for name,t in
            (("value",value),("normalized",normalized),("rate",rate),("alpha",alpha),("candidate",candidate),("mix",mix))}
        sequence=ordered_recurrence(normalized,alpha,candidate,mix)
        self.calls+=1
        return value+self.module.dropout(self.module.output(sequence))


def install(model):
    """Replace only the ten original SSM callables, preserving module weights."""
    if model.training:raise CUDARecurrenceError("Model must be in eval mode")
    if not torch.are_deterministic_algorithms_enabled():
        raise CUDARecurrenceError("Enable the declared deterministic runtime before installation")
    if os.environ.get("CUDA_LAUNCH_BLOCKING")!="0":
        raise CUDARecurrenceError("Launch a fresh process with CUDA_LAUNCH_BLOCKING=0")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":
        raise CUDARecurrenceError("Launch with CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
    modules=[module for module in model.modules() if type(module).__name__=="TemporalSelectiveStateSpace"]
    if len(modules)!=10:raise CUDARecurrenceError(f"Expected ten original SSM modules, found {len(modules)}")
    if any(isinstance(module.forward,FusedTemporalSelectiveStateSpace) for module in modules):
        raise CUDARecurrenceError("SSM fusion is already installed")
    device=next(model.parameters()).device
    _get_kernel(device)  # Compile/load before timed inference, not inside capture.
    wrappers=[]
    for module in modules:
        wrapper=FusedTemporalSelectiveStateSpace(module)
        module.forward=wrapper
        wrappers.append(wrapper)
    return wrappers


def uninstall(wrappers):
    for wrapper in wrappers:wrapper.module.forward=wrapper.original_forward


def provenance():
    return {"implementation":"one CUDA thread per batch/slot/feature, sequential C loop over time",
        "compiled_source_sha256":hashlib.sha256(CUDA_SOURCE.encode()).hexdigest(),
        "file_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "weights_changed":False,"model_layers_changed":False,"dtype_change":False,
        "one_minus_mix":"original BF16 PyTorch subtraction before FP32 promotion",
        "recurrence_arithmetic":"FP32 round-to-nearest separate subtract/multiply/add; fmad disabled",
        "cuda_launch_blocking":os.environ.get("CUDA_LAUNCH_BLOCKING"),
        "cublas_workspace_config":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        "devices":{str(index):{"options":kernel.options,"launches":kernel.launch_count,
                              "compiler_log":kernel.compiler_log} for index,kernel in _KERNELS.items()}}
