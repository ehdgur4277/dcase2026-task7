"""
ddp_util.py — exp07_AdaptorSplit 분산학습(DDP) 플러밍 (train.py 에서 분리)

torch.distributed 관련 로직 전부를 모은다. train.py 는 이 함수들만 호출하고
process group / broadcast / all-reduce 세부는 알지 못한다.

  setup_distributed       : env(WORLD_SIZE/RANK/LOCAL_RANK) → process group 초기화
  raw                     : DDP 래퍼 해제 (model.module)
  barrier                 : rank 동기화 배리어
  sync_model              : rank0 파라미터·버퍼를 전체로 브로드캐스트
  all_reduce_mean         : 스칼라 평균 (로그용 loss 등)
  broadcast_stop          : early-stop 플래그를 rank0→전체 전파
  broadcast_object_attrs  : 객체의 list-of-tensor 속성(예: DeltaGPAdaptor SVD 기저) 동기화
"""
import atexit
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    world_size  = int(os.environ.get('WORLD_SIZE',  '1'))
    rank        = int(os.environ.get('RANK',        '0'))
    local_rank  = int(os.environ.get('LOCAL_RANK',  '0'))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError('DDP는 CUDA GPU가 필요합니다.')
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl',
                                device_id=torch.device(f'cuda:{local_rank}'))
        atexit.register(
            lambda: dist.destroy_process_group() if dist.is_initialized() else None)
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    return rank, local_rank, world_size, distributed, device


def raw(model):
    return model.module if isinstance(model, DDP) else model


def barrier(distributed):
    if distributed and dist.is_initialized():
        dist.barrier()


def _broadcast_tensor(t, src=0):
    if t.is_contiguous():
        dist.broadcast(t, src=src)
    else:
        tmp = t.contiguous(); dist.broadcast(tmp, src=src); t.copy_(tmp)


def sync_model(model, distributed):
    if not distributed: return
    with torch.no_grad():
        for p in model.parameters(): _broadcast_tensor(p.data)
        for b in model.buffers():    _broadcast_tensor(b.data)


def all_reduce_mean(value, device, distributed):
    if not distributed: return value
    t = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / dist.get_world_size()


def broadcast_stop(stop, device, distributed):
    if not distributed: return stop
    t = torch.tensor([int(stop)], device=device)
    dist.broadcast(t, src=0)
    return bool(t.item())


def broadcast_object_attrs(obj, rank, distributed, device, attrs):
    """obj의 여러 list-of-tensor 속성을 rank0에서 브로드캐스트.
    DeltaGPAdaptor 의 SVD 기저(_d1_v/_d2_delta_v) 동기화에 사용."""
    if not distributed: return
    if rank == 0:
        container = []
        for attr in attrs:
            lst = getattr(obj, attr, [])
            container.append([t.cpu() if t is not None else None for t in lst])
    else:
        container = [None] * len(attrs)
    dist.broadcast_object_list(container, src=0)
    for attr, lst in zip(attrs, container):
        setattr(obj, attr,
                [t.to(device) if t is not None else None for t in lst])
