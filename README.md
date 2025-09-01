 DDP for Multi-Agent Double DQN on Intel GPUs with oneCCL/OFI (single-node, multi-GPU)

 oneAPI Collective Communications Library (oneCCL) provides high-performance implementations of collective operations—all-reduce, all-gather, reduce-scatter, and all-to-all—for deep learning workloads. Its transport layer is designed to run over MPI or OFI (libfabric). In PyTorch, the oneccl_bindings_for_pytorch package exposes a C10D ProcessGroup; by selecting backend="ccl", DDP/FSDP collectives are executed on top of oneCCL.

The Abstract Transport Layer (ATL) in oneCCL is selected via CCL_ATL_TRANSPORT and can be set to ofi or mpi. With ofi, you choose a libfabric provider via FI_PROVIDER (e.g., tcp, verbs, psm3) to match the cluster interconnect. For single-node runs or when minimizing dependencies, ofi + tcp is robust; with InfiniBand use verbs, and with Omni-Path use psm3. Key-value store (KVS) bootstrapping is handled within oneCCL (PMI/MPI modes), but in practice using PyTorch’s init_method="env://"—environment variables populated by torchrun or a scheduler—is the simplest approach.

On Intel GPUs, GPU-buffer collectives proceed in two phases: scale-up (intra-node GPU↔GPU) and scale-out (inter-node), with the former leveraging Level Zero device paths. On the PyTorch side you run on the XPU device type and specify backend="ccl"; the Level Zero path is used transparently. Application code can focus on DDP usage without dealing with low-level transfer paths or memory registration.

oneCCL advances communication via dedicated worker threads. The number of workers is controlled with CCL_WORKER_COUNT, and CPU pinning with CCL_WORKER_AFFINITY. Under Slurm (or any cgroup-constrained CPU set), pinning workers to cores outside the allowed set will fail (e.g., pthread_create(22)). Use core IDs within /proc/self/status → Cpus_allowed_list. For single-node DL/RL, start with one worker and scale up only if needed. Pinning the application processes themselves (e.g., srun --cpu-bind=cores) helps avoid affinity mismatches.

Initialize DDP once per process. Assign the device with torch.xpu.set_device(local_rank), import oneccl_bindings_for_pytorch, then create the process group:
dist.init_process_group(backend="ccl", init_method="env://")
With env://, the launcher provides MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE (and LOCAL_RANK). PyTorch forms the process group from these. After initialization, move the model to the device and wrap it with DistributedDataParallel before training. If WORLD_SIZE=1 and you are not using DDP, do not initialize a process group; also avoid starting oneCCL workers unnecessarily.

For single-node OFI/TCP, setting MASTER_ADDR=127.0.0.1 ensures communication uses the OS TCP/IP loopback interface; packets never leave the host and the setup is very reliable. When optimizing performance for a given fabric, switch FI_PROVIDER to verbs (InfiniBand) or psm3 (Omni-Path) to reduce latency and improve throughput.

In DDP, gradient synchronization happens during the backward pass. DDP installs autograd hooks on parameters; as soon as a parameter’s gradient is computed, it is placed into a bucket. When a bucket fills, an asynchronous all-reduce is launched via oneCCL, overlapping communication with the remaining backward compute. Assuming ring all-reduce, per step and per rank the data volume is roughly:

bytes moved  ≈  2 (N−1)N×(total gradient size in bytes)

where N is the number of ranks. Consequently, if the model is small and update frequency is high, communication can dominate. Mitigations include gradient accumulation with no_sync() to reduce synchronization frequency, increasing the effective batch size to better hide communication under compute, low-precision communication (bf16/fp16) via a communication hook to cut bytes on the wire, tuning PyTorch’s bucket_cap_mb, and stabilizing the autograd graph with find_unused_parameters=False and static_graph=True where applicable.
