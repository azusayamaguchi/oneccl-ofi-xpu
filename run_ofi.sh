#!/bin/bash -l
#SBATCH --job-name=ofi
#SBATCH --partition=pvc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4                 
#SBATCH --cpus-per-task=24       # DataLoader 6  GPU 4 : 6×4=24
#SBATCH --mem=32G               
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -u -o pipefail
set +u
set +e
module purge
module load default-dawn
module load intel-oneapi-compilers/2025.0.3/gcc/sb5vj5us
module load intelpython-conda/2025.0
module load intel-oneapi-mpi/2021.11.0/oneapi/h7nq7sah
conda activate pytorch-gpu-2.3.1
set -e
set -u

export IPEX_LIB_DIR=$(python - <<'PY'
import pathlib
import intel_extension_for_pytorch as ipex
print(str(pathlib.Path(ipex.__file__).parent / "lib"))
PY
)
export LD_LIBRARY_PATH="$IPEX_LIB_DIR:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
ls -l "$IPEX_LIB_DIR/libintel-ext-pt-gpu.so"

export CCL_ATL_TRANSPORT=ofi
#export CCL_ATL_TRANSPORT=mpi
export FI_PROVIDER=tcp
export CCL_WORKER_COUNT=1
unset  CCL_WORKER_AFFINITY    
export CCL_LOG_LEVEL=warn


# OpenMP/MKL 
export KMP_AFFINITY=disabled
export KMP_BLOCKTIME=0

#Memory
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

# Intel GPU + Level Zero
unset SYCL_DEVICE_FILTER
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=0
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0:2
export SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY=1

# stack unlimited
ulimit -s unlimited

ALLOWED=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)
FIRST=$(echo "$ALLOWED" | awk -F, '{print $1}' | awk -F- '{print $1}')
export CCL_WORKER_AFFINITY=${FIRST}
# to avoid the error from init_method="env://"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"

srun --cpu-bind=cores -n 1 -c ${SLURM_CPUS_PER_TASK} \
python - <<'PY'
import os, torch, torch.distributed as dist
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch as ccl
print("ipex ok; ccl:", getattr(ccl, "__version__", "unknown"))
print("xpu count:", getattr(torch, "xpu", None) and torch.xpu.device_count())
os.environ.setdefault("RANK","0"); os.environ.setdefault("WORLD_SIZE","1")
dist.init_process_group(backend="ccl", init_method="env://")
print("Barrier")
dist.barrier()
print("ccl barrier ok")
dist.destroy_process_group()
PY
