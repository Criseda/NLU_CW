**CSF3 (Computational Shared Facility 3) – SLURM Job Submission Documentation**  
(Updated for current SLURM usage on CSF3 at University of Manchester)

CSF3 now uses the **SLURM** batch scheduler (previously SGE/Grid Engine). Old `qsub` + `#$` jobscripts no longer work unchanged. Rewrite them with `#SBATCH` directives.

### 1. Basic Job Script Structure
```bash
#!/bin/bash --login
#SBATCH -J my_job_name                 # Job name (optional but recommended)
#SBATCH -o %x.o%j                      # Output file (mimics old SGE style)
#SBATCH -e %x.e%j                      # Error file
#SBATCH --time=01:00:00                # Walltime limit (required or default may apply)
#SBATCH --partition=serial             # Partition/queue (default = serial)
#SBATCH --ntasks=1                     # Number of tasks/cores (or -n 1)

# Your commands here
module load <your-module>
python my_script.py
```

Submit with:
```bash
sbatch my_job_script.sh
```

### 2. Available Partitions & How to Select Them
Use `#SBATCH --partition=<name>` (or `-p <name>`).

From official CSF3 SLURM config (manchester-CSF3.yaml):

| Partition   | Purpose                              | Cores / Node Range          | Nodes Range | Notes / Memory                              |
|-------------|--------------------------------------|-----------------------------|-------------|---------------------------------------------|
| `serial`    | Default single-core jobs             | 1                           | 1           | 4 GB/core default, max ~7 days              |
| `multicore` | Single-node parallel (OpenMP/MPI)    | 2 – 168                     | 1           | 4 GB/core, good for most parallel work      |
| `himem`     | High-memory single-node jobs         | 1 – 32                      | 1           | More RAM per core than multicore            |
| `hpcpool`   | Large-scale / high-core-count jobs   | Up to 1024 total cores      | Up to 124   | For very large distributed jobs             |

**Tips for selection**
- Single-core → omit `-p` or explicitly use `serial`.
- 2–40 cores on one node → `multicore`.
- Need lots of RAM → `himem`.
- Very large MPI jobs (>80 cores) → `hpcpool` (or `multinode` on CSF4 – check with `sinfo` on CSF3).
- Always set `--time` explicitly (default is only ~10 minutes in some configs).

### 3. Key SLURM Flags & Options

**Core & Node Resource Flags**
- `--ntasks=N` / `-n N` → total CPU cores/tasks (most common)
- `--nodes=N` / `-N N` → number of nodes (required for multi-node jobs)
- `--cpus-per-task=C` / `-c C` → CPUs per MPI task (for hybrid MPI+OpenMP)
- `--ntasks-per-node=M` → cores per node

**Memory Flags**
- `--mem=XXG` → total memory for the job
- `--mem-per-cpu=XXG` → memory per core (recommended)
- Default is ~4 GB per core on most partitions.

**GPU Flags** (still supported, legacy from old v100 nodes)
- `--gres=gpu:1` or `--gres=gpu:v100:1` (or `a100:1` if available)
- Often combined with a specific GPU partition/queue if one exists – check with `sinfo -o "%P %G"`.

**Time Limits**
- `--time=DD-HH:MM:SS` or `--time=HH:MM:SS`
- Short jobs (< few hours) → serial/multicore
- Max usually 7 days on standard partitions

**Job Arrays** (run many similar jobs)
```bash
#SBATCH --array=1-100%10     # 100 jobs, max 10 concurrent
# Inside script:
echo "Task ID = $SLURM_ARRAY_TASK_ID"
```
Environment variables:
- `$SLURM_ARRAY_TASK_ID`
- `$SLURM_ARRAY_JOB_ID`

**Dependencies**
```bash
#SBATCH --dependency=afterok:123456   # Run after job 123456 finishes OK
```

**Interactive Jobs**
```bash
srun --pty --partition=serial --ntasks=1 --time=02:00:00 bash
# or
srun --pty --gres=gpu:1 bash
```

**Other Useful Flags**
- `--exclusive` → request whole node(s)
- `--mail-type=ALL` + `--mail-user=you@manchester.ac.uk`
- `--export=ALL` (default, passes your environment)
- `--no-requeue` (if you don’t want automatic requeue on failure)

### 4. Common Job Types & Examples

**Serial (1 core)**
```bash
#SBATCH -J serial_job
#SBATCH --time=12:00:00
# no -p needed (default serial)
```

**Multicore OpenMP (single node)**
```bash
#SBATCH -p multicore
#SBATCH -n 16                  # 16 cores
export OMP_NUM_THREADS=$SLURM_NTASKS
```

**Hybrid MPI + OpenMP**
```bash
#SBATCH -p hpcpool
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -c 8                   # 8 OpenMP threads per MPI rank
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mpirun -n $SLURM_NTASKS my_mpi_app
```

**GPU Example (modern syntax)**
```bash
#SBATCH -p multicore           # or whatever partition allows GPUs
#SBATCH --gres=gpu:1
#SBATCH -n 8
module load cuda
```

### 5. Monitoring & Management Commands
- `sbatch job.sh` → submit
- `squeue -u $USER` → your jobs
- `squeue` → all jobs
- `scancel JOBID` → cancel
- `sacct -j JOBID --format=JobID,JobName,State,Elapsed,MaxRSS` → accounting
- `sinfo` → partition/node status
- `scontrol show partition` → detailed partition info

### 6. Migration Notes from Old SGE CSF3
- `#$ -pe smp.pe 8` → `#SBATCH -n 8`
- `#$ -l v100=1` → `#SBATCH --gres=gpu:1`
- `qsub` → `sbatch`
- `qstat` → `squeue`
- `$NSLOTS` → `$SLURM_NTASKS`
- `$SGE_TASK_ID` → `$SLURM_ARRAY_TASK_ID`

**Best Practice**
Always start with a small test job (`--time=00:10:00 --ntasks=2`). Use `sacct` to check actual resource usage before scaling up.

You can verify current partitions and limits directly on the CSF3 login node with:
```bash
sinfo -o "%P %c %m %l %G"   # partition, cores, memory, time limit, GPUs
```

This covers all common flags, options, and selection methods used on CSF3 today. If you need GPU-specific partitions or updated limits, run the `sinfo` commands above on the cluster. Happy computing!