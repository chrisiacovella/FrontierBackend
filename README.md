# FrontierBackend
OpenFF Evaluator Backend designed to work with Frontier

This backend is designed such that we pre-allocate all the resources we need for a given set of calculations, 
rather than having dask submit jobs to the cluster on demand. 

This is done as the wall-time limits are a function of the number of nodes requested. 

This also requires some changes to the scripts that actually run the calculations and the slurm submission scripts; 
it is not just a one step drop-in replacement.

A few changes that must be made to scripts:

Need to set the multiprocessing method to spawn. This is required because we use openmmtools to figure out 
the platforms we can run on, but once the the GPU has been touched by openmmtools, we will get an error
`hipErrorNoDevice` unless we use the spawn method to create a fresh process.

```Python
import multiprocessing

try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    # already set -- e.g. if this module gets re-imported by a spawned child, do nothing
    pass

```

The structure of the `main` function is also changed; here we now create a context manager for `SLURMRunner` 
(which is a dask_jobqueue class), which calls a wrapper for the functions that had previously been in `main`. 

The wrapper function changes slightly: we no longer need to setup an instance of `QueueWorkerResources`, but rather 
an instance of `ComputeResources` (since we will not be spawning new workers, but rather using the resources we have already allocated).
We replace the `DaskSlurmBackend` with our new `DaskExistingClusterBackend`. 

The key portions of the wrapper function are shown below:

```Python
    worker_resources = ComputeResources(
        number_of_threads=n_threads,
        number_of_gpus=n_gpus,
        preferred_gpu_toolkit=ComputeResources.GPUToolkit[gpu_toolkit],
    )

    # Attaches to the already-connected client from SLURMRunner instead of
    # submitting new SLURM jobs the way DaskSLURMBackend did.
    backend = DaskExistingClusterBackend(
        client=client,
        resources_per_worker=worker_resources,
    )
    backend.start()
    logger.info(f"backend started {backend}")

    # rename any previous log files
    rename_log_file(log_file)

    server = EvaluatorServer(
        calculation_backend=backend,
        working_directory=working_directory,
        port=port,
        enable_data_caching=enable_data_caching,
        delete_working_files=False,  
        storage_backend=LocalFileStorage(cache_objects_in_memory=True),
    )   
```

Submitting jobs to the cluster use srun to call the python script, since the script will not longer be spawning new 
workers.  

Notes: 
- Two of our requested cores will go to the dask scheduler/server. Srun will get the total number of cores requested, n-workers we pass to the python script will be two less.   
- For frontier, we need to load the `rocm` module, even thought we are using the openmm conda install (which does install `rocm` libraries).  If we we do not load the `rocm` module, we will not be able to see the GPUs. 


An example script for frontier is below, which has 8 GPUs per node:

```bash
#!/bin/bash
#SBATCH -J benchmark_1
#SBATCH -t 0-2:00:00
#SBATCH --nodes=1
#SBATCH -A PROJECT_ID <-- replace with your project ID
#SBATCH --output slurm-%x.%A.out

# Get the total resources we have been allocated for this script, which will be passed to srun
TOTAL_RESOURCES=$(( SLURM_JOB_NUM_NODES * 8))

# The TOTAL_WORKERS will be 2 less the total resources, as we need 2 workers for dask server and scheduler.
TOTAL_WORKERS=$(( TOTAL_RESOURCES - 2 ))

module load miniforge3
module load rocm/6.2.4 # for some reason loading rocm/7x does not show up the HIP platform
                        # despite the fact that rocm-core = 7.0.2 is what is in the conda environment for openmm 8.5
conda activate ash-sage-refit-frontier1

PORT=8217

rocm-smi

export DASK_DISTRIBUTED__COMM__INTERFACE="ib0"

srun --overlap --ntasks="$TOTAL_RESOURCES" --ntasks-per-node=8\
    python equilibrate-slurm-preallocate_frontier.py \
        --n-workers "$TOTAL_WORKERS" \
        --port                      "$PORT"                             \
        --dataset                   "./training-set.json"                       \
        --force-field               "./force-field.offxml"                     \


```