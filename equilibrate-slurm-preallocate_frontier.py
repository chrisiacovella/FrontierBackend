'''
this was tested on Frontier, which uses AMD GPUs and SLURM.

A few things are different than when running on an NVIDIA GPU cluster, specifically related to
the method for multiprocessing.

'''

import os
import pickle
import logging
import multiprocessing
import click

# We use openmmtools to figure out the platforms we can run on; this will cause a `hipErrorNoDevice` error when
# we actually try to go run a process (since the HIP runtime has already been touched by the parent).
# To fix this, we can change the starting method to `spawn`  that will create a fresh process instead of a
# copy of the  parent that has already been touched.

try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    # already set -- e.g. if this module gets re-imported by a spawned child, do nothing
    pass

from dask_jobqueue.slurm import SLURMRunner
from dask.distributed import Client

from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.evaluator.client import RequestOptions, EvaluatorClient, ConnectionOptions
from openff.evaluator.backends import ComputeResources
from openff.evaluator.storage import LocalFileStorage
from openff.evaluator.server.server import EvaluatorServer
from openff.evaluator.forcefield import SmirnoffForceFieldSource

from dask_existing_cluster_backend import DaskExistingClusterBackend




@click.command()
@click.option("--dataset_path","--dataset", "-d", default="training_set.json")
@click.option("--force-field", "-f", default="openff-2.2.1.offxml")
@click.option("--port", "-p", default=8000)
@click.option(
    "--n-workers",
    default=8,
    help="Number of dask workers requested. Must match TOTAL_WORKERS in "
    "slurm submission script -- used here only to size wait_for_workers().",
)
def main(dataset_path, force_field, port, n_workers):
    # Rank 0 becomes the scheduler and blocks inside this context
    # manager forever. Every other rank except SLURM_PROCID==1 becomes
    # a worker and also blocks here forever. Only SLURM_PROCID==1 (the
    # "client" rank) continues past this block -- so everything below
    # only ever runs once, on exactly one rank.
    #
    # Verify these kwarg names against your installed dask_jobqueue
    # version (`pip show dask_jobqueue`); worker_options in particular
    # may need adjusting (e.g. resources, nthreads) depending on version.
    with SLURMRunner(
        scheduler_file="scheduler-{job_id}.json",
        worker_options={"nthreads": 1, "resources": {"GPU": 1}},
    ) as runner:
        with Client(runner) as client:
            client.wait_for_workers(n_workers)
            run_calculation(client, dataset_path, force_field, port)


def run_calculation(client, dataset_path, force_field, port):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    dataset = PhysicalPropertyDataSet.from_json(dataset_path)
    logger.info(f"Loaded {len(dataset.properties)} properties from {dataset_path}")

    options = RequestOptions.from_json("options.json")
    force_field_source = SmirnoffForceFieldSource.from_path(force_field)

    worker_resources = ComputeResources(
        number_of_threads=1,
        number_of_gpus=1,
        preferred_gpu_toolkit=ComputeResources.GPUToolkit.HIP,
    )

    # Reuses the already-connected client from SLURMRunner rather than
    # attaching a new one from a scheduler file/address.
    backend = DaskExistingClusterBackend(
        client=client,
        resources_per_worker=worker_resources,
    )
    backend.start()
    logger.info(f"backend started {backend}")

    server = EvaluatorServer(
        calculation_backend=backend,
        working_directory="working-directory",
        delete_working_files=False,
        enable_data_caching = True,
        storage_backend=LocalFileStorage(cache_objects_in_memory=True),
        port=port,
    )
    server.start(asynchronous=True)

    evaluator_client = EvaluatorClient(
        connection_options=ConnectionOptions(server_port=port)
    )

    request, error = evaluator_client.request_estimate(
        dataset, force_field_source, options
    )
    logger.info(f"Request ID: {request.id}")

    results, exception = request.results(synchronous=True, polling_interval=30)
    assert exception is None

    logger.info("Equilibration complete")
    logger.info(f"# estimated: {len(results.estimated_properties)}")
    logger.info(f"# equilibrated: {len(results.equilibrated_properties)}")
    logger.info(f"# unsuccessful: {len(results.unsuccessful_properties)}")
    logger.info(f"# exceptions: {len(results.exceptions)}")

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()

