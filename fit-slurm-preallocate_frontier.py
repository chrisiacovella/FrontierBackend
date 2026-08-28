'''
This was testing on the IRIS cluster which features NVIDIA GPUS and runs slurm.

'''
"""
This script executes a distributed ForceBalance optimization using the OpenFF Evaluator.
It sets up the environment, prepares the ForceBalance input, and runs the optimization
on a SLURM cluster using Dask for distributed computing.
\b
It does the following steps:
- sets up ForceBalance options and writes them to `targets/phys-prop/options.json`
- checks if a previous run can be continued, and prepares the restart
- renames any previous log files to avoid overwriting
- sets up a Dask SLURM backend with the specified resources
- starts an Evaluator server to handle requests
- runs ForceBalance with the specified input file and logs the output to force_balance.log
\b
This run generates the following output:
- force_balance.log: The log file for the ForceBalance run
- results/: the directory with resulting force field
- optimize.tmp: directory with training properties calculated from intermediate force fields
- optimize.sav: the saved state of the ForceBalance optimization, if it was continued
- optimize.bak: a backup of the ForceBalance optimization state
- working-directory/: the directory with the working files for the Evaluator server
- worker-logs/: the directory with the logs for the Dask workers
- conda-env.yaml: the conda environment used for the run
- slurm-*.out: the SLURM output files for the workers

The option settings are as follows:
- equilibration runs for up to 1 ns, with 15 uncorrelated samples requested of density and potential energy
- production simulation runs for up to 10 ns, with 100 uncorrelated samples requested of density and enthalpies
"""

import logging
import sys
import subprocess
import os
import pathlib
import re
import shutil
import typing
import click
from click_option_group import optgroup

from dask_jobqueue.slurm import SLURMRunner
from dask.distributed import Client



from openff.units import unit
from openff.evaluator.server import EvaluatorServer
from openff.evaluator.backends import ComputeResources

from openff.evaluator.storage import LocalFileStorage

from openff.evaluator.layers.equilibration import EquilibrationProperty
from openff.evaluator.utils.observables import ObservableType
from openff.evaluator.properties import Density, EnthalpyOfMixing
from forcebalance.evaluator_io import Evaluator_SMIRNOFF

from dask_existing_cluster_backend import DaskExistingClusterBackend


logger = logging.getLogger(__name__)


def _remove_previous_files():
    """Remove any previous files that might interfere with the run."""
    restart_files = [
        # "optimize.tmp",
        "optimize.bak",
        "optimize.sav",
        "result",
        "worker-logs",
        "working-data",
    ]
    for restart_file in restart_files:

        path = pathlib.Path(restart_file)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            os.unlink(path)
        else:
            continue

        logger.info(f"Removing {restart_file}.")


def write_options(
        n_molecules: int = 1000,
        port: int = 8998,
        output_file: str = "targets/phys-prop/options.json"
):
    """
    Write the options for the ForceBalance optimization to a JSON file.
    This includes the equilibration properties, estimation options, and weights.
    The options are used by the Evaluator to run the optimization.
    """
    options = Evaluator_SMIRNOFF.OptionsFile()
    options.connection_options.server_port = port

    # set up request options
    potential_energy = EquilibrationProperty()
    potential_energy.observable_type = ObservableType.PotentialEnergy
    potential_energy.n_uncorrelated_samples = 15

    density = EquilibrationProperty()
    density.observable_type = ObservableType.Density
    density.n_uncorrelated_samples = 15

    options.estimation_options.calculation_layers = ["PreequilibratedSimulationLayer"]
    density_schema = Density.default_preequilibrated_simulation_schema(
        n_molecules=n_molecules,
        equilibration_error_tolerances=[potential_energy, density],
        equilibration_max_iterations=5,  # 5 * 200 ps = 1 ns
        equilibration_error_on_failure=False,
        n_uncorrelated_samples=100,
        max_iterations=5,  # 10 ns
        error_on_failure=False,

    )

    dhmix_schema = EnthalpyOfMixing.default_preequilibrated_simulation_schema(
        n_molecules=n_molecules,
        equilibration_error_tolerances=[potential_energy, density],
        equilibration_max_iterations=5,  # 5 * 200 ps = 1 ns
        equilibration_error_on_failure=False,
        n_uncorrelated_samples=100,
        max_iterations=5,  # 10 ns
        error_on_failure=False

    )

    options.estimation_options.add_schema(
        "PreequilibratedSimulationLayer", "Density", density_schema
    )
    options.estimation_options.add_schema(
        "PreequilibratedSimulationLayer", "EnthalpyOfMixing", dhmix_schema
    )

    # weights and denominators
    options.data_set_path = "training-set.json"
    options.weights["Density"] = 1.0
    options.weights["EnthalpyOfMixing"] = 1.0
    options.denominators["Density"] = 0.05 * unit.grams / unit.milliliters
    options.denominators["EnthalpyOfMixing"] = 1.6 * unit.kilojoules / unit.mole

    output_file = pathlib.Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        f.write(options.to_json())
    logger.info(f"Wrote options to {output_file}.")


def _prepare_restart(
        input_file: str = "optimize.in",
        target_name: str = "phys-prop",
):
    """
    Check for correct completed data and remove incomplete data from previous runs.

    This reads the input file to determine the maximum number of iterations,
    and checks for completed iterations. If any incomplete iterations are found,
    it removes them from the target directory.
    """

    # parse input file
    with open(input_file, "r") as file:
        content = file.read()

    # get max iterations
    try:
        maxstep = int(re.search(r"maxstep\s*(\d+)", content).groups()[0])
    except AttributeError:
        maxstep = 100  # default

    # check how many iterations have been completed
    incomplete_iteration_subdirs = [
        f"iter_{iteration:04d}"
        for iteration in range(maxstep)
    ]
    while incomplete_iteration_subdirs:
        subdir = incomplete_iteration_subdirs[0]
        target_directory = pathlib.Path("optimize.tmp") / target_name / subdir
        # has objective.p been written?
        if (target_directory / "objective.p").exists():
            # all's good
            incomplete_iteration_subdirs = incomplete_iteration_subdirs[1:]
        else:
            # check if mvals.txt and force-field.offxml exist
            if (
                    (target_directory / "mvals.txt").exists()
                    and (target_directory / "force-field.offxml").exists()
            ):
                # we can leave these, but need to remove any later directories
                incomplete_iteration_subdirs = incomplete_iteration_subdirs[1:]
            else:
                # start deleting from here
                break
    for subdir in incomplete_iteration_subdirs:
        target_directory = pathlib.Path("optimize.tmp") / target_name / subdir
        if target_directory.exists():
            shutil.rmtree(target_directory)
            logger.info(f"Removing {target_directory}.")


def rename_log_file(log_file):
    """Rename the log file to have the correct extension"""
    counter = 0
    original_log_file = pathlib.Path(log_file)
    log_file = pathlib.Path(log_file)
    if not log_file.exists():
        return
    while log_file.exists():
        counter += 1
        log_file = pathlib.Path(
            f"{original_log_file.stem}_{counter}{original_log_file.suffix}"
        )
    original_log_file.rename(log_file)
    logger.info(f"Renamed existing {original_log_file} to {log_file}.")



@click.command(help=__doc__)
@click.option(
    "--input",
    "input_file",
    type=str,
    default="optimize.in",
    help="The input file for ForceBalance",
)
@click.option(
    "--log",
    "log_file",
    type=str,
    default="force_balance.log",
    help="The output log file for ForceBalance",
)
@optgroup.group("Server configuration")
@optgroup.option(
    "--port",
    type=int,
    default=8000,
    help="The port for the server",
)
@optgroup.option(
    "--working-directory",
    type=str,
    default="working-directory",
    help="The working directory for the server",
)
@optgroup.option(
    "--enable-data-caching/--no-enable-data-caching",
    default=True,
    help="Enable data caching",
)
@optgroup.option(
    "--continue-run",
    type=bool,
    default=True,
    help="Continue a previous run",
)
@optgroup.group("Distributed configuration")
@optgroup.option(
    "--n-workers",
    type=int,
    default=1,
    help="Number of dask workers already running in this SLURM allocation "
    "(started by submit-fit.slurm via SLURMRunner). Must match TOTAL_WORKERS "
    "in that script -- used here only to size wait_for_workers().",
)
@optgroup.option(
    "--n-threads",
    type=int,
    default=1,
    help="The number of threads per worker",
)
@optgroup.option(
    "--n-gpus",
    type=int,
    default=1,
    help="The number of GPUs per worker; not sure changing this will work as this is hardcoded into the backend",
)
@optgroup.option(
    "--gpu-toolkit",
    type=click.Choice(["CUDA", "OpenCL", "HIP"]),
    default="HIP",
    help="The GPU toolkit to use",
)
def main(
        input_file: str = "optimize.in",
        log_file: str = "force_balance.log",
        # run args
        port: int = 8000,
        working_directory: str = "working-directory",
        enable_data_caching: bool = True,
        continue_run: bool = True,
        # distributed args
        n_workers: int = 1,
        n_threads: int = 1,
        n_gpus: int = 1,
        gpu_toolkit: typing.Literal["CUDA", "OpenCL", "HIP"] = "HIP",
):

    with SLURMRunner(
        scheduler_file="scheduler-{job_id}.json",
        worker_options={"nthreads": n_threads, "resources": {"GPU": n_gpus}},
        "nanny": False,

    ) as runner:
        with Client(runner) as client:
            client.wait_for_workers(n_workers)
            run_fit(
                client=client,
                input_file=input_file,
                log_file=log_file,
                port=port,
                working_directory=working_directory,
                enable_data_caching=enable_data_caching,
                continue_run=continue_run,
                n_threads=n_threads,
                n_gpus=n_gpus,
                gpu_toolkit=gpu_toolkit,
            )



def run_fit(
    client,
    input_file,
    log_file,
    port,
    working_directory,
    enable_data_caching,
    continue_run,
    n_threads,
    n_gpus,
    gpu_toolkit):

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # prepare ForceBalance arguments
    force_balance_arguments = ["ForceBalance.py", input_file]

    # check if we should continue
    if continue_run and pathlib.Path("optimize.sav").exists():
        _prepare_restart(input_file)
        force_balance_arguments = ["ForceBalance.py", "--continue", input_file]
    else:
        # _remove_previous_files()
        pass

    write_options(port=port)

    # Resources per worker are informational only here -- they tell the
    # evaluator/protocols how many GPUs/threads each worker has access to.
    # They do NOT cause anything to be requested from SLURM: that already
    # happened in submit-fit.slurm, and the scheduler/workers are already
    # running by the time this function is reached.
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
        delete_working_files=False,  # for debugging -- set true if too large
        # we need to cache for speed reasons
        # otherwise just querying the storage backend can take 10+ hours for 1k properties
        storage_backend=LocalFileStorage(cache_objects_in_memory=True),
    )

    with server:
        with open(log_file, "w") as file:
            subprocess.check_call(force_balance_arguments, stderr=file, stdout=file)


if __name__ == "__main__":
    main()