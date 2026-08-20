"""
A calculation backend that attaches to an already running dask cluster, rather than submitting new jobs.

On certain systems, e.g., Frontier, max wall time is depending on the number of nodes requested by
a job.  This allows us to run on a pre-allocated set of nodes

"""
import logging

from dask import distributed

from openff.evaluator.backends import CalculationBackend, ComputeResources
from openff.evaluator.backends.dask import BaseDaskBackend, _Multiprocessor

logger = logging.getLogger(__name__)


class DaskExistingClusterBackend(BaseDaskBackend):
    """A backend which attaches to an existing dask scheduler/worker
    pool instead of spinning up and submitting SLURM jobs for itself.
    """

    def __init__(
        self,

        client=distributed.Client,
        resources_per_worker=ComputeResources(),
        number_of_workers=1,
    ):
        """
        Parameters
        ----------
        client: distributed.Client
            An already-connected client, e.g. the one returned by
            `dask.distributed.Client(runner)` when using
            `dask_jobqueue.slurm.SLURMRunner`.
        resources_per_worker: ComputeResources
            Only used to tell the evaluator how many GPUs/threads each
            worker has access to -- it does NOT cause any resources to
            be requested from SLURM (that already happened in the
            submission script).
        number_of_workers: int
            Informational only; used by the base class for bookkeeping.
        """
        super().__init__(number_of_workers, resources_per_worker)

        self._external_client = client

    def start(self):

        CalculationBackend.start(self)

        self._client = self._external_client

        n_workers = len(self._client.scheduler_info()["workers"])
        logger.info(f"Attached to existing dask cluster with {n_workers} worker(s)")

    def stop(self):
        # Do NOT close the client/cluster here as it is owned by the SLURM job script that we are hooking up to
        pass

    @staticmethod
    def _wrapped_function(function, *args, **kwargs):
        available_resources = kwargs["available_resources"]
        kwargs.pop("gpu_assignments", None)
        kwargs.pop("per_worker_logging", None)

        from distributed import get_worker

        # add in some logging:
        per_worker_logging = kwargs.pop("per_worker_logging", False)
        if per_worker_logging:
            import os
            os.makedirs("worker-logs", exist_ok=True)
            kwargs["logger_path"] = os.path.join(
                "worker-logs", f"{get_worker().id}.log"
            )

        if available_resources.number_of_gpus > 0:
            # Inside the worker process the GPU always appears as device 0.
            available_resources._gpu_device_indices = "0"

        return _Multiprocessor.run(function, *args, **kwargs)

    def submit_task(self, function, *args, **kwargs):
        from openff.evaluator.workflow.plugins import registered_workflow_protocols

        key = kwargs.pop("key", None)

        protocols_to_import = [
            protocol_class.__module__ + "." + protocol_class.__qualname__
            for protocol_class in registered_workflow_protocols.values()
        ]



        # `resources={"GPU": 1}` tells the dask scheduler this task may
        # only run on a worker that has registered GPU capacity
        # Any task beyond the number of available resources just get queued up.
        submit_kwargs = {}
        if self._resources_per_worker.number_of_gpus > 0:
            submit_kwargs["resources"] = {"GPU": 1}

        return self._client.submit(
            DaskExistingClusterBackend._wrapped_function,
            function,
            *args,
            **kwargs,
            available_resources=self._resources_per_worker,
            registered_workflow_protocols=protocols_to_import,
            per_worker_logging=True,
            key=key,
            **submit_kwargs,
        )
