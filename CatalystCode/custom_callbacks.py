from typing import Dict, List
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.utils.dict import split_dict_to_subdicts
from tensorboardX import SummaryWriter
from trains import Task

import os


class ILoggerCallback(Callback):
    """Logger callback interface, abstraction over logging step"""

    pass


class TbLogger(ILoggerCallback):
    """Logger callback, translates ``runner.metric_manager`` to tensorboard."""

    def __init__(
        self,
        metric_names: List[str] = None,
        log_on_batch_end: bool = True,
        log_on_epoch_end: bool = True,
    ):
        """
        Args:
            metric_names: list of metric names to log,
                if none - logs everything
            log_on_batch_end: logs per-batch metrics if set True
            log_on_epoch_end: logs per-epoch metrics if set True
        """
        super().__init__(order=CallbackOrder.logging, node=CallbackNode.master)
        self.metrics_to_log = metric_names
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end

        if not (self.log_on_batch_end or self.log_on_epoch_end):
            raise ValueError("You have to log something!")

        self.loggers = {}

    def _log_metrics(
        self, metrics: Dict[str, float], step: int, mode: str, suffix=""
    ):
        if self.metrics_to_log is None:
            metrics_to_log = sorted(metrics.keys())
        else:
            metrics_to_log = self.metrics_to_log

        for name in metrics_to_log:
            if name in metrics:
                self.loggers[mode].add_scalar(
                    f"{name}{suffix}", metrics[name], step
                )

    def on_stage_start(self, runner: "IRunner") -> None:
        """Stage start hook. Check ``logdir`` correctness.
        Args:
            runner: current runner
        """
        assert runner.logdir is not None

        extra_mode = "_base"
        log_dir = os.path.join(runner.logdir, f"{extra_mode}_log")
        self.loggers[extra_mode] = SummaryWriter(log_dir)

    def on_loader_start(self, runner: "IRunner"):
        """Prepare tensorboard writers for the current stage."""
        if runner.loader_key not in self.loggers:
            log_dir = os.path.join(runner.logdir, f"{runner.loader_key}_log")
            self.loggers[runner.loader_key] = SummaryWriter(log_dir)

    def on_batch_end(self, runner: "IRunner"):
        """Translate batch metrics to tensorboard."""
        if runner.logdir is None:
            return

        if self.log_on_batch_end:
            mode = runner.loader_key
            metrics = runner.batch_metrics
            self._log_metrics(
                metrics=metrics,
                step=runner.global_sample_step,
                mode=mode,
                suffix="/batch",
            )

    def on_epoch_end(self, runner: "IRunner"):
        """Translate epoch metrics to tensorboard."""
        if runner.logdir is None:
            return

        if self.log_on_epoch_end:
            per_mode_metrics = split_dict_to_subdicts(
                dct=runner.epoch_metrics,
                prefixes=list(runner.loaders.keys()),
                extra_key="_base",
            )

            for mode, metrics in per_mode_metrics.items():
                # suffix = "" if mode == "_base" else "/epoch"
                self._log_metrics(
                    metrics=metrics,
                    step=runner.global_epoch,
                    mode=mode,
                    suffix="/epoch",
                )

        for logger in self.loggers.values():
            logger.flush()

    def on_stage_end(self, runner: "IRunner"):
        """Close opened tensorboard writers."""
        if runner.logdir is None:
            return

        for logger in self.loggers.values():
            logger.close()
