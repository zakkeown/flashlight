"""
Optional TensorBoard integration for MLX.

Wraps the tensorboard library for compatibility with MLX tensors.
Requires: pip install tensorboard

Example:
    >>> from flashlight.utils.tensorboard import SummaryWriter
    >>> import flashlight
    >>>
    >>> writer = SummaryWriter('runs/experiment_1')
    >>> for step in range(100):
    ...     loss = flashlight.tensor(0.1 * step)
    ...     writer.add_scalar('loss', loss, step)
    >>> writer.close()
"""

import warnings
from typing import Any, Dict, Optional, Union

__all__ = ["SummaryWriter"]

# Check for TensorBoard availability
_TENSORBOARD_AVAILABLE = False
_SummaryWriterBase = None

try:
    from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter

    _SummaryWriterBase = _TorchSummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        import os as _os
        import time as _time

        from tensorboard.compat.proto import summary_pb2
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.summary.writer.writer import FileWriter as _FileWriter

        _TENSORBOARD_AVAILABLE = True
    except ImportError:
        pass


def _to_numpy(tensor: Any) -> Any:
    """
    Convert tensor to numpy-compatible format if needed.

    Args:
        tensor: A tensor or scalar value.

    Returns:
        Numpy array or scalar value.
    """
    # MLX Compat Tensor
    if hasattr(tensor, "_mlx_array"):
        import mlx.core as mx

        mx.eval(tensor._mlx_array)
        arr = tensor._mlx_array
        # Convert to Python scalar if 0-d
        if arr.ndim == 0:
            return float(arr.item())
        return arr.tolist()

    # MLX array directly
    if hasattr(tensor, "tolist"):
        try:
            import mlx.core as mx

            if isinstance(tensor, mx.array):
                mx.eval(tensor)
                if tensor.ndim == 0:
                    return float(tensor.item())
                return tensor.tolist()
        except (ImportError, AttributeError):
            pass

    # NumPy array
    if hasattr(tensor, "numpy"):
        return tensor.numpy()

    # Already a scalar or compatible type
    return tensor


class SummaryWriter:
    """
    TensorBoard SummaryWriter for MLX.

    Wraps TensorBoard's SummaryWriter, automatically converting MLX
    tensors to numpy for logging. Provides a PyTorch-compatible API.

    Args:
        log_dir: Directory to save TensorBoard logs. If None, uses
                 'runs/CURRENT_DATETIME_HOSTNAME'.
        comment: Comment to append to the log directory name.
        purge_step: Step at which to purge events (delete later events).
        max_queue: Size of the queue for pending events (default 10).
        flush_secs: Interval in seconds to flush events (default 120).
        filename_suffix: Suffix to append to event file names.

    Example:
        >>> writer = SummaryWriter('runs/exp1')
        >>> for i in range(100):
        ...     writer.add_scalar('train/loss', 0.1 / (i + 1), i)
        >>> writer.close()
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        comment: str = "",
        purge_step: Optional[int] = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = "",
    ):
        if not _TENSORBOARD_AVAILABLE:
            raise ImportError(
                "TensorBoard not available. Install with: pip install tensorboard\n"
                "Or install with torch: pip install torch"
            )

        self._closed = False

        # Try to use torch's SummaryWriter if available (more feature-complete)
        try:
            from torch.utils.tensorboard import SummaryWriter as _SW

            self._writer = _SW(
                log_dir=log_dir,
                comment=comment,
                purge_step=purge_step,
                max_queue=max_queue,
                flush_secs=flush_secs,
                filename_suffix=filename_suffix,
            )
            self._mode = "torch"
        except ImportError:
            # Fallback to raw tensorboard
            from tensorboard.summary.writer.writer import FileWriter

            if log_dir is None:
                import socket
                from datetime import datetime

                current_time = datetime.now().strftime("%b%d_%H-%M-%S")
                log_dir = f"runs/{current_time}_{socket.gethostname()}{comment}"

            import os

            os.makedirs(log_dir, exist_ok=True)
            self._writer = FileWriter(log_dir, max_queue=max_queue, flush_secs=flush_secs)
            self._mode = "raw"
            self.log_dir = log_dir

    def add_scalar(
        self,
        tag: str,
        scalar_value: Union[float, Any],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        new_style: bool = False,
        double_precision: bool = False,
    ) -> None:
        """
        Add a scalar value to the log.

        Args:
            tag: Name for the scalar (e.g., 'loss', 'accuracy').
            scalar_value: The scalar value to log (can be MLX tensor).
            global_step: Training step number.
            walltime: Wall clock time for the event.
            new_style: Use new scalar format (torch only).
            double_precision: Use double precision (torch only).

        Example:
            >>> writer.add_scalar('loss', loss_tensor, step)
        """
        value = _to_numpy(scalar_value)
        if hasattr(value, "item"):
            value = value.item()

        if self._mode == "torch":
            self._writer.add_scalar(tag, value, global_step, walltime)
        else:
            # Raw tensorboard mode
            from tensorboard.compat.proto import summary_pb2

            summary = summary_pb2.Summary()
            summary.value.add(tag=tag, simple_value=float(value))
            self._writer.add_summary(summary, global_step)

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, Union[float, Any]],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> None:
        """
        Add multiple scalars to the log under a main tag.

        Args:
            main_tag: Parent tag for the scalars.
            tag_scalar_dict: Dict mapping sub-tags to scalar values.
            global_step: Training step number.
            walltime: Wall clock time for the event.

        Example:
            >>> writer.add_scalars('metrics', {'loss': 0.1, 'accuracy': 0.9}, step)
        """
        converted = {k: _to_numpy(v) for k, v in tag_scalar_dict.items()}
        for k, v in converted.items():
            if hasattr(v, "item"):
                converted[k] = v.item()

        if self._mode == "torch":
            self._writer.add_scalars(main_tag, converted, global_step, walltime)
        else:
            # Raw mode: add each scalar individually with combined tag
            for sub_tag, value in converted.items():
                self.add_scalar(f"{main_tag}/{sub_tag}", value, global_step, walltime)

    def add_histogram(
        self,
        tag: str,
        values: Any,
        global_step: Optional[int] = None,
        bins: str = "tensorflow",
        walltime: Optional[float] = None,
        max_bins: Optional[int] = None,
    ) -> None:
        """
        Add a histogram of values to the log.

        Args:
            tag: Name for the histogram.
            values: Values to create histogram from (can be MLX tensor).
            global_step: Training step number.
            bins: Binning strategy ('tensorflow', 'auto', 'fd', etc.).
            walltime: Wall clock time for the event.
            max_bins: Maximum number of bins.

        Example:
            >>> writer.add_histogram('weights', model.layer.weight, step)
        """
        values = _to_numpy(values)

        if self._mode == "torch":
            self._writer.add_histogram(tag, values, global_step, bins, walltime, max_bins)
        else:
            warnings.warn(
                "Histogram logging requires torch tensorboard integration. "
                "Install torch for full histogram support.",
                UserWarning,
            )

    def add_image(
        self,
        tag: str,
        img_tensor: Any,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        dataformats: str = "CHW",
    ) -> None:
        """
        Add an image to the log.

        Args:
            tag: Name for the image.
            img_tensor: Image tensor (can be MLX tensor).
            global_step: Training step number.
            walltime: Wall clock time for the event.
            dataformats: Format of img_tensor ('CHW', 'HWC', 'HW', etc.).

        Example:
            >>> writer.add_image('sample', image_tensor, step)
        """
        img = _to_numpy(img_tensor)

        if self._mode == "torch":
            self._writer.add_image(tag, img, global_step, walltime, dataformats)
        else:
            warnings.warn(
                "Image logging requires torch tensorboard integration. "
                "Install torch for full image support.",
                UserWarning,
            )

    def add_images(
        self,
        tag: str,
        img_tensor: Any,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        dataformats: str = "NCHW",
    ) -> None:
        """
        Add multiple images to the log.

        Args:
            tag: Name for the images.
            img_tensor: Batch of image tensors.
            global_step: Training step number.
            walltime: Wall clock time for the event.
            dataformats: Format of img_tensor ('NCHW', 'NHWC', etc.).
        """
        img = _to_numpy(img_tensor)

        if self._mode == "torch":
            self._writer.add_images(tag, img, global_step, walltime, dataformats)
        else:
            warnings.warn(
                "Image logging requires torch tensorboard integration.",
                UserWarning,
            )

    def add_graph(
        self,
        model: Any,
        input_to_model: Any = None,
        verbose: bool = False,
        use_strict_trace: bool = True,
    ) -> None:
        """
        Add model graph to the log.

        Note: Graph logging has limited support for MLX models.

        Args:
            model: The model to graph.
            input_to_model: Sample input for tracing.
            verbose: Print graph structure.
            use_strict_trace: Use strict tracing mode.
        """
        warnings.warn(
            "add_graph has limited support for MLX models. "
            "Consider using add_text to document architecture.",
            UserWarning,
        )

    def add_text(
        self,
        tag: str,
        text_string: str,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> None:
        """
        Add text to the log.

        Args:
            tag: Name for the text entry.
            text_string: The text content.
            global_step: Training step number.
            walltime: Wall clock time for the event.

        Example:
            >>> writer.add_text('config', str(config_dict), 0)
        """
        if self._mode == "torch":
            self._writer.add_text(tag, text_string, global_step, walltime)
        else:
            # Raw mode doesn't easily support text, log as scalar with warning
            warnings.warn(
                "Text logging requires torch tensorboard integration.",
                UserWarning,
            )

    def add_embedding(
        self,
        mat: Any,
        metadata: Optional[Any] = None,
        label_img: Optional[Any] = None,
        global_step: Optional[int] = None,
        tag: str = "default",
        metadata_header: Optional[Any] = None,
    ) -> None:
        """
        Add embedding data to the log for visualization.

        Args:
            mat: Embedding matrix (N x D).
            metadata: Labels for each embedding point.
            label_img: Images for each embedding point.
            global_step: Training step number.
            tag: Name for the embedding.
            metadata_header: Headers for metadata columns.
        """
        mat = _to_numpy(mat)

        if self._mode == "torch":
            self._writer.add_embedding(mat, metadata, label_img, global_step, tag, metadata_header)
        else:
            warnings.warn(
                "Embedding logging requires torch tensorboard integration.",
                UserWarning,
            )

    def add_hparams(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Dict[str, Any],
        hparam_domain_discrete: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        global_step: Optional[int] = None,
    ) -> None:
        """
        Add hyperparameter configuration and metrics.

        Args:
            hparam_dict: Dict of hyperparameter names to values.
            metric_dict: Dict of metric names to values.
            hparam_domain_discrete: Domain of discrete hyperparameters.
            run_name: Name for this hyperparameter run.
            global_step: Training step number.

        Example:
            >>> writer.add_hparams(
            ...     {'lr': 0.01, 'batch_size': 32},
            ...     {'accuracy': 0.95, 'loss': 0.1}
            ... )
        """
        # Convert metrics
        metric_dict = {k: _to_numpy(v) for k, v in metric_dict.items()}
        for k, v in metric_dict.items():
            if hasattr(v, "item"):
                metric_dict[k] = v.item()

        if self._mode == "torch":
            self._writer.add_hparams(
                hparam_dict, metric_dict, hparam_domain_discrete, run_name, global_step
            )
        else:
            warnings.warn(
                "Hyperparameter logging requires torch tensorboard integration.",
                UserWarning,
            )

    def flush(self) -> None:
        """Flush pending events to disk."""
        if not self._closed:
            self._writer.flush()

    def close(self) -> None:
        """Close the writer and release resources."""
        if not self._closed:
            self._writer.close()
            self._closed = True

    def __enter__(self) -> "SummaryWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
