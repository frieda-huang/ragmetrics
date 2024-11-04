import time
import tracemalloc
from enum import Enum
from functools import wraps
from typing import List, Optional

import torch
from datasets import load_dataset
from performance_logger import PerformanceLogger
from torchmetrics.retrieval import (
    RetrievalMAP,
    RetrievalMRR,
    RetrievalPrecision,
    RetrievalRecall,
)


class MetricsType(Enum):
    recall = "recall"
    precision = "precision"
    mrr = "mrr"
    map = "map"

    def __str__(self) -> str:
        return self.value


metrics_function_mapping = {
    MetricsType.recall.value: RetrievalRecall,
    MetricsType.precision.value: RetrievalPrecision,
    MetricsType.mrr.value: RetrievalMRR,
    MetricsType.map.value: RetrievalMAP,
}


def measure_latency_for_gpu(dummy_input, model, nb_iters=10):
    """Measure latency for GPU-bound tasks

    Args:
        dummy_input: A sample input matching your production input shape
        model: Model to be evaluated with dummy input for warmup
        nb_iters: Number of iterations to average the execution time
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # GPU warmup
            for _ in range(10):
                _ = model(dummy_input)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(nb_iters):
                result = func(*args, **kwargs)
            end.record()

            torch.cuda.current_stream().synchronize()
            avg_latency = start.elapsed_time(end) / nb_iters * 1e3

            PerformanceLogger().log(
                "gpu_latency",
                func.__name__,
                f"GPU Average Latency (over {nb_iters} iterations): {avg_latency:.3f} ms",
            )

            return result

        return wrapper

    return decorator


def measure_latency_for_cpu(func):
    """Measure latency for CPU-bound tasks"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        duration_ms = (end - start) * 1e3

        PerformanceLogger().log(
            "cpu_latency", func.__name__, f"CPU Latency: {duration_ms:.3f} ms"
        )

        return result

    return wrapper


def measure_ram(func):
    """Measure CPU memory"""

    @wraps(func)
    def wrapper(*args, **kwargs):

        tracemalloc.start()

        result = func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()

        PerformanceLogger(
            "ram_usage",
            func.__name__,
            f"RAM Usage - Current: {current / (1024 ** 2):.2f} MB, Peak: {peak / (1024 ** 2):.2f} MB",
        )

        tracemalloc.stop()

        return result

    return wrapper


def measure_vram(func):
    """Measure GPU memory"""

    @wraps(func)
    def wrapper(*args, **kwargs):

        torch.cuda.reset_peak_memory_stats(device=None)

        result = func(*args, **kwargs)

        vram_used = torch.cuda.max_memory_allocated(device=None)

        print(f"{func.__name__} used {vram_used / (1024 ** 2):.2f} MB of VRAM")

        PerformanceLogger(
            "vram_usage",
            func.__name__,
            f"VRAM Usage ({func.__name__}): {vram_used / (1024 ** 2):.2f} MB",
        )

        return result

    return wrapper


def generate_target_tensor(top_k_indices: torch.Tensor) -> torch.Tensor:
    """Generate a target tensor with 1s where elements in top_k_indices
    match their row index, 0s otherwise

    Example:
        top_k_indices = tensor([[0, 3, 4],  # Query 0's top-k docs: [0, 3, 4]
                                [1, 2, 5]]) # Query 1's top-k docs: [1, 2, 5]

        result = generate_target_tensor(top_k_indices)
        # result: tensor([1, 0, 0, 1, 0, 0]) -> 1s for [0, 3] matching query IDs, 0 otherwise
    """
    row_indices = torch.arange(top_k_indices.size(0)).view(-1, 1)
    matches = (top_k_indices == row_indices).to(torch.int)
    return matches.flatten()


def generate_indexes_tensor(query_indices: List[int], top_k: int) -> torch.Tensor:
    """Generate row indices for each query

    Example:
        If top_k=5 and there are 2 queries,
        this will create a tensor like tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        where each number corresponds to the query ID for the top_k retrieved items
    """
    return torch.cat(
        (
            torch.full(
                top_k,
            ),
            i,
        )
        for i in range(len(query_indices))
    )


def calculate_metrics_colpali(
    metrics: List[MetricsType],
    top_k=10,
    dataset="vidore/syntheticDocQA_artificial_intelligence_test",
    dataset_size: Optional[int] = None,
):
    """Evaluate retrieval performance using specified metrics:

    - Precision: Measures the accuracy of retrieved relevant documents
    - Recall: Assesses the proportion of all relevant documents retrieved
    - MRR (Mean Reciprocal Rank): Reflects the rank position of the first relevant document
    - MAP (Mean Average Precision): Provides an overall measure of precision across recall levels

    Args:
        metrics (List[MetricsType]): List of metrics to calculate,
            e.g., [MetricsType.precision, MetricsType.recall]

        top_k (int, optional): Number of top results to consider. Defaults to TOP_K
        dataset (str, optional): The name of the Hugging Face dataset to be loaded
        dataset_size (int, optional): The number of queries to use for metric calculation.
        Loads the entire dataset if set to None
    """

    ds = load_dataset(
        dataset, split=f"test[:{dataset_size}]" if dataset_size else "test"
    )

    def decorator(func):
        @wraps
        def wrapper(*args, **kwargs):
            scores = func(*args, **kwargs)
            top_k_scores, top_k_indices = torch.topk(scores, k=top_k, dim=1)

            preds = top_k_scores.flatten()
            query_indices = list(range(len(ds)))
            target = generate_target_tensor(top_k_indices)
            indexes = generate_indexes_tensor(query_indices, top_k)

            for metric in metrics:
                metric_function = metrics_function_mapping.get(metric)
                if metric_function:
                    m = metric_function()
                    percentage = m(preds, target, indexes=indexes)
                    result = percentage.item() * 1e2

                    PerformanceLogger().log(
                        f"{metric}",
                        func.__name__,
                        f"{metric}@{top_k}: {result:.2f}%",
                    )

        return wrapper

    return decorator
