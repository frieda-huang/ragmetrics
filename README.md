# Performance and Metrics Monitoring for RAG System with ColPali

## About

This project provides tools to measure and log performance metrics for a RAG system that uses the ColPali model for document retrieval. The included decorators measure latency, CPU/RAM usage, VRAM usage, and retrieval metrics such as recall, precision, mean reciprocal rank (MRR), and mean average precision (MAP). All metrics are logged in a structured format for performance analysis and optimization.

## Features

-   **Latency Measurement**: Measure GPU and CPU latency.
-   **Memory Usage**: Track RAM and VRAM usage during execution.
-   **Evaluation Metrics**: Calculate Recall, Precision, Mean Reciprocal Rank (MRR), and Mean Average Precision (MAP).
-   **Customizable Decorators**: Use decorators to measure specific functions or model-related tasks.
-   **Automated Logging**: Logs results to a file after program execution for analysis.

## Requirements

-   Python 3.8+
-   Poetry for dependency management
-   CUDA-compatible device (for GPU-based measurement)

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/frieda-huang/ragmetrics.git
    cd ragmetrics
    ```

2. Install dependencies using Poetry:

    `poetry install`

## Usage

Example: Running ColPali Retrieval with Metrics Logging

To use the decorators for latency and memory measurement, wrap your functions in the provided decorators and specify the metrics to track. The results are automatically saved to a log file upon program completion.

```
from metrics import (
    measure_latency_for_cpu,
    measure_latency_for_gpu,
    measure_ram,
    measure_vram,
    calculate_metrics_colpali,
    MetricsType,
    )
```

## Define metrics and parameters

```
metrics = [MetricsType.recall, MetricsType.precision, MetricsType.mrr, MetricsType.map]
top_k = 10
dataset_size = 100 # Or set to None to use the full dataset

@measure_latency_for_cpu
@measure_ram
@calculate_metrics_colpali(metrics=metrics, top_k=top_k, dataset_size=dataset_size)
def run_retrieval_with_metrics(): # Your retrieval and scoring code goes here
    pass
```

### Supported Decorators

-   **@measure_latency_for_cpu**: Measures and logs CPU latency
-   **@measure_latency_for_gpu**: Measures and logs GPU latency
-   **@measure_ram**: Measures and logs RAM usage
-   **@measure_vram**: Measures and logs VRAM usage
-   **@calculate_metrics_colpali**: Calculate Recall, Precision, MRR, and MAP

### Logging Metrics

Metrics are automatically saved to a JSON log file in the root of the project. To set a custom logging location, you can modify the PerformanceLogger class.

### Example Log Output

```
[
   {
      "metric":"gpu_latency",
      "function":"run_retrieval_with_metrics",
      "value":"GPU Average Latency (10 iterations): 20.5 ms"
   },
   {
      "metric":"recall",
      "function":"run_retrieval_with_metrics",
      "value":"Recall@10: 85.7%"
   },
   {
      "metric":"ram_usage",
      "function":"run_retrieval_with_metrics",
      "value":"RAM Usage - Current: 250 MB, Peak: 400 MB"
   }
]
```
