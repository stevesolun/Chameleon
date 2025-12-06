# 🦎 Chameleon: LLM Robustness Testing Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive framework for testing large language model robustness under various distortion conditions. Evaluate GPT, Claude, and local models across text, image, and other modalities with systematic perturbation analysis.

## ✨ Features

- **🔬 Multi-Model Support**: OpenAI (GPT-4o, GPT-5), Anthropic (Claude), MLX (Apple Silicon), CUDA (NVIDIA GPUs)
- **📊 Per-Project Organization**: Self-contained evaluation projects with standardized structure
- **🧮 Statistical Analysis**: McNemar's test, confidence intervals, significance testing
- **📈 Visualizations**: Degradation heatmaps, accuracy plots, key insights summaries
- **🚀 Batch Processing**: Async batch APIs for efficient large-scale evaluation
- **🎯 Modular Design**: Extensible backends, distortion engines, and analysis modules

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/stevesolun/Chameleon.git
cd Chameleon

# Install core dependencies
pip install -r requirements.txt

# Or install as a package (recommended)
pip install -e ".[all]"

# Set up API keys (for remote backends)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  # optional
```

### Create Your First Project

```bash
# Interactive project creation
python cli.py init

# Or with command-line arguments
python cli.py init --name my_gpt5_test --modality text --model gpt-4o --backend openai
```

### Project Workflow

```bash
# 1. Create project
python cli.py init --name mmlu_distortion_test

# 2. Add your data to projects/mmlu_distortion_test/original_data/
#    (CSV with columns: question_id, original_question, correct_answer, subject, etc.)

# 3. Run analysis (after you have results)
python cli.py analyze --project mmlu_distortion_test

# 4. View results
ls projects/mmlu_distortion_test/analysis/
```

## 📁 Project Structure

Each project follows a standardized structure:

```
projects/
└── my_project/
    ├── original_data/      # Raw/clean input data before distortion
    ├── distorted_data/     # Distorted versions (generated or imported)
    ├── results/            # Model outputs from batch runs
    ├── analysis/           # Analysis outputs (tables, plots, metrics)
    ├── project_config.yaml # Project configuration
    └── README.md           # Project-specific documentation
```

## 🏗️ Repository Architecture

```
Chameleon/
├── chameleon/              # Main package
│   ├── core/               # Config, project, and data schemas
│   ├── models/             # Model backends (OpenAI, Anthropic, MLX, CUDA)
│   ├── analysis/           # Metrics, McNemar, visualizations
│   └── cli/                # Command-line interface
├── projects/               # Evaluation projects (created at runtime)
├── config/                 # Global configuration files
├── distortions/            # Example distortion datasets
├── tests/                  # Unit tests
├── archive/                # Legacy scripts (preserved for reference)
├── cli.py                  # CLI entry point
├── pyproject.toml          # Package configuration
└── requirements.txt        # Dependencies
```

## 📊 Analysis Features

### McNemar's Statistical Test

Compares paired binary outcomes (correct/incorrect) across conditions:

```python
from chameleon.analysis.mcnemar import analyze_distortion_significance

# Compare each distortion level vs baseline
results = analyze_distortion_significance(
    df,
    baseline_col="miu",
    baseline_value=0.0,
    is_correct_col="is_correct"
)
```

### Visualizations

```python
from chameleon.analysis.visualizations import (
    create_degradation_heatmap,
    create_key_insights_summary,
)

# Generate degradation heatmap
create_degradation_heatmap(performance_df, output_path="analysis/heatmap.png")
```

## 🔧 Model Backends

### OpenAI (Batch API)

```python
from chameleon.models import get_backend
from chameleon.core.schemas import BackendType

backend = get_backend(BackendType.OPENAI, "gpt-4o")

# Single completion
response = backend.complete("What is 2+2?")

# Batch processing (async)
batch_id = backend.submit_batch(requests, description="MMLU Evaluation")
status = backend.get_batch_status(batch_id)
results = backend.get_batch_results(batch_id)
```

### Anthropic (Claude)

```python
backend = get_backend(BackendType.ANTHROPIC, "claude-3-5-sonnet-20241022")
response = backend.complete("What is the capital of France?")
```

### Local MLX (Apple Silicon)

```python
backend = get_backend(BackendType.MLX, "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
response = backend.complete("Explain quantum computing.")
```

### Local CUDA (NVIDIA)

```python
backend = get_backend(BackendType.CUDA_LOCAL, "mistralai/Mistral-7B-Instruct-v0.3")
responses = backend.complete_batch(requests, batch_size=4)
```

## 📋 CLI Commands

```bash
# Project Management
python cli.py init              # Create new project (interactive)
python cli.py list              # List all projects
python cli.py status -p NAME    # Show project status

# Analysis
python cli.py analyze -p NAME   # Run statistical analysis

# Configuration
python cli.py config --show     # Show global configuration
python cli.py help              # Show help
```

## 🔬 Research Background

This project is inspired by research on LLM robustness evaluation:

- **"Forget What You Know about LLMs Evaluations - LLMs are Like a Chameleon"** by Cohen-Inger et al. ([ArXiv:2502.07445v2](https://arxiv.org/html/2502.07445v2))
- **MMLU Benchmark** by Hendrycks et al. ([ArXiv:2009.03300](https://arxiv.org/abs/2009.03300))

### Key Findings from Original Chameleon Study

- **18,200 questions** tested with **85.8% overall accuracy**
- **15.4% performance drop** from baseline to maximum distortion (μ=0.9)
- **Statistical significance**: All distortion levels showed highly significant degradation (p < 0.001)
- **Domain-specific vulnerability**: Mathematical/logical subjects showed highest degradation (36-51%)

## 📈 Example Results

| μ Level | Accuracy | Degradation |
|---------|----------|-------------|
| 0.0     | 95.5%    | 0.0%        |
| 0.3     | 89.6%    | 5.9%        |
| 0.6     | 85.5%    | 10.0%       |
| 0.9     | 80.9%    | 14.6%       |

## 🛠️ Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black chameleon/
ruff check chameleon/

# Type checking
mypy chameleon/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT API access
- Anthropic for Claude API access
- MMLU benchmark creators for the dataset
- The open-source ML community

---

**Built with ❤️ for the AI research community**
