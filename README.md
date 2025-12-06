# 🦎 Chameleon: LLM Robustness Benchmark Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

**Evaluate LLM robustness under lexical distortions using semantic paraphrasing.**

Chameleon tests how well language models handle semantically equivalent but lexically varied questions. It applies controlled distortions (μ=0.0 to μ=0.9) while preserving meaning and correct answers, then measures performance degradation.

## ✨ Key Features

- 🔬 **Semantic Distortion Engine**: Uses Mistral to generate meaning-preserving paraphrases at 10 intensity levels
- 📊 **Statistical Analysis**: McNemar's tests, confidence intervals, significance testing
- 📈 **Rich Visualizations**: Heatmaps, accuracy plots, degradation analysis
- 🚀 **Batch API Support**: OpenAI & Mistral batch APIs for efficient large-scale evaluation
- 🤖 **Multi-Model**: Test OpenAI GPT, Anthropic Claude, or local models
- 📝 **Executive Reports**: Auto-generated markdown reports with charts and insights

## 📦 Installation

### Option 1: pip install (Recommended)

```bash
# Clone the repository
git clone https://github.com/stevesolun/Chameleon.git
cd Chameleon

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install as editable package (optional)
pip install -e .
```

### Option 2: Docker

```bash
# Build the Docker image
docker build -t chameleon .

# Run interactive CLI
docker run -it --rm \
  -v $(pwd)/Projects:/app/Projects \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e MISTRAL_API_KEY=$MISTRAL_API_KEY \
  chameleon python cli.py --help

# Run analysis on a project
docker run -it --rm \
  -v $(pwd)/Projects:/app/Projects \
  chameleon python cli.py analyze --project MyProject
```

### API Keys Setup

Create a `.env` file in your project directory or export environment variables:

```bash
# Required for distortion generation
export MISTRAL_API_KEY="your-mistral-key"

# Required for target model evaluation (if using OpenAI)
export OPENAI_API_KEY="your-openai-key"

# Optional
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## 🚀 Quick Start

### 1. Create a Project

```bash
python cli.py init
```

Follow the interactive prompts to configure:
- Project name
- Target model (e.g., GPT-5.1, Claude)
- Distortion settings (μ values, distortions per question)
- API keys

### 2. Upload Your Data

When prompted, provide CSV files with questions. Required columns:
- `question_text` - The question
- `options_json` - Answer options as JSON (e.g., `{"A": "...", "B": "..."}`)
- `answer` - Correct answer(s) (e.g., "A" or "A, D")

Optional: `subject`, `question_id`

### 3. Generate Distortions

```bash
python cli.py distort --project MyProject
```

This uses Mistral to create semantic paraphrases at each μ level.

### 4. Evaluate Target Model

```bash
python cli.py evaluate --project MyProject
```

Sends distorted questions to your target model (e.g., GPT-5.1) via batch API.

### 5. Run Analysis

```bash
python cli.py analyze --project MyProject
```

Generates:
- Statistical analysis (McNemar's tests)
- Visualizations (heatmaps, plots)
- Executive report (markdown)

## 📁 Project Structure

```
Chameleon/
├── chameleon/                 # Main package
│   ├── core/                  # Config, project management, schemas
│   ├── models/                # Model backends (OpenAI, Anthropic, etc.)
│   ├── distortion/            # Distortion engine and validation
│   ├── evaluation/            # Batch evaluation processor
│   ├── analysis/              # Statistics, visualizations, reports
│   └── cli/                   # Command-line interface
├── Projects/                  # Your evaluation projects
│   └── MyProject/
│       ├── original_data/     # Input CSV files
│       ├── distorted_data/    # Generated distortions
│       ├── results/           # Evaluation results & analysis
│       ├── config.yaml        # Project settings
│       └── .env               # API keys (gitignored)
├── cli.py                     # CLI entry point
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker support
└── README.md
```

## 📋 CLI Commands

```bash
# Project Management
python cli.py init                    # Create new project (interactive)
python cli.py list                    # List all projects
python cli.py status -p PROJECT       # Show project status

# Distortion & Evaluation
python cli.py distort -p PROJECT      # Generate distortions
python cli.py evaluate -p PROJECT     # Evaluate target model

# Analysis
python cli.py analyze -p PROJECT      # Run full analysis

# Help
python cli.py help                    # Show all commands
python cli.py COMMAND --help          # Command-specific help
```

## 📊 Understanding μ (Miu) Levels

| μ Level | Distortion Type | Example |
|---------|-----------------|---------|
| 0.0 | None (baseline) | Original question unchanged |
| 0.1-0.2 | Minimal | 1-3 word synonyms |
| 0.3-0.4 | Moderate | Phrase restructuring |
| 0.5-0.6 | Mixed | Lexical + structural changes |
| 0.7-0.8 | Heavy | Major paraphrasing |
| 0.9 | Full | Complete reconstruction |

## 📈 Example Results

From Medical Certification Exam benchmark (58,786 questions):

| μ Level | Accuracy | Degradation from Baseline |
|---------|----------|---------------------------|
| 0.0 | 63.3% | — (baseline) |
| 0.1 | 62.1% | -1.2% |
| 0.5 | 61.1% | -2.2% |
| 0.9 | 60.6% | -2.7% |

**Key Finding**: GPT-5.1 shows ~2.7% degradation from baseline to maximum distortion, indicating moderate robustness to lexical variations.

## 🔬 Statistical Methods

### McNemar's Test

Used for paired binary outcomes (correct/incorrect) to determine if accuracy differences are statistically significant:

```python
from chameleon.analysis import analyze_distortion_significance

results = analyze_distortion_significance(
    df,
    baseline_col="miu",
    baseline_value=0.0,
    is_correct_col="is_correct"
)
```

### Confidence Intervals

Wilson score intervals for accuracy proportions with 95% confidence.

## 🐳 Docker Usage

### Build

```bash
docker build -t chameleon .
```

### Run Commands

```bash
# Interactive shell
docker run -it --rm \
  -v $(pwd)/Projects:/app/Projects \
  -e MISTRAL_API_KEY=$MISTRAL_API_KEY \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  chameleon bash

# Run specific command
docker run --rm \
  -v $(pwd)/Projects:/app/Projects \
  chameleon python cli.py list
```

## 🛠️ Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black chameleon/
ruff check chameleon/
```

## 📄 Citation

If you use Chameleon in your research, please cite:

```bibtex
@software{chameleon2024,
  title={Chameleon: LLM Robustness Testing Framework},
  author={Steve Solun},
  year={2024},
  url={https://github.com/stevesolun/Chameleon}
}
```

**Foundational Work:**

```bibtex
@article{cohen2025forget,
  title={Forget What You Know about LLMs Evaluations - LLMs are Like a Chameleon},
  author={Cohen-Inger, Nurit and Elisha, Yehonatan and Shapira, Bracha and Rokach, Lior and Cohen, Seffi},
  journal={arXiv preprint arXiv:2502.07445},
  year={2025},
  url={https://arxiv.org/abs/2502.07445}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Mistral AI](https://mistral.ai/) for distortion generation
- [OpenAI](https://openai.com/) for GPT evaluation
- The authors of the original Chameleon research paper
- The open-source ML community

---

**Built with ❤️ for the AI research community**
