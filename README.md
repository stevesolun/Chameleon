# 🦎 Chameleon: LLM Robustness Benchmark Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

**Evaluate LLM robustness under lexical distortions using semantic paraphrasing.**

Chameleon tests how well language models handle semantically equivalent but lexically varied questions. It applies controlled distortions (μ=0.0 to μ=0.9) while preserving meaning and correct answers, then measures performance degradation.

## ⚡ Requirements

| Component | Provider | Purpose |
|-----------|----------|---------|
| **Distortion Engine** | [Mistral AI](https://console.mistral.ai/) | Generates semantic paraphrases |
| **Target Model** | [OpenAI](https://platform.openai.com/) | Model being evaluated (GPT-4o, GPT-5.1, etc.) |

> **Note:** You need API keys from both providers. Get your Mistral key at [console.mistral.ai](https://console.mistral.ai/) and OpenAI key at [platform.openai.com](https://platform.openai.com/api-keys).

## ✨ Key Features

- 🔬 **Semantic Distortion Engine**: Uses Mistral to generate meaning-preserving paraphrases at 10 intensity levels
- 📊 **Statistical Analysis**: McNemar's tests, confidence intervals, significance testing
- 📈 **Rich Visualizations**: Heatmaps, accuracy plots, degradation analysis
- 🚀 **Batch API Support**: OpenAI & Mistral batch APIs for efficient large-scale evaluation
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
```

## 🔑 API Keys Setup

You need API keys from:
- **Mistral AI**: [console.mistral.ai](https://console.mistral.ai/) - for distortion generation
- **OpenAI**: [platform.openai.com](https://platform.openai.com/api-keys) - for target model evaluation

Set them as environment variables or the CLI will prompt you:

```bash
export MISTRAL_API_KEY="your-mistral-key"
export OPENAI_API_KEY="your-openai-key"
```

## 🚀 Quick Start

### 1. Create a Project

```bash
python cli.py init
```

Follow the interactive prompts to configure:
- Project name
- Target model (e.g., gpt-5.1, gpt-4o)
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

Uses Mistral to create semantic paraphrases at each μ level.

### 4. Evaluate Target Model

```bash
python cli.py evaluate --project MyProject
```

Sends distorted questions to your target model via OpenAI Batch API.

### 5. Run Analysis

```bash
python cli.py analyze --project MyProject
```

Generates statistical analysis, visualizations, and executive report.

## 📋 CLI Commands

```bash
# Project Management
python cli.py init                    # Create new project (interactive)
python cli.py list                    # List all projects
python cli.py status -p PROJECT       # Show project status
python cli.py edit -p PROJECT         # Edit project configuration
python cli.py delete -p PROJECT       # Delete project (double confirmation)

# Distortion & Evaluation
python cli.py distort -p PROJECT      # Generate distortions (requires Mistral)
python cli.py evaluate -p PROJECT     # Evaluate target model (requires OpenAI)

# Analysis
python cli.py analyze -p PROJECT      # Run full analysis

# Help
python cli.py help                    # Show all commands
```

## 📁 Project Structure

```
Chameleon/
├── chameleon/                 # Main package
│   ├── core/                  # Config, project management
│   ├── distortion/            # Mistral-based distortion engine
│   ├── evaluation/            # OpenAI batch evaluation
│   └── analysis/              # Statistics and visualizations
├── Projects/                  # Your evaluation projects
│   └── MyProject/
│       ├── original_data/     # Input CSV files
│       ├── distorted_data/    # Generated distortions
│       ├── results/           # Evaluation results & analysis
│       └── config.yaml        # Project settings
├── cli.py                     # CLI entry point
├── requirements.txt           # Dependencies
└── Dockerfile                 # Docker support
```

## 📊 Understanding μ (Miu) Distortion Levels

| μ Level | Distortion Type | Description |
|---------|-----------------|-------------|
| 0.0 | None (baseline) | Original question unchanged |
| 0.1-0.2 | Minimal | 1-3 word synonyms |
| 0.3-0.4 | Moderate | Phrase restructuring |
| 0.5-0.6 | Mixed | Lexical + structural changes |
| 0.7-0.8 | Heavy | Major paraphrasing |
| 0.9 | Full | Complete reconstruction |

## 📈 Output

After running analysis, you get:

- **Visualizations**: Accuracy plots, degradation heatmaps, statistical significance charts
- **Statistics**: McNemar's test results, confidence intervals, per-subject breakdown
- **Reports**: `Executive_Report.md` with full analysis and findings

All outputs are saved to `Projects/YourProject/results/`

## 🐳 Docker Usage

```bash
# Build
docker build -t chameleon .

# Run with mounted projects and API keys
docker run -it --rm \
  -v $(pwd)/Projects:/app/Projects \
  -e MISTRAL_API_KEY=$MISTRAL_API_KEY \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  chameleon python cli.py init
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

---

**Built with ❤️ for the AI research community**
