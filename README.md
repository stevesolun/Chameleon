# 🦎 Chameleon: GPT-5 Robustness Testing Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenAI API](https://img.shields.io/badge/OpenAI-GPT--5-green.svg)](https://openai.com/api/)

A comprehensive framework for testing large language model robustness under lexical distortion. The Chameleon project systematically evaluates GPT-5's performance across 20 academic subjects with varying levels of lexical perturbation (μ = 0.0 to 0.9).

## 🎯 Project Overview

The Chameleon framework addresses a critical question in AI evaluation: **How robust are large language models to lexical variations in input text?** By applying controlled lexical distortions to academic assessment questions, we can measure model performance degradation and identify vulnerable domains.

### Key Features

- 🔬 **Systematic Distortion Testing**: 10 distortion levels (μ = 0.0 to 0.9)
- 📚 **Multi-Domain Coverage**: 20 academic subjects from medicine to mathematics
- 🤖 **GPT-5 Integration**: Automated batch processing via OpenAI API
- 📊 **Comprehensive Analysis**: Statistical evaluation with visualizations
- 🔄 **Reproducible Pipeline**: End-to-end automation with quality assurance

## 📊 Key Results

- **18,200 questions processed** with 100% completion
- **85.8% overall accuracy** across all conditions
- **15.4% performance drop** from baseline to maximum distortion
- **97.4% format consistency** maintained under distortion

## 🚀 Quick Start

### Prerequisites

```bash
# Required Python packages
pip install -r requirements.txt

# OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

```bash
# 1. Prepare distorted questions
python3 chameleon.py --generate-distortions

# 2. Process with GPT-5
python3 gpt5_manager.py submit --auto

# 3. Monitor progress
python3 gpt5_manager.py monitor

# 4. Generate analysis
python3 -c "import analysis; analysis.generate_report()"
```

## 🏗️ Architecture

### Core Components

```
Chameleon/
├── 📁 config/
│   └── config.yaml              # Project configuration
├── 📁 modules/
│   ├── data_preparation.py      # Question preparation and distortion
│   ├── gpt5_batch_processor.py  # Batch processing logic
│   └── mistral_server.py        # Distortion generation server
├── 📁 batches/
│   ├── requests/                # OpenAI batch requests
│   ├── results/                 # GPT-5 responses
│   └── metadata/                # Processing metadata
├── 📁 distortions/
│   └── comprehensive_distortion_dataset_FINAL_*.csv  # Results
├── chameleon.py                 # Main orchestration script
├── gpt5_manager.py             # GPT-5 batch management
└── monitor_repair.py           # Monitoring and repair utilities
```

### Data Flow

```mermaid
graph LR
    A[Original Questions] --> B[Distortion Engine]
    B --> C[Batch Creator]
    C --> D[OpenAI API]
    D --> E[Response Processor]
    E --> F[Analysis Engine]
    F --> G[Reports & Visualizations]
```

## 🔧 System Functionality

### 1. Distortion Generation

The system applies controlled lexical distortions using a parameter μ (mu) that controls distortion intensity:

```python
# Example distortion levels
μ = 0.0  # No distortion (baseline)
μ = 0.3  # Light distortion
μ = 0.6  # Moderate distortion  
μ = 0.9  # Heavy distortion
```

**Original Question:**
```
Which operator in Python carries out exponentiation?
A) **   B) ^   C) exp   D) pow
```

**Distorted Question (μ = 0.5):**
```
Which operator in Python 3 carries out exponentiation on operands?
A) **   B) ^   C) exp   D) pow
```

### 2. Batch Processing Pipeline

The system efficiently processes large question sets using OpenAI's Batch API:

```python
# Automatic batch creation and submission
python3 gpt5_manager.py submit --auto

# Custom batch size
python3 gpt5_manager.py create --batch-size 100

# Monitor specific batches
python3 gpt5_manager.py monitor --ids batch_123 batch_124
```

### 3. Quality Assurance

- **Format Validation**: Ensures A/B/C/D response format
- **Duplicate Detection**: Removes redundant API calls
- **Error Recovery**: Automatic retry for failed requests
- **Completeness Check**: Validates 100% question coverage

### 4. Analysis Framework

Comprehensive statistical analysis with multiple dimensions:

```bash
# Generate full analysis report
python3 analysis.py --generate-report

# Subject-specific analysis
python3 analysis.py --subject "college_mathematics"

# Distortion-level analysis
python3 analysis.py --miu 0.5

# Export results
python3 analysis.py --export-csv --export-plots
```

## 📚 Usage Examples

### Example 1: Single Subject Analysis

```python
from modules.data_preparation import DistortionEngine
from modules.gpt5_batch_processor import BatchProcessor

# Initialize components
engine = DistortionEngine()
processor = BatchProcessor()

# Generate distortions for mathematics
questions = engine.load_subject("college_mathematics")
distorted = engine.apply_distortions(questions, miu_levels=[0.0, 0.3, 0.6, 0.9])

# Process with GPT-5
results = processor.process_batch(distorted)
print(f"Accuracy: {results.accuracy:.1f}%")
```

### Example 2: Custom Distortion Testing

```python
# Test specific distortion levels
custom_mius = [0.1, 0.25, 0.5, 0.75]
results = {}

for miu in custom_mius:
    distorted_questions = engine.apply_distortion(questions, miu=miu)
    result = processor.process_questions(distorted_questions)
    results[miu] = result.accuracy

# Plot results
import matplotlib.pyplot as plt
plt.plot(custom_mius, list(results.values()))
plt.xlabel('Distortion Level (μ)')
plt.ylabel('Accuracy (%)')
plt.title('Custom Distortion Analysis')
plt.show()
```

### Example 3: Batch Monitoring

```python
from gpt5_manager import GPT5Manager

manager = GPT5Manager()

# Submit all pending batches
batch_ids = manager.submit_all_batches()

# Monitor with progress tracking
for batch_id in batch_ids:
    status = manager.monitor_batch(batch_id)
    print(f"Batch {batch_id}: {status.progress}% complete")
    
# Download results when complete
manager.download_all_results()
```

## 📊 Configuration

### Main Configuration (`config/config.yaml`)

```yaml
settings:
  batch_split_percentage: 0.017
  max_batch_size: 80
  max_concurrent_workers: 2
  request_timeout: 600

subjects:
  - name: college_mathematics
    question_count: 910
    mius: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  - name: professional_medicine
    question_count: 910
    mius: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

### API Configuration

```bash
# Required environment variables
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_ORG_ID="your-organization-id"  # Optional

# Optional: Custom endpoints
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

## 📈 Results and Analysis

### Performance Summary

| Metric | Value |
|--------|--------|
| **Total Questions** | 18,200 |
| **Overall Accuracy** | 85.8% |
| **Completion Rate** | 100% |
| **Valid Format Rate** | 97.4% |
| **Processing Time** | ~24 hours |

### Top Performing Subjects

1. **Professional Medicine**: 98.1% accuracy
2. **Medical Genetics**: 97.7% accuracy  
3. **High School Psychology**: 97.6% accuracy
4. **Astronomy**: 96.8% accuracy
5. **College Physics**: 96.6% accuracy

### Distortion Impact

| μ Level | Accuracy | Degradation |
|---------|----------|-------------|
| 0.0 | 95.0% | 0.0% |
| 0.3 | 88.3% | 6.7% |
| 0.6 | 84.0% | 11.0% |
| 0.9 | 79.6% | 15.4% |

## 🔬 Research Applications

### Academic Research

- **Model Robustness Studies**: Systematic evaluation of LLM resilience
- **Educational Assessment**: Validate AI tutoring systems
- **Benchmark Development**: Create robustness benchmarks for LLMs
- **Domain Analysis**: Study subject-specific model vulnerabilities

### Industry Applications

- **Quality Assurance**: Test production LLM deployments
- **A/B Testing**: Compare model versions under stress
- **Risk Assessment**: Evaluate model reliability for critical applications
- **Performance Monitoring**: Continuous robustness evaluation

## 🛠️ Development

### Adding New Subjects

```python
# 1. Add to config.yaml
subjects:
  - name: new_subject_name
    question_count: 910
    mius: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 2. Prepare question data
python3 chameleon.py --add-subject new_subject_name

# 3. Generate distortions
python3 chameleon.py --subject new_subject_name --generate-distortions
```

### Custom Distortion Functions

```python
from modules.data_preparation import DistortionEngine

class CustomDistortionEngine(DistortionEngine):
    def apply_custom_distortion(self, text, intensity):
        # Implement custom distortion logic
        return modified_text

# Use custom engine
engine = CustomDistortionEngine()
results = engine.process_questions(questions, distortion_func=custom_func)
```

### Extending Analysis

```python
# Add custom metrics
from analysis import AnalysisFramework

framework = AnalysisFramework()
framework.add_metric("custom_score", custom_scoring_function)
framework.add_visualization("custom_plot", custom_plot_function)

# Generate extended report
framework.generate_report(include_custom=True)
```

## 📊 Data Formats

### Input Question Format

```json
{
  "question_id": "math_001",
  "subject": "college_mathematics", 
  "topic": "calculus",
  "original_question": "What is the derivative of x²?",
  "answer_options": {
    "A": "2x",
    "B": "x", 
    "C": "2x²",
    "D": "x²"
  },
  "correct_answer": "A",
  "difficulty": "medium"
}
```

### Output Results Format

```json
{
  "question_id": "math_001",
  "miu": 0.3,
  "distortion_index": 5,
  "distorted_question": "What represents the derivative of x²?",
  "gpt5_answer": "A",
  "is_correct": true,
  "response_time": 1.2,
  "confidence": 0.95
}
```

## 🔍 Troubleshooting

### Common Issues

**Batch Processing Failures:**
```bash
# Check batch status
python3 gpt5_manager.py monitor --failed-only

# Repair failed batches
python3 monitor_repair.py --auto-repair

# Resubmit specific batches
python3 gpt5_manager.py submit --ids batch_123 batch_124
```

**API Rate Limits:**
```bash
# Adjust batch size
python3 gpt5_manager.py config --batch-size 50

# Add delays between requests
python3 gpt5_manager.py config --delay 0.5
```

**Memory Issues:**
```bash
# Process in smaller chunks
python3 chameleon.py --chunk-size 1000

# Enable memory optimization
python3 chameleon.py --optimize-memory
```

### Debugging

```bash
# Enable verbose logging
export CHAMELEON_DEBUG=1
python3 chameleon.py --verbose

# Check system resources
python3 -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# Validate data integrity
python3 chameleon.py --validate-data
```

## 📋 Requirements

### System Requirements

- **Python**: 3.9 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 5GB free space for datasets and results
- **Network**: Stable internet connection for API calls

### Python Dependencies

```txt
pandas>=1.5.0
numpy>=1.20.0
matplotlib>=3.5.0
seaborn>=0.11.0
openai>=1.0.0
pyyaml>=6.0
requests>=2.28.0
tqdm>=4.64.0
```

### API Requirements

- OpenAI API key with GPT-5 access
- Sufficient API credits for batch processing
- Organization permissions for batch API usage

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/chameleon.git
cd chameleon

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python3 -m pytest tests/

# Run linting
flake8 src/ tests/
black src/ tests/
```

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Documentation**: [Wiki](https://github.com/your-username/chameleon/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-username/chameleon/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/chameleon/discussions)
- **Email**: chameleon-support@example.com

## 🙏 Acknowledgments

- OpenAI for GPT-5 API access and batch processing capabilities
- Academic institutions providing question datasets
- Open source community for foundational tools and libraries
- Research collaborators and beta testers

## 📚 Citations

If you use Chameleon in your research, please cite:

```bibtex
@software{chameleon2024,
  title={Chameleon: GPT-5 Robustness Testing Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/chameleon}
}
```

## 🗺️ Future Ideas & Improvements

### 🚀 LLM Expansion
- [ ] **Multi-Provider Support**: Anthropic Claude, Google Gemini, Cohere, Llama
- [ ] **Model Comparison**: Side-by-side robustness analysis across different LLMs
- [ ] **API Standardization**: Unified interface for testing any language model
- [ ] **Local Model Support**: Integration with Ollama, Hugging Face Transformers

### 🎯 Advanced Distortion Techniques
- [ ] **Semantic Distortions**: Meaning-preserving paraphrasing attacks
- [ ] **Syntactic Variations**: Grammar and sentence structure modifications
- [ ] **Adversarial Prompting**: Injection and jailbreaking resistance testing
- [ ] **Cultural/Linguistic Bias**: Cross-language and cultural sensitivity analysis

### 🖼️ Multi-Modal Capabilities
- [ ] **Vision-Language Models**: Test robustness of GPT, Claude, Gemini, etc.
- [ ] **Image Distortions**: Visual noise, compression, style transfer effects
- [ ] **Mathematical Notation**: LaTeX equation parsing and distortion
- [ ] **Audio Processing**: Speech-to-text robustness analysis

### 🏗️ Infrastructure & Scaling
- [ ] **Real-Time Processing**: Live distortion testing and monitoring
- [ ] **Distributed Computing**: Multi-node processing for large-scale studies
- [ ] **Cloud Integration**: AWS, GCP, Azure deployment templates
- [ ] **Auto-Scaling**: Dynamic resource allocation based on workload

### 📊 Enhanced Analytics
- [ ] **Interactive Dashboards**: Web-based visualization and exploration
- [ ] **Statistical Significance**: Bayesian analysis and confidence intervals
- [ ] **Causal Analysis**: Understanding why certain distortions affect performance
- [ ] **Benchmark Suite**: Standardized robustness testing protocols

### 🔬 Research Applications
- [ ] **Educational Assessment**: Integration with learning management systems
- [ ] **Medical AI Safety**: Healthcare-specific robustness testing
- [ ] **Legal AI Compliance**: Regulatory and bias testing frameworks
- [ ] **Experiment Tracking**: MLflow, Weights & Biases integration

---

**Built with ❤️ for the AI research community**