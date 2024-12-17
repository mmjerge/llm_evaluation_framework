# LLM Reliability Framework

A framework for evaluating and comparing different LLM-based methods across multiple benchmarks. This framework enables researchers to test their methods against various language models and provides standardized evaluation metrics.

## Project Structure
```
llm_reliability_framework/
├── assets/                  # Project assets and resources
├── benchmarks/             # Benchmark datasets and evaluation logic
│   ├── data/              # Benchmark data files
│   ├── evaluators/        # Evaluation implementations
│   └── data_loaders/      # Data loading utilities
├── config/                 # Configuration files
│   ├── methods/           # Method-specific configs
│   ├── benchmarks/        # Benchmark-specific configs
│   └── default.yaml       # Default framework config
├── evaluate/              # Directory for testing your method
├── evaluated_methods/     # Reference implementations for paper
├── notebooks/             # Jupyter notebooks for analysis
├── results/               # Test results and evaluations
├── tests/                 # Framework unit tests
├── utils/                 # Utility functions and model interfaces
├── environment.yaml       # Conda environment specification
├── main.py               # Main testing framework
├── run_coverage.sh       # Test coverage script
├── run_pylint.sh         # Code quality checking
├── run.slurm             # Slurm job submission script
└── setup.py              # Package setup file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm_reliability_framework
cd llm_reliability_framework
```

2. Create and activate the conda environment:
```bash
# If using conda
conda env create -f environment.yaml
conda activate llm_reliability

# If using mamba (faster alternative)
mamba env create -f environment.yaml
mamba activate llm_reliability
```

3. Set up API keys:
```bash
# Add to your .bashrc or .zshrc for persistence
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Testing Your Method

### 1. Add Your Method
Place your method's code in the `evaluate` directory:

```
evaluate/
└── your-method/              # Your method's directory
    ├── method/              # Your implementation
    │   ├── __init__.py
    │   └── main.py         # Contains your solve function
    └── requirements.txt     # Any additional dependencies
```

Your entry point function should accept a model and input text:

```python
def solve_function(model, input_text: str) -> str:
    """
    Args:
        model: LLM model instance provided by the framework
        input_text: Problem text from the benchmark
        
    Returns:
        str: Solution in the format expected by the benchmark
    """
    pass
```

### 2. Create Method Configuration
Create a configuration file in `config/methods/your-method.yaml`:

```yaml
entry_module: method.main
entry_function: solve_function
description: "Brief description of your method"
```

### 3. Create Test Configuration
Create or modify `config/default.yaml`:

```yaml
benchmark_name: "gsm8k"  # or other supported benchmark
benchmark_path: "/path/to/benchmark/data"
model_names:
  - "gpt-4"
  - "gpt-3.5-turbo"
  - "llama-base"
method_name: "your-method"  # Name of your directory in evaluate/
output_dir: "./results"
batch_size: 10
```

### 4. Run Tests
```bash
python main.py --config config/default.yaml
```

## Reference Implementations

The `evaluated_methods` directory contains implementations that were evaluated for our paper. You can use these as examples for implementation patterns and expected outputs.

## Supported Models

- GPT-4
- GPT-3.5-turbo
- Llama (base)
- Llama (large)
- Mistral (base)
- Phi-2

## Supported Benchmarks

### GSM8K
- Mathematical reasoning problems
- Evaluation based on numerical answer accuracy
- Expected answer format: "#### <number>"

### TruthfulQA
- Tests model truthfulness
- Multiple-choice and free-response evaluation
- Automatic scoring based on reference answers

## Adding New Benchmarks

The framework supports two ways to add new benchmarks:

### 1. Using Hugging Face Datasets

```python
# benchmarks/data_loaders/huggingface_loader.py
from datasets import load_dataset
from typing import Iterator

def load_hf_benchmark(dataset_name: str, split: str = "test") -> Iterator:
    """Load a benchmark dataset from Hugging Face"""
    dataset = load_dataset(dataset_name, split=split)
    return dataset

class CustomHFBenchmarkEvaluator(BaseEvaluator):
    def __init__(self, dataset_name: str, split: str = "test"):
        self.dataset = load_hf_benchmark(dataset_name, split)
    
    def get_problems(self) -> Iterator:
        return self.dataset
```

Configuration for Hugging Face dataset:
```yaml
benchmark_name: "custom_hf"
benchmark_config:
  dataset_name: "bigscience/P3"
  split: "test"
```

### 2. Adding Custom Benchmark Data

1. Create a new directory in `benchmarks/data/`:
```
benchmarks/
├── data/
│   ├── your_benchmark/
│   │   ├── __init__.py
│   │   ├── data/               # Your benchmark data files
│   │   └── metadata.yaml       # Benchmark metadata
```

2. Create metadata.yaml:
```yaml
name: "your_benchmark"
version: "1.0"
description: "Description of your benchmark"
input_format: "Description of input format"
output_format: "Description of expected outputs"
metrics:
  - "metric1"
  - "metric2"
citation: "Optional citation"
```

3. Implement your evaluator:
```python
from typing import Dict, Iterator
from .base import BaseEvaluator

class YourBenchmarkEvaluator(BaseEvaluator):
    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.data = self._load_data()
    
    def _load_data(self):
        """Load your benchmark data"""
        pass
    
    def get_problems(self) -> Iterator:
        yield from self.data
    
    def evaluate(self, prediction: str, reference: Any) -> Dict:
        """Implement evaluation logic"""
        pass
```

## Results

Results are saved in YAML format in the `results` directory:
```yaml
method_name:
  problem_id:
    model_outputs:
      gpt-4:
        solution: "model's solution"
        score: 1.0
      gpt-3.5-turbo:
        solution: "another solution"
        score: 0.8
    evaluation:
      accuracy: 0.9
      other_metrics:
        metric1: 0.85
        metric2: 0.92
```

## Running on HPC

The framework includes a Slurm script for running on HPC clusters:
```bash
sbatch run.slurm
```

Modify `run.slurm` to match your cluster's configuration and requirements.

## Development

### Code Quality
Run linting checks:
```bash
./run_pylint.sh
```

### Tests
Run unit tests with coverage:
```bash
./run_coverage.sh
```

## Common Issues and Troubleshooting

1. API Key Errors:
   - Ensure all required API keys are properly exported
   - Check API key permissions and quotas

2. Import Errors:
   - Verify conda environment is activated
   - Check all dependencies are installed

3. Data Loading Issues:
   - Verify benchmark data paths are correct
   - Check data format matches evaluator expectations

4. Method Loading Issues:
   - Ensure your method is in the correct directory (`evaluate/`)
   - Verify method configuration exists in `config/methods/`
   - Check entry module and function names match your implementation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

Guidelines:
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation as needed

## Citation

If you use this framework in your research, please cite:
```
@misc{llm_reliability_framework,
    author = {Michael Jerge},
    title = {LLM Reliability Framework},
    year = {2024},
    publisher = {GitHub},
    url = {https://github.com/yourusername/llm_reliability_framework}
}
```

## License

See LICENSE file for details

## Contact

For questions and support:
- Create an issue in the repository
- Contact: mj6ux@virginia.edu
