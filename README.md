# LLM Evaluation Framework

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

This framework allows you to evaluate how your method performs across different language models and benchmarks. The testing process involves several key steps, each serving a specific purpose:

### 1. Add Your Method to the Framework

Place your method's code in the `evaluate` directory with the following structure:
```
evaluate/
└── your-method/              # Your method's directory
    ├── method/              # Your implementation
    │   ├── __init__.py
    │   └── main.py         # Contains your solve function
    └── requirements.txt     # Any additional dependencies
```

Your entry point function should follow this interface:
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

**Why this structure?**
- The `evaluate` directory is separate from `evaluated_methods` to clearly distinguish your method from reference implementations
- The standardized structure ensures the framework can reliably find and load your code
- The common interface (`model, input_text -> str`) allows your method to work with any supported model and benchmark

**Important Considerations:**
- Your method receives a model instance that handles all API calls and token management
- The input text format depends on the benchmark being used (e.g., math problems for GSM8K, questions for TruthfulQA)
- Your method must return a string in the format expected by the benchmark (e.g., "#### 42" for GSM8K)

### 2. Configure Your Method

Create a configuration file in `config/methods/your-method.yaml`:
```yaml
entry_module: method.main        # Python module path to your code
entry_function: solve_function   # Function name to call
description: "Brief description of your method"
```

**Why Configuration Files?**
- Separates implementation from configuration
- Allows changing entry points without modifying code
- Provides documentation and metadata about your method
- Enables the framework to properly load and execute your code

**Configuration Options:**
- `entry_module`: Path to your Python module (relative to your method directory)
- `entry_function`: Name of the function that implements your method
- Additional parameters specific to your method can be added here

### 3. Create Test Configuration

Create or modify `config/default.yaml` to specify how to test your method:
```yaml
benchmark_name: "gsm8k"         # Which benchmark to use
benchmark_path: "/path/to/data" # Where benchmark data is stored
model_names:                    # Which models to test with
  - "gpt-4"
  - "gpt-3.5-turbo"
  - "llama-base"
method_name: "your-method"      # Your method's directory name
output_dir: "./results"         # Where to save results
batch_size: 10                  # How many problems to process at once
```

**Why These Settings?**
- `benchmark_name`: Different benchmarks test different capabilities (reasoning, truthfulness, etc.)
- `model_names`: Testing across models reveals how your method performs with different LLMs
- `batch_size`: Controls memory usage and allows for efficient processing
- `output_dir`: Organizes results for analysis and comparison

**Benchmark Selection Considerations:**
- GSM8K: Best for testing mathematical reasoning
- TruthfulQA: Evaluates model truthfulness and factual accuracy
- Choose based on what aspect of LLM behavior your method aims to improve

### 4. Run Tests

Execute the framework:
```bash
python main.py --config config/default.yaml
```

**What Happens During Testing:**
1. Framework loads your method and configuration
2. Initializes specified models
3. Loads benchmark data
4. For each problem in the benchmark:
   - Passes the problem to your method with each model
   - Evaluates responses using benchmark-specific metrics
   - Records results and any errors
5. Saves detailed results in YAML format

**Understanding Results:**
Results are saved in `results/your-method/` with the following structure:
```yaml
method_name:
  problem_id:
    model_outputs:           # Raw outputs from each model
      gpt-4:
        solution: "..."      # Your method's solution
        score: 1.0          # Benchmark-specific score
    evaluation:             # Overall evaluation metrics
      accuracy: 0.9         # Percentage correct
      other_metrics:        # Benchmark-specific metrics
        metric1: 0.85
```

These results allow you to:
- Compare performance across different models
- Identify where your method succeeds or fails
- Compare against baseline methods
- Generate statistics for research papers

### 5. Compare with Reference Implementations

The `evaluated_methods` directory contains previously tested methods that you can use to:
- Understand implementation patterns
- Compare performance against established baselines
- Verify your results are reasonable

Each reference implementation includes:
- Complete source code
- Configuration files
- Documentation of results

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
