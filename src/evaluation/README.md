# O-RAN RAG Evaluation Framework

This directory contains the evaluation framework for the O-RAN RAG system. It allows for the evaluation of different RAG pipelines against a set of Q&A pairs.

## Components

- `evaluator.py`: Core evaluation functionality and metrics calculation
- `run_evaluation.py`: Command-line script for running evaluations with different pipelines
- `rat_processor_eval.py`: Implementation of RAT (Retrieval-Augmented Thinking) pipeline for evaluation

## Usage

### Running an Evaluation

The `run_evaluation.py` script provides a command-line interface for running evaluations:

```bash
python src/evaluation/run_evaluation.py --pipeline [default|chain_of_rag|rat] [OPTIONS]
```

### Available Pipelines

- `default`: Standard RAG pipeline
- `chain_of_rag`: Chain of RAG pipeline (multi-step retrieval and reasoning)
- `rat`: Retrieval-Augmented Thinking pipeline (iterative retrieval and reasoning)

### Options

- `--pipeline [PIPELINE]`: Select the RAG pipeline to evaluate (default, chain_of_rag, rat)
- `--config [CONFIG_FILE]`: Path to the configuration file
- `--num-questions [NUM]`: Number of questions to evaluate
- `--max-workers [NUM]`: Maximum number of worker threads for parallel evaluation
- `--results-dir [DIR]`: Directory to save evaluation results
- `--visualize`: Generate visualization of evaluation results
- `--create-sample-dataset`: Create a sample Q&A dataset if none exists

### Examples

Evaluate 10 questions using the default RAG pipeline:
```bash
python src/evaluation/run_evaluation.py --pipeline default --num-questions 10
```

Evaluate 5 questions using the Chain of RAG pipeline:
```bash
python src/evaluation/run_evaluation.py --pipeline chain_of_rag --num-questions 5
```

Evaluate 3 questions using the RAT pipeline and visualize results:
```bash
python src/evaluation/run_evaluation.py --pipeline rat --num-questions 3 --visualize
```

Create a sample dataset and evaluate:
```bash
python src/evaluation/run_evaluation.py --pipeline default --create-sample-dataset
```

## Output

The evaluation script generates an Excel file with the evaluation results in the specified results directory, organized by pipeline type. The results include:

- Questions and correct answers
- Predicted answers from the selected RAG pipeline
- Predicted answers from the Gemini model (for comparison)
- Correctness assessment for each answer
- Accuracy metrics for both the RAG pipeline and Gemini 