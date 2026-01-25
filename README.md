# AI/ML Domain LLM

An engineering-focused project covering data acquisition, dataset curation, tokenization, training, and evaluation of a domain-specific AI/ML LLM.

## Project Goal

The primary goal of this project is to train a domain-specific Large Language Model (LLM) from scratch, focused on Artificial Intelligence and Machine Learning knowledge.

This repository documents and implements the entire LLM pipeline end-to-end, with a strong emphasis on engineering clarity rather than speed or scale.

**This is not about using pre-trained APIs. This is about understanding and building the system.**

## Scope of the Project

This repository covers:

### 1. Data Acquisition
- Automated and semi-automated collection of AI/ML-related books and resources
- Topic-driven data gathering (subdomains within AI/ML)

### 2. Dataset Curation
- Metadata generation
- Duplicate detection
- Content validation and filtering

### 3. Tokenization
- Text normalization
- Vocabulary construction
- Token statistics and analysis

### 4. Model Training
- Training a language model from scratch
- Domain-focused learning objectives

### 5. Evaluation
- Domain-specific evaluation prompts
- Qualitative and quantitative assessment

### 6. Production Deployment
- Model optimization and quantization
- Inference pipeline setup
- API endpoint design and implementation
- Performance monitoring and logging

## System Design Philosophy

- Engineering-first approach
- Automation where possible, manual control where necessary
- Slow is acceptable, correctness is not optional
- Reproducibility over convenience
- Production-ready code from the start

The system is designed to reduce manual effort while keeping the learning process transparent and debuggable.

## Repository Structure (Planned)

```
ai-ml-domain-llm/
│
├── data-collection/      # Agents, tools, metadata generators
├── data-checking/        # Deduplication, validation, statistics
├── datasets/             # Curated and processed datasets
├── tokenization/         # Tokenizer experiments and analysis
├── training/             # Model architecture and training scripts
├── evaluation/           # Evaluation prompts and metrics
├── deployment/           # Production deployment configurations
├── docs/                 # Architecture and design notes
└── README.md
```

## Current Status

- Project initialization
- Data collection architecture design
- Tooling for metadata and duplicate checking

The repository will evolve incrementally as each stage of the pipeline is implemented.

## Team

This project is developed collaboratively by a small team, with shared responsibility across system design, data engineering, and model experimentation.

## Disclaimer

This project is intended for educational and research purposes only. All data handling is performed with respect to learning objectives and system design exploration.

## Why This Project Exists

Most people use LLMs. Few understand how they are built and deployed.

This project exists to gain real, hands-on experience with LLM engineering—from data collection through training to production deployment. It's an end-to-end journey through the complete lifecycle of building and shipping a language model.

**From raw text → tokens → parameters → behavior → production.**
