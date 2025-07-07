# Simple LLM from Scratch in Go

A minimal implementation of a Large Language Model (LLM) built from scratch in Go, featuring a transformer architecture with self-attention mechanism.

## Features

- **Pure Go Implementation**: No external ML frameworks required
- **Transformer Architecture**: Includes self-attention, positional encoding, and feedforward layers
- **Custom Matrix Operations**: Built-in matrix operations for neural network computations
- **Simple Tokenizer**: Word-based tokenization with special tokens
- **Wikipedia Data Integration**: Download and train on Wikipedia articles
- **Training & Inference**: Both training and text generation capabilities
- **CLI Interface**: Easy-to-use command line interface

## Architecture

The LLM consists of several key components:

1. **Matrix Operations** (`pkg/matrix/`): Basic linear algebra operations
2. **Tokenizer** (`pkg/tokenizer/`): Text preprocessing and vocabulary management
3. **Data Processing** (`pkg/data/`): Wikipedia data download and processing
4. **Transformer Layers** (`pkg/llm/transformer.go`): 
   - Self-attention mechanism
   - Feed-forward networks
   - Positional encoding
   - Embedding layers
5. **Model Core** (`pkg/llm/model.go`): Main LLM structure and forward pass
6. **Training** (`pkg/training/`): Training loop and weight updates

## Usage

### Building

```bash
# Using Go directly
go build -o llm main.go

# Using Makefile
make build
```

### Testing

Run the comprehensive test suite:

```bash
# Run all tests
make test

# Run specific package tests
make test-matrix
make test-tokenizer
make test-data
make test-llm
make test-training

# Run tests with coverage
make test-coverage

# Generate HTML coverage report
make test-coverage-html
```

Test coverage:
- Matrix package: **96.2%**
- Tokenizer package: **96.6%**
- Data package: **100.0%**
- LLM package: **96.2%**
- Training package: **100.0%**

### Training

Train the model on built-in sample text:

```bash
# Using binary directly
./llm -mode=train -epochs=10 -lr=0.001

# Using Makefile
make train

# Custom training parameters
make train-custom EPOCHS=20 LR=0.01
```

Parameters:
- `-epochs`: Number of training epochs (default: 10)
- `-lr`: Learning rate (default: 0.001)

### Wikipedia Data

Download and train on Wikipedia articles for higher quality training data:

```bash
# Download Wikipedia sample data
make download-wiki

# Download full Wikipedia data (larger dataset)
make download-wiki-full

# Train with Wikipedia data
make train-wiki

# Train with custom parameters
make train-wiki-custom EPOCHS=20 LR=0.01

# Complete Wikipedia demo
make demo-wiki
```

You can also use the CLI directly:

```bash
# Download Wikipedia data
./llm -mode=download -sample=true -datadir=data

# Train with Wikipedia data
./llm -mode=train -data=data/training_data.txt -epochs=10 -lr=0.001
```

Parameters:
- `-sample`: Use sample data instead of full Wikipedia dump
- `-datadir`: Directory to store downloaded data (default: ./data)
- `-maxarticles`: Maximum number of articles to process (default: 1000)
- `-data`: Path to training data file

### Text Generation

Generate text based on a prompt:

```bash
# Using binary directly
./llm -mode=generate -text="Hello world"

# Using Makefile
make generate TEXT="Hello world"
```

Parameters:
- `-text`: Input prompt for text generation

### Quick Demo

Run a complete demo with training and generation:

```bash
make demo
```

## Model Architecture Details

- **Vocabulary Size**: Dynamic (based on training text)
- **Embedding Dimension**: 64
- **Hidden Size**: 128
- **Number of Layers**: 2
- **Attention Heads**: 4
- **Max Sequence Length**: 512

## Example Output

### Basic Training
```bash
$ ./llm -mode=train -epochs=3
Starting LLM training...
Starting training with 39 tokens for 3 epochs
Epoch 1: Average Loss = 6.1115
Epoch 2: Average Loss = 6.1166
Epoch 3: Average Loss = 6.1220
Training completed!

$ ./llm -mode=generate -text="Hello world"
Generating text...
Input: Hello world
Generated: hello world
```

### Wikipedia Training
```bash
$ make demo-wiki
Running Wikipedia demo...
1. Downloading Wikipedia data...
Downloading Wikipedia data...
Created sample training data at data/sample_data.txt
Parsing Wikipedia dump: data/sample_data.txt
Saved 10 articles to data/training_data.txt
Successfully processed 10 articles

2. Training with Wikipedia data...
Starting LLM training...
Loaded 489 tokens from data/training_data.txt
Starting training with 489 tokens for 5 epochs
Epoch 1: Average Loss = 6.3492
Epoch 2: Average Loss = 6.3778
Epoch 3: Average Loss = 6.4499
Epoch 4: Average Loss = 6.5698
Epoch 5: Average Loss = 6.7425
Training completed!

3. Generating text...
Input: Albert Einstein
Generated: albert einstein
Input: Computer programming
Generated: computer programming
```

## Educational Purpose

This implementation is designed for educational purposes to demonstrate:

- How transformer architectures work
- Basic neural network operations in Go
- Text tokenization and processing
- Autoregressive language model training

**Note**: This is a simplified implementation. Production LLMs require:
- Proper backpropagation algorithm
- Advanced optimization techniques (Adam, learning rate scheduling)
- Large-scale training datasets
- More sophisticated architectures
- GPU acceleration
- Model parallelization

## Code Structure

```
├── main.go                 # CLI entry point
├── Makefile               # Build and test automation
├── pkg/
│   ├── matrix/
│   │   ├── matrix.go      # Matrix operations
│   │   └── matrix_test.go # Matrix tests
│   ├── tokenizer/
│   │   ├── tokenizer.go   # Text tokenization  
│   │   └── tokenizer_test.go # Tokenizer tests
│   ├── data/
│   │   ├── wikipedia.go   # Wikipedia data download/processing
│   │   └── wikipedia_test.go # Data processing tests
│   ├── llm/
│   │   ├── model.go       # Main LLM model
│   │   ├── model_test.go  # Model tests
│   │   └── transformer.go # Transformer components
│   └── training/
│       ├── trainer.go     # Training logic
│       └── trainer_test.go # Training tests
├── go.mod                 # Go module definition
└── README.md             # Project documentation
```

### Makefile Targets

```bash
make help                # Show all available targets
make build               # Build the binary
make test                # Run all tests
make test-coverage       # Run tests with coverage
make clean               # Clean build artifacts
make demo                # Run training and generation demo
make demo-wiki           # Run Wikipedia demo
make train               # Train the model
make train-wiki          # Train with Wikipedia data
make generate            # Generate text with prompt
make download-wiki       # Download Wikipedia sample data
make download-wiki-full  # Download full Wikipedia data
```

## Contributing

This is an educational project. Feel free to:
- Add more sophisticated training algorithms
- Implement proper gradient computation
- Add more tokenization strategies
- Experiment with different architectures
- Add model persistence (save/load)

## License

MIT License - feel free to use this code for learning and experimentation. 