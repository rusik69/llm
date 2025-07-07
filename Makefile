# Simple LLM Makefile
# 
# This Makefile provides convenient targets for building, testing, and running
# the simple LLM implementation.

# Variables
BINARY_NAME=shittyllm
MAIN_FILE=cmd/shittyllm/main.go
BUILD_DIR=build
TEST_TIMEOUT=30s

# Go commands
GO=go
GO_BUILD=$(GO) build
GO_TEST=$(GO) test
GO_CLEAN=$(GO) clean
GO_GET=$(GO) get
GO_MOD=$(GO) mod

# Default target
.PHONY: all
all: build

# Build the binary
.PHONY: build
build:
	@echo "Building $(BINARY_NAME)..."
	$(GO_BUILD) -o $(BINARY_NAME) $(MAIN_FILE)
	@echo "Build complete: $(BINARY_NAME)"

# Build for Linux (used in CI/releases)
.PHONY: build-linux
build-linux:
	@echo "Building $(BINARY_NAME) for Linux..."
	GOOS=linux GOARCH=amd64 $(GO_BUILD) -o $(BINARY_NAME) $(MAIN_FILE)
	@echo "Linux build complete: $(BINARY_NAME)"

# Build for Linux ARM64
.PHONY: build-linux-arm64
build-linux-arm64:
	@echo "Building $(BINARY_NAME) for Linux ARM64..."
	GOOS=linux GOARCH=arm64 $(GO_BUILD) -o $(BINARY_NAME) $(MAIN_FILE)
	@echo "Linux ARM64 build complete: $(BINARY_NAME)"

# Build for multiple platforms
.PHONY: build-all
build-all:
	@echo "Building for multiple platforms..."
	@mkdir -p $(BUILD_DIR)
	GOOS=linux GOARCH=amd64 $(GO_BUILD) -o $(BUILD_DIR)/$(BINARY_NAME)-linux-amd64 $(MAIN_FILE)
	GOOS=darwin GOARCH=amd64 $(GO_BUILD) -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-amd64 $(MAIN_FILE)
	GOOS=darwin GOARCH=arm64 $(GO_BUILD) -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-arm64 $(MAIN_FILE)
	GOOS=windows GOARCH=amd64 $(GO_BUILD) -o $(BUILD_DIR)/$(BINARY_NAME)-windows-amd64.exe $(MAIN_FILE)
	@echo "Multi-platform build complete"

# Run tests
.PHONY: test
test:
	@echo "Running tests..."
	$(GO_TEST) -timeout $(TEST_TIMEOUT) -v ./...

# Run tests with race detection
.PHONY: test-race
test-race:
	@echo "Running tests with race detection..."
	$(GO_TEST) -timeout 5m -v -race ./...

# Run integration tests
.PHONY: test-integration
test-integration:
	@echo "Running integration tests..."
	$(GO_TEST) -timeout 10m -v -tags=integration ./pkg/data

# Run tests with coverage
.PHONY: test-coverage
test-coverage:
	@echo "Running tests with coverage..."
	$(GO_TEST) -timeout $(TEST_TIMEOUT) -v -cover ./...

# Run tests with coverage report
.PHONY: test-coverage-html
test-coverage-html:
	@echo "Running tests with HTML coverage report..."
	$(GO_TEST) -timeout $(TEST_TIMEOUT) -v -coverprofile=coverage.out ./...
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

# Run specific package tests
.PHONY: test-matrix
test-matrix:
	@echo "Running matrix package tests..."
	$(GO_TEST) -timeout $(TEST_TIMEOUT) -v ./pkg/matrix

.PHONY: test-tokenizer
test-tokenizer:
	@echo "Running tokenizer package tests..."
	$(GO_TEST) -timeout $(TEST_TIMEOUT) -v ./pkg/tokenizer

.PHONY: test-llm
test-llm:
	@echo "Running LLM package tests..."
	$(GO_TEST) -timeout $(TEST_TIMEOUT) -v ./pkg/llm

.PHONY: test-training
test-training:
	@echo "Running training package tests..."
	$(GO_TEST) -timeout $(TEST_TIMEOUT) -v ./pkg/training

.PHONY: test-data
test-data:
	@echo "Running data package tests..."
	$(GO_TEST) -timeout $(TEST_TIMEOUT) -v ./pkg/data

# Benchmark tests
.PHONY: bench
bench:
	@echo "Running benchmarks..."
	$(GO_TEST) -bench=. -benchmem ./...

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	$(GO_CLEAN)
	rm -f $(BINARY_NAME)
	rm -rf $(BUILD_DIR)
	rm -f coverage.out coverage.html
	@echo "Clean complete"

# Format code
.PHONY: fmt
fmt:
	@echo "Formatting code..."
	$(GO) fmt ./...

# Run linter
.PHONY: lint
lint:
	@echo "Running linter..."
	@if command -v golangci-lint >/dev/null 2>&1; then \
		echo "Using golangci-lint"; \
		golangci-lint run ./...; \
	else \
		echo "golangci-lint not found, using go vet instead"; \
		$(GO) vet ./...; \
	fi

# Tidy dependencies
.PHONY: tidy
tidy:
	@echo "Tidying dependencies..."
	$(GO_MOD) tidy

# Get and update dependencies
.PHONY: get
get:
	@echo "Getting dependencies..."
	$(GO_GET) ./...

# Update all dependencies
.PHONY: get-update
get-update:
	@echo "Updating all dependencies..."
	@echo "Note: This project uses only Go standard library packages"
	$(GO_MOD) tidy

# Run the application in training mode
.PHONY: train
train: build
	@echo "Training the model..."
	./$(BINARY_NAME) -mode=train -epochs=10 -lr=0.001

# Run the application in training mode with custom parameters
.PHONY: train-custom
train-custom: build
	@echo "Training the model with custom parameters..."
	@echo "Usage: make train-custom EPOCHS=20 LR=0.01"
	./$(BINARY_NAME) -mode=train -epochs=$(or $(EPOCHS),10) -lr=$(or $(LR),0.001)

# Run the application in generation mode
.PHONY: generate
generate: build
	@echo "Generating text..."
	@echo "Usage: make generate TEXT='Hello world' TEMP=0.8 MAXTOKENS=50"
	./$(BINARY_NAME) -mode=generate -text="$(or $(TEXT),Hello world)" -temperature=$(or $(TEMP),0.8) -maxtokens=$(or $(MAXTOKENS),50)

# Download Wikipedia data
.PHONY: download-wiki
download-wiki: build
	@echo "Downloading Wikipedia data..."
	./$(BINARY_NAME) -mode=download -sample=true -datadir=data

# Download full Wikipedia data (larger)
.PHONY: download-wiki-full
download-wiki-full: build
	@echo "Downloading full Wikipedia data..."
	@echo "Warning: This will download a large file (~500MB+)"
	./$(BINARY_NAME) -mode=download -sample=false -datadir=data -maxarticles=5000

# Train with Wikipedia data
.PHONY: train-wiki
train-wiki: build
	@echo "Training with Wikipedia data..."
	@if [ ! -f data/training_data.txt ]; then \
		echo "Training data not found. Downloading sample data..."; \
		make download-wiki; \
	fi
	./$(BINARY_NAME) -mode=train -data=data/training_data.txt -epochs=10 -lr=0.001

# Train with Wikipedia data (custom parameters)
.PHONY: train-wiki-custom
train-wiki-custom: build
	@echo "Training with Wikipedia data (custom parameters)..."
	@echo "Usage: make train-wiki-custom EPOCHS=20 LR=0.01"
	@if [ ! -f data/training_data.txt ]; then \
		echo "Training data not found. Downloading sample data..."; \
		make download-wiki; \
	fi
	./$(BINARY_NAME) -mode=train -data=data/training_data.txt -epochs=$(or $(EPOCHS),10) -lr=$(or $(LR),0.001)

# Quick test with simple text
.PHONY: demo
demo: build
	@echo "Running demo..."
	@echo "1. Training the model..."
	./$(BINARY_NAME) -mode=train -epochs=5 -lr=0.001
	@echo ""
	@echo "2. Generating text..."
	./$(BINARY_NAME) -mode=generate -text="Hello world" -temperature=0.8 -maxtokens=30
	@echo ""
	./$(BINARY_NAME) -mode=generate -text="The quick brown" -temperature=0.9 -maxtokens=25

# Demo with Wikipedia data
.PHONY: demo-wiki
demo-wiki: build
	@echo "Running Wikipedia demo..."
	@echo "1. Downloading Wikipedia data..."
	make download-wiki
	@echo ""
	@echo "2. Training with Wikipedia data..."
	./$(BINARY_NAME) -mode=train -data=data/training_data.txt -epochs=5 -lr=0.001
	@echo ""
	@echo "3. Generating text..."
	./$(BINARY_NAME) -mode=generate -text="Albert Einstein" -temperature=0.7 -maxtokens=40
	@echo ""
	./$(BINARY_NAME) -mode=generate -text="Computer programming" -temperature=0.8 -maxtokens=35

# Install dependencies (if any external deps are added)
.PHONY: deps
deps:
	@echo "Installing dependencies..."
	$(GO_MOD) download

# Development setup
.PHONY: dev-setup
dev-setup: deps tidy fmt lint test
	@echo "Development setup complete"

# CI pipeline
.PHONY: ci
ci: fmt lint test build
	@echo "CI pipeline complete"

# CI pipeline with race detection
.PHONY: ci-check
ci-check: tidy fmt lint test-race build
	@echo "CI pipeline with race detection complete"

# Show help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build          - Build the binary"
	@echo "  build-linux    - Build the binary for Linux"
	@echo "  build-linux-arm64 - Build the binary for Linux ARM64"
	@echo "  build-all      - Build for multiple platforms"
	@echo "  test           - Run all tests"
	@echo "  test-race      - Run tests with race detection"
	@echo "  test-integration - Run integration tests"
	@echo "  test-coverage  - Run tests with coverage"
	@echo "  test-matrix    - Run matrix package tests"
	@echo "  test-tokenizer - Run tokenizer package tests"
	@echo "  test-llm       - Run LLM package tests"
	@echo "  test-training  - Run training package tests"
	@echo "  test-data      - Run data package tests"
	@echo "  bench          - Run benchmarks"
	@echo "  clean          - Clean build artifacts"
	@echo "  fmt            - Format code"
	@echo "  lint           - Run linter"
	@echo "  tidy           - Tidy dependencies"
	@echo "  get            - Get dependencies"
	@echo "  get-update     - Update all dependencies"
	@echo "  train          - Train the model"
	@echo "  train-custom   - Train with custom params (EPOCHS=N LR=X)"
	@echo "  train-wiki     - Train with Wikipedia data"
	@echo "  train-wiki-custom - Train with Wikipedia data (custom params)"
	@echo "  generate       - Generate text (TEXT='prompt' TEMP=0.8 MAXTOKENS=50)"
	@echo "  demo           - Run a quick demo"
	@echo "  demo-wiki      - Run Wikipedia demo"
	@echo "  download-wiki  - Download Wikipedia sample data"
	@echo "  download-wiki-full - Download full Wikipedia data"
	@echo "  deps           - Install dependencies"
	@echo "  dev-setup      - Complete development setup"
	@echo "  ci             - Run CI pipeline"
	@echo "  ci-check       - Run CI pipeline with race detection"
	@echo "  help           - Show this help message"

# Default goal
.DEFAULT_GOAL := help 