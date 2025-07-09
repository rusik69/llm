# ShittyLLM Makefile

# Variables
BINARY_NAME=shittyllm
MAIN_FILE=cmd/shittyllm/main.go

# Go commands
GO=go
GO_BUILD=$(GO) build
GO_TEST=$(GO) test
GO_CLEAN=$(GO) clean
GO_MOD=$(GO) mod

# Default target
.PHONY: all
all: build

# Build the binary
.PHONY: build
build:
	@echo "Building $(BINARY_NAME)..."
	$(GO_BUILD) -o $(BINARY_NAME) $(MAIN_FILE)

# Run tests
.PHONY: test
test:
	@echo "Running tests..."
	$(GO_TEST) -v ./...

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
	$(GO_TEST) -v -cover ./...

# Run benchmarks
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
	rm -f coverage.out coverage.html

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

# Train the model
.PHONY: train
train: build
	@echo "Training the model..."
	./$(BINARY_NAME) -mode=train -epochs=10 -lr=0.001

# Generate text
.PHONY: generate
generate: build
	@echo "Generating text..."
	./$(BINARY_NAME) -mode=generate -text="$(or $(TEXT),Hello world)" -temperature=0.8 -maxtokens=50

# Download Wikipedia data
.PHONY: download-wiki
download-wiki: build
	@echo "Downloading Wikipedia data..."
	./$(BINARY_NAME) -mode=download -sample=false -datadir=data -maxarticles=100000

# Fast parallel training with Wikipedia data (uses all CPU cores)
.PHONY: train-wiki-parallel
train-wiki-parallel: build
	@echo "🚀 Fast parallel training with Wikipedia data (using all CPU cores)..."
	@echo "System CPU count: $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 'unknown')"
	@if [ ! -f data/training_data.txt ]; then \
		echo "Training data not found. Downloading..."; \
		make download-wiki > /dev/null 2>&1; \
	fi
	@echo "Starting fast parallel training..."
	./$(BINARY_NAME) -mode=train -data=data/training_data.txt -epochs=5 -lr=0.01

# Show help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build          - Build the binary"
	@echo "  test           - Run all tests"
	@echo "  test-race      - Run tests with race detection"
	@echo "  test-integration - Run integration tests"
	@echo "  test-coverage  - Run tests with coverage"
	@echo "  bench          - Run benchmarks"
	@echo "  clean          - Clean build artifacts"
	@echo "  fmt            - Format code"
	@echo "  lint           - Run linter"
	@echo "  tidy           - Tidy dependencies"
	@echo "  train          - Train the model"
	@echo "  generate       - Generate text (TEXT='prompt')"
	@echo "  download-wiki  - Download Wikipedia data (up to 100K articles)"
	@echo "  train-wiki-parallel - 🚀 Fast parallel training using all CPU cores (2 epochs, LR=0.03)"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "System info:"
	@echo "  CPU cores: $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 'unknown')"

# Default goal
.DEFAULT_GOAL := help 