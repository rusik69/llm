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
	@echo "  help           - Show this help message"

# Default goal
.DEFAULT_GOAL := help 