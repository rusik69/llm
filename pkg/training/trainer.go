package training

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"sync"

	"github.com/rusik69/shittyllm/pkg/llm"
	"github.com/rusik69/shittyllm/pkg/tokenizer"
)

// Trainer handles model training with proper backpropagation
type Trainer struct {
	Model     *llm.LLM
	Optimizer *llm.AdamOptimizer
	Config    *TrainingConfig
}

// TrainingConfig holds training configuration
type TrainingConfig struct {
	LearningRate    float64
	BatchSize       int
	MaxGradNorm     float64 // For gradient clipping
	ValidationSplit float64 // Fraction of data to use for validation
	WarmupSteps     int
	UseWarmup       bool
	NumWorkers      int // Number of parallel workers for training
}

// DefaultTrainingConfig returns default training configuration
func DefaultTrainingConfig() *TrainingConfig {
	return &TrainingConfig{
		LearningRate:    0.001,
		BatchSize:       64, // Increased for better parallel processing
		MaxGradNorm:     1.0,
		ValidationSplit: 0.0,              // Disable validation for speed
		WarmupSteps:     100,              // Reduced warmup steps
		UseWarmup:       false,            // Disable warmup for debugging
		NumWorkers:      runtime.NumCPU(), // Use all available CPU cores
	}
}

// New creates a new trainer
func New(model *llm.LLM, learningRate float64) *Trainer {
	config := DefaultTrainingConfig()
	config.LearningRate = learningRate

	optimizerConfig := llm.DefaultOptimizerConfig()
	optimizerConfig.LearningRate = learningRate

	return &Trainer{
		Model:     model,
		Optimizer: llm.NewAdamOptimizer(model, optimizerConfig),
		Config:    config,
	}
}

// NewWithConfig creates a new trainer with custom configuration
func NewWithConfig(model *llm.LLM, config *TrainingConfig) *Trainer {
	if config == nil {
		config = DefaultTrainingConfig()
	}

	optimizerConfig := llm.DefaultOptimizerConfig()
	optimizerConfig.LearningRate = config.LearningRate

	return &Trainer{
		Model:     model,
		Optimizer: llm.NewAdamOptimizer(model, optimizerConfig),
		Config:    config,
	}
}

// Train performs efficient training with batching and sampling
func (t *Trainer) Train(tokens []int, epochs int) error {
	if len(tokens) < 2 {
		return fmt.Errorf("training sequence must have at least 2 tokens")
	}

	fmt.Printf("Starting training with %d tokens for %d epochs\n", len(tokens), epochs)
	fmt.Printf("Using Adam optimizer with learning rate: %.6f\n", t.Config.LearningRate)
	fmt.Printf("Training configuration: %d workers, batch size %d\n", t.Config.NumWorkers, t.Config.BatchSize)

	// Set model to training mode
	t.Model.Train()

	// Configure training parameters for maximum speed
	maxSequenceLength := 10                      // Reduced from 20 for faster processing
	samplesPerEpoch := min(len(tokens)/20, 1000) // Further reduced for lightning speed
	// Ensure batch size is large enough for parallel processing by default
	minParallelBatch := t.Config.NumWorkers * 4                     // 4 examples per worker
	batchSize := max(min(t.Config.BatchSize, 32), minParallelBatch) // Larger batches for parallel processing

	// Fast mode for moderate learning rates
	if t.Config.LearningRate >= 0.03 {
		maxSequenceLength = 8                      // Shorter sequences
		samplesPerEpoch = min(len(tokens)/30, 800) // Fewer samples
		// Ensure batch size is large enough for parallel processing
		minParallelBatch := t.Config.NumWorkers * 4                    // 4 examples per worker (reduced from 8)
		batchSize = max(min(t.Config.BatchSize, 64), minParallelBatch) // Larger batches for parallel processing
		fmt.Printf("🚀 FAST MODE: Optimized parallel training activated\n")
	}

	// Ultra-fast mode for very high learning rates
	if t.Config.LearningRate >= 0.05 {
		maxSequenceLength = 5                      // Even shorter sequences
		samplesPerEpoch = min(len(tokens)/50, 500) // Minimal samples
		// Ensure batch size is large enough for parallel processing even in lightning mode
		minParallelBatch := t.Config.NumWorkers * 4                    // 4 examples per worker
		batchSize = max(min(t.Config.BatchSize, 32), minParallelBatch) // Larger batches for parallel processing
		fmt.Printf("⚡ LIGHTNING MODE: Ultra-fast training activated\n")
	}

	fmt.Printf("Training with %d samples per epoch, batch size %d\n", samplesPerEpoch, batchSize)

	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("Starting epoch %d/%d...\n", epoch+1, epochs)
		totalLoss := 0.0
		numBatches := 0
		processed := 0

		// Process in batches for efficiency
		for batchStart := 0; batchStart < samplesPerEpoch; batchStart += batchSize {
			batchEnd := min(batchStart+batchSize, samplesPerEpoch)
			currentBatchSize := batchEnd - batchStart

			// Debug: Show first batch
			if batchStart == 0 {
				fmt.Printf("Processing first batch (size: %d)...\n", currentBatchSize)
			}

			// Pre-allocate batch slices for better memory efficiency
			batchInputs := make([][]int, 0, currentBatchSize)
			batchTargets := make([][]int, 0, currentBatchSize)

			// Create batch of training examples
			for i := 0; i < currentBatchSize; i++ {
				// Sample random position from tokens (more efficient than sequential)
				maxPos := len(tokens) - 2 // Need at least 2 tokens (input + target)
				if maxPos < 0 {
					maxPos = 0
				}
				randomPos := rand.Intn(maxPos + 1)

				// Create input sequence
				seqLen := min(maxSequenceLength, randomPos+1)
				startIdx := randomPos + 1 - seqLen

				input := make([]int, seqLen)
				copy(input, tokens[startIdx:randomPos+1])

				target := []int{tokens[randomPos+1]}

				batchInputs = append(batchInputs, input)
				batchTargets = append(batchTargets, target)
			}

			// Train on batch
			batchLoss := t.TrainBatch(batchInputs, batchTargets)
			totalLoss += batchLoss
			numBatches++
			processed += currentBatchSize

			// Skip debug output for speed

			// Apply learning rate warmup if enabled
			if t.Config.UseWarmup && t.Optimizer.State.Step < int64(t.Config.WarmupSteps) {
				warmupLR := t.Config.LearningRate * float64(t.Optimizer.State.Step) / float64(t.Config.WarmupSteps)
				t.Optimizer.SetLearningRate(warmupLR)
			}

			// Progress indicator (more frequent for visibility)
			if numBatches%20 == 0 || batchStart+batchSize >= samplesPerEpoch {
				progress := float64(processed) / float64(samplesPerEpoch) * 100
				fmt.Printf("\rEpoch %d: %.0f%% complete, Loss: %.3f, Batches: %d",
					epoch+1, progress, batchLoss, numBatches)
			}
		}

		avgLoss := totalLoss / float64(numBatches)
		fmt.Printf("\nEpoch %d: Loss = %.3f\n", epoch+1, avgLoss)
	}

	// Set model to evaluation mode
	t.Model.Eval()
	fmt.Println("Training completed!")
	return nil
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// max returns the maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// TrainBatch performs training on a batch of sequences with proper backpropagation
func (t *Trainer) TrainBatch(inputs [][]int, targets [][]int) float64 {
	if len(inputs) != len(targets) {
		panic("Number of input sequences must match number of target sequences")
	}

	totalLoss := 0.0

	// Set model to training mode
	t.Model.Train()

	// Use parallel processing only if we have enough examples per worker (minimum 4 examples per worker)
	minExamplesPerWorker := 4
	if t.Config.NumWorkers > 1 && len(inputs) >= (t.Config.NumWorkers*minExamplesPerWorker) {
		// Debug: Show that we're using parallel processing
		if t.Optimizer.State.Step <= 1 {
			fmt.Printf("Using parallel processing with %d workers for batch size %d\n", t.Config.NumWorkers, len(inputs))
		}
		return t.trainBatchParallel(inputs, targets)
	}

	// Sequential processing for small batches
	if t.Optimizer.State.Step <= 1 {
		fmt.Printf("Using sequential processing for batch size %d\n", len(inputs))
	}
	for i := 0; i < len(inputs); i++ {
		// Compute loss (no debug output for speed)
		loss := t.Model.ComputeLoss(inputs[i], targets[i])
		totalLoss += loss

		// Show progress for every 10th example in first batch (unless in lightning mode)
		if t.Optimizer.State.Step <= 1 && i%10 == 0 && t.Config.LearningRate < 0.05 {
			fmt.Printf("  Processing example %d/%d...\n", i+1, len(inputs))
		}

		// Compute gradients
		gradients := t.Model.ComputeGradients(inputs[i], targets[i])

		// Apply gradient clipping
		if t.Config.MaxGradNorm > 0 {
			gradients.ClipGradients(t.Config.MaxGradNorm)
		}

		// Update parameters
		t.Optimizer.Step(t.Model, gradients)
	}

	return totalLoss / float64(len(inputs))
}

// trainBatchParallel performs parallel training on a batch of sequences
func (t *Trainer) trainBatchParallel(inputs [][]int, targets [][]int) float64 {
	numWorkers := t.Config.NumWorkers
	batchSize := len(inputs)
	chunkSize := batchSize / numWorkers

	// Debug: Show parallel setup
	if t.Optimizer.State.Step <= 1 {
		fmt.Printf("Parallel setup: %d examples, %d workers, chunk size %d\n", batchSize, numWorkers, chunkSize)
	}

	// Channels for collecting results
	type result struct {
		loss      float64
		gradients *llm.Gradients
	}

	results := make(chan result, numWorkers)
	var wg sync.WaitGroup

	// Process chunks in parallel
	for worker := 0; worker < numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if worker == numWorkers-1 {
			end = batchSize // Last worker handles remainder
		}

		wg.Add(1)
		go func(startIdx, endIdx int) {
			defer wg.Done()

			chunkLoss := 0.0
			var combinedGradients *llm.Gradients

			for i := startIdx; i < endIdx; i++ {
				// Compute loss (no debug output for speed)
				loss := t.Model.ComputeLoss(inputs[i], targets[i])
				chunkLoss += loss

				// Compute gradients
				gradients := t.Model.ComputeGradients(inputs[i], targets[i])

				// Accumulate gradients
				if combinedGradients == nil {
					combinedGradients = gradients
				} else {
					combinedGradients.Add(gradients)
				}
			}

			// Average gradients for this chunk
			if combinedGradients != nil {
				combinedGradients.Scale(1.0 / float64(endIdx-startIdx))
			}

			results <- result{
				loss:      chunkLoss,
				gradients: combinedGradients,
			}
		}(start, end)
	}

	// Wait for all workers to finish
	wg.Wait()
	close(results)

	// Debug: Show completion
	if t.Optimizer.State.Step <= 1 {
		fmt.Printf("All workers completed, collecting results...\n")
	}

	// Collect and combine results
	totalLoss := 0.0
	var finalGradients *llm.Gradients

	for res := range results {
		totalLoss += res.loss

		if finalGradients == nil {
			finalGradients = res.gradients
		} else if res.gradients != nil {
			finalGradients.Add(res.gradients)
		}
	}

	// Average gradients across all workers
	if finalGradients != nil {
		finalGradients.Scale(1.0 / float64(numWorkers))

		// Apply gradient clipping
		if t.Config.MaxGradNorm > 0 {
			finalGradients.ClipGradients(t.Config.MaxGradNorm)
		}

		// Update parameters with combined gradients
		t.Optimizer.Step(t.Model, finalGradients)
	}

	return totalLoss / float64(len(inputs))
}

// TrainOnBatch trains on a batch of sequences (legacy method - use TrainBatch instead)
func (t *Trainer) TrainOnBatch(sequences [][]int, targets [][]int) float64 {
	return t.TrainBatch(sequences, targets)
}

// PrepareTrainingData creates input-target pairs from a token sequence
func PrepareTrainingData(tokens []int, maxSeqLen int) ([][]int, [][]int) {
	var inputs [][]int
	var targets [][]int

	for i := 0; i < len(tokens)-1; i++ {
		// Create input sequence
		seqLen := int(math.Min(float64(i+1), float64(maxSeqLen)))
		startIdx := i + 1 - seqLen

		input := make([]int, seqLen)
		copy(input, tokens[startIdx:i+1])

		// Target is the next token
		target := []int{tokens[i+1]}

		inputs = append(inputs, input)
		targets = append(targets, target)
	}

	return inputs, targets
}

// Evaluate runs evaluation on a test sequence
func (t *Trainer) Evaluate(testTokens []int) float64 {
	if len(testTokens) < 2 {
		return 0.0
	}

	totalLoss := 0.0
	numPredictions := 0

	// Test on sliding windows
	for i := 0; i < len(testTokens)-1; i++ {
		seqLen := int(math.Min(float64(i+1), 20))
		startIdx := i + 1 - seqLen

		input := testTokens[startIdx : i+1]
		target := []int{testTokens[i+1]} // Single target token for next-token prediction

		loss := t.Model.ComputeLoss(input, target)
		totalLoss += loss
		numPredictions++
	}

	return totalLoss / float64(numPredictions)
}

// LoadTrainingDataFromFile loads training data from a text file
func LoadTrainingDataFromFile(filename string, tokenizer *tokenizer.Tokenizer) ([]int, error) {
	content, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read training file: %v", err)
	}

	text := string(content)

	// Clean and preprocess the text
	text = strings.ReplaceAll(text, "\n\n", " ")
	text = strings.ReplaceAll(text, "\n", " ")
	text = strings.TrimSpace(text)

	// Tokenize the text
	tokens := tokenizer.Tokenize(text)

	fmt.Printf("Loaded %d tokens from %s\n", len(tokens), filename)
	return tokens, nil
}

// TrainFromFile trains the model on text data from a file
func (t *Trainer) TrainFromFile(filename string, tokenizer *tokenizer.Tokenizer, epochs int) error {
	tokens, err := LoadTrainingDataFromFile(filename, tokenizer)
	if err != nil {
		return fmt.Errorf("failed to load training data: %v", err)
	}

	return t.Train(tokens, epochs)
}

// TrainWithSave trains the model and saves it to disk
func (t *Trainer) TrainWithSave(tokens []int, epochs int, modelDir string, tokenizer *tokenizer.Tokenizer) error {
	// First do regular training
	err := t.Train(tokens, epochs)
	if err != nil {
		return err
	}

	// Save the model
	if err := t.Model.SaveModel(modelDir); err != nil {
		return fmt.Errorf("failed to save model: %v", err)
	}

	// Save the tokenizer
	if err := tokenizer.SaveTokenizer(modelDir); err != nil {
		return fmt.Errorf("failed to save tokenizer: %v", err)
	}

	fmt.Printf("Training completed and model saved to %s\n", modelDir)
	return nil
}
