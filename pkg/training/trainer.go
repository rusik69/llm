package training

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"

	"github.com/rusik69/llm/pkg/llm"
	"github.com/rusik69/llm/pkg/tokenizer"
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
}

// DefaultTrainingConfig returns default training configuration
func DefaultTrainingConfig() *TrainingConfig {
	return &TrainingConfig{
		LearningRate:    0.001,
		BatchSize:       16,
		MaxGradNorm:     1.0,
		ValidationSplit: 0.1,
		WarmupSteps:     1000,
		UseWarmup:       false, // Disable warmup for debugging
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

	// Set model to training mode
	t.Model.Train()

	// Configure training parameters for efficiency
	maxSequenceLength := 20
	samplesPerEpoch := min(len(tokens)/10, 10000) // Sample at most 10k examples per epoch
	batchSize := t.Config.BatchSize

	fmt.Printf("Training with %d samples per epoch, batch size %d\n", samplesPerEpoch, batchSize)

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		numBatches := 0
		processed := 0

		// Process in batches for efficiency
		for batchStart := 0; batchStart < samplesPerEpoch; batchStart += batchSize {
			batchEnd := min(batchStart+batchSize, samplesPerEpoch)
			currentBatchSize := batchEnd - batchStart

			var batchInputs [][]int
			var batchTargets [][]int

			// Create batch of training examples
			for i := 0; i < currentBatchSize; i++ {
				// Sample random position from tokens (more efficient than sequential)
				randomPos := rand.Intn(len(tokens) - maxSequenceLength - 1)

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

			// Debug: Check optimizer state
			if numBatches == 1 {
				fmt.Printf("First batch - Step: %d, Config LR: %.6f, Effective LR: %.6f\n",
					t.Optimizer.State.Step, t.Optimizer.Config.LearningRate, t.Optimizer.GetLearningRate())
			}

			// Apply learning rate warmup if enabled
			if t.Config.UseWarmup && t.Optimizer.State.Step < int64(t.Config.WarmupSteps) {
				warmupLR := t.Config.LearningRate * float64(t.Optimizer.State.Step) / float64(t.Config.WarmupSteps)
				t.Optimizer.SetLearningRate(warmupLR)
			}

			// Progress indicator
			if numBatches%10 == 0 || batchStart+batchSize >= samplesPerEpoch {
				progress := float64(processed) / float64(samplesPerEpoch) * 100
				fmt.Printf("\rEpoch %d: %.1f%% complete, Current Loss: %.4f, LR: %.6f",
					epoch+1, progress, batchLoss, t.Optimizer.GetLearningRate())
			}
		}

		avgLoss := totalLoss / float64(numBatches)
		currentLR := t.Optimizer.GetLearningRate()
		fmt.Printf("\nEpoch %d: Average Loss = %.4f, Learning Rate = %.6f\n", epoch+1, avgLoss, currentLR)
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

// TrainBatch performs training on a batch of sequences with proper backpropagation
func (t *Trainer) TrainBatch(inputs [][]int, targets [][]int) float64 {
	if len(inputs) != len(targets) {
		panic("Number of input sequences must match number of target sequences")
	}

	totalLoss := 0.0

	// Set model to training mode
	t.Model.Train()

	// Process each example in the batch
	for i := 0; i < len(inputs); i++ {
		// Debug: Check training data
		if i == 0 {
			fmt.Printf("Sample input: %v, target: %v\n", inputs[i], targets[i])
		}

		// Compute loss
		loss := t.Model.ComputeLoss(inputs[i], targets[i])
		totalLoss += loss

		// Debug: Check loss values
		if i == 0 {
			fmt.Printf("Sample loss: %.6f\n", loss)
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
