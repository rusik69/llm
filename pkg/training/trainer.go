package training

import (
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/rusik69/llm/pkg/llm"
	"github.com/rusik69/llm/pkg/tokenizer"
)

// Trainer handles model training
type Trainer struct {
	Model        *llm.LLM
	LearningRate float64
}

// New creates a new trainer
func New(model *llm.LLM, learningRate float64) *Trainer {
	return &Trainer{
		Model:        model,
		LearningRate: learningRate,
	}
}

// Train performs training on the given sequence
func (t *Trainer) Train(tokens []int, epochs int) error {
	if len(tokens) < 2 {
		return fmt.Errorf("training sequence must have at least 2 tokens")
	}

	fmt.Printf("Starting training with %d tokens for %d epochs\n", len(tokens), epochs)

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		numBatches := 0

		// Create sliding window training examples
		for i := 0; i < len(tokens)-1; i++ {
			// Prepare input and target sequences
			seqLen := int(math.Min(float64(i+1), 20)) // Max sequence length of 20
			startIdx := i + 1 - seqLen

			input := tokens[startIdx : i+1]
			target := tokens[startIdx+1 : i+2]

			// Compute loss
			loss := t.Model.ComputeLoss(input, target)
			totalLoss += loss
			numBatches++

			// Simple gradient descent (very basic implementation)
			t.updateWeights(input, target, loss)
		}

		avgLoss := totalLoss / float64(numBatches)
		fmt.Printf("Epoch %d: Average Loss = %.4f\n", epoch+1, avgLoss)
	}

	fmt.Println("Training completed!")
	return nil
}

// updateWeights performs improved weight updates with gradient clipping
func (t *Trainer) updateWeights(input []int, target []int, loss float64) {
	// Clip gradient to prevent exploding gradients
	maxGrad := 1.0
	if loss > maxGrad {
		loss = maxGrad
	}

	// Simple gradient approximation for educational purposes
	// In a real implementation, you would compute actual gradients through backpropagation

	// Adjust learning rate based on loss to prevent divergence
	adaptiveLR := t.LearningRate
	if loss > 10.0 {
		adaptiveLR = t.LearningRate * 0.1 // Reduce learning rate if loss is too high
	} else if loss > 5.0 {
		adaptiveLR = t.LearningRate * 0.5
	}

	// Extremely simple weight update (for educational purposes)
	// This is not real backpropagation but shows the concept

	// Small random perturbation in the direction that should reduce loss
	perturbation := adaptiveLR * 0.001 // Very small updates

	// Update a small subset of embedding weights
	vocabSize := t.Model.VocabSize
	embedSize := t.Model.EmbedSize

	// Update weights for the tokens in the input
	for _, tokenID := range input {
		if tokenID >= 0 && tokenID < vocabSize {
			for j := 0; j < embedSize; j++ {
				current := t.Model.Embedding.Weights.Get(tokenID, j)
				// Small random update weighted by loss
				// Simple pseudo-random based on token and position
				randVal := float64((tokenID+j*17)%100)/100.0 - 0.5 // Random between -0.5 and 0.5
				update := randVal * perturbation
				newValue := current + update

				// Clamp weights to prevent them from growing too large
				if newValue > 1.0 {
					newValue = 1.0
				} else if newValue < -1.0 {
					newValue = -1.0
				}

				t.Model.Embedding.Weights.Set(tokenID, j, newValue)
			}
		}
	}

	// Update target token weights slightly
	for _, tokenID := range target {
		if tokenID >= 0 && tokenID < vocabSize {
			for j := 0; j < embedSize; j++ {
				current := t.Model.Embedding.Weights.Get(tokenID, j)
				// Encourage target tokens slightly
				update := perturbation * 0.1
				newValue := current + update

				// Clamp weights
				if newValue > 1.0 {
					newValue = 1.0
				} else if newValue < -1.0 {
					newValue = -1.0
				}

				t.Model.Embedding.Weights.Set(tokenID, j, newValue)
			}
		}
	}
}

// TrainOnBatch trains on a batch of sequences
func (t *Trainer) TrainOnBatch(sequences [][]int, targets [][]int) float64 {
	if len(sequences) != len(targets) {
		panic("Number of sequences must match number of targets")
	}

	totalLoss := 0.0

	for i := 0; i < len(sequences); i++ {
		if len(sequences[i]) != len(targets[i]) {
			continue // Skip invalid sequences
		}

		loss := t.Model.ComputeLoss(sequences[i], targets[i])
		totalLoss += loss

		// Update weights
		t.updateWeights(sequences[i], targets[i], loss)
	}

	return totalLoss / float64(len(sequences))
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
		target := testTokens[startIdx+1 : i+2]

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
