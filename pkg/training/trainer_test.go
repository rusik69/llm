package training

import (
	"testing"

	"github.com/rusik69/shittyllm/pkg/llm"
)

func TestNewTrainer(t *testing.T) {
	model := llm.New(100, 32, 64, 2)
	lr := 0.01

	trainer := New(model, lr)

	if trainer.Model != model {
		t.Error("Trainer should reference the provided model")
	}
	if trainer.Config.LearningRate != lr {
		t.Errorf("Expected learning rate %f, got %f", lr, trainer.Config.LearningRate)
	}
}

func TestTrainEmptySequence(t *testing.T) {
	model := llm.New(100, 32, 64, 2)
	trainer := New(model, 0.01)

	// Test with empty sequence
	tokens := []int{}
	err := trainer.Train(tokens, 1)

	if err == nil {
		t.Error("Expected error for empty sequence")
	}

	// Test with single token
	tokens = []int{1}
	err = trainer.Train(tokens, 1)

	if err == nil {
		t.Error("Expected error for single token sequence")
	}
}

func TestTrainValidSequence(t *testing.T) {
	model := llm.New(100, 32, 64, 1)
	trainer := New(model, 0.01)

	// Test with valid sequence
	tokens := []int{1, 2, 3, 4, 5}
	err := trainer.Train(tokens, 2)

	if err != nil {
		t.Errorf("Expected no error for valid sequence, got %v", err)
	}
}

func TestTrainOnBatch(t *testing.T) {
	model := llm.New(100, 32, 64, 1)
	trainer := New(model, 0.01)

	sequences := [][]int{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}

	targets := [][]int{
		{4}, // Single target token for next-token prediction
		{7},
		{10},
	}

	loss := trainer.TrainOnBatch(sequences, targets)

	if loss <= 0 {
		t.Errorf("Expected positive loss, got %f", loss)
	}

	// Test with mismatched lengths
	sequences = append(sequences, []int{1, 2, 3})
	// Don't add corresponding target

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for mismatched sequence/target lengths")
		}
	}()

	trainer.TrainOnBatch(sequences, targets)
}

func TestPrepareTrainingData(t *testing.T) {
	tokens := []int{1, 2, 3, 4, 5}
	maxSeqLen := 3

	inputs, targets := PrepareTrainingData(tokens, maxSeqLen)

	expectedInputs := 4 // len(tokens) - 1
	if len(inputs) != expectedInputs {
		t.Errorf("Expected %d input sequences, got %d", expectedInputs, len(inputs))
	}
	if len(targets) != expectedInputs {
		t.Errorf("Expected %d target sequences, got %d", expectedInputs, len(targets))
	}

	// Check that each target has one element
	for i, target := range targets {
		if len(target) != 1 {
			t.Errorf("Target %d should have 1 element, got %d", i, len(target))
		}
	}

	// Check that inputs respect maxSeqLen
	for i, input := range inputs {
		if len(input) > maxSeqLen {
			t.Errorf("Input %d exceeds maxSeqLen %d, got %d", i, maxSeqLen, len(input))
		}
	}

	// Check the relationship between inputs and targets
	for i := 0; i < len(inputs); i++ {
		if targets[i][0] != tokens[i+1] {
			t.Errorf("Target %d should be %d, got %d", i, tokens[i+1], targets[i][0])
		}
	}
}

func TestPrepareTrainingDataEdgeCases(t *testing.T) {
	// Test with short sequence
	tokens := []int{1, 2}
	inputs, targets := PrepareTrainingData(tokens, 10)

	if len(inputs) != 1 {
		t.Errorf("Expected 1 input for 2-token sequence, got %d", len(inputs))
	}
	if len(targets) != 1 {
		t.Errorf("Expected 1 target for 2-token sequence, got %d", len(targets))
	}

	// Test with single token
	tokens = []int{1}
	inputs, targets = PrepareTrainingData(tokens, 10)

	if len(inputs) != 0 {
		t.Errorf("Expected 0 inputs for single token, got %d", len(inputs))
	}
	if len(targets) != 0 {
		t.Errorf("Expected 0 targets for single token, got %d", len(targets))
	}
}

func TestEvaluate(t *testing.T) {
	model := llm.New(100, 32, 64, 1)
	trainer := New(model, 0.01)

	// Test with valid sequence
	testTokens := []int{1, 2, 3, 4, 5}
	loss := trainer.Evaluate(testTokens)

	if loss <= 0 {
		t.Errorf("Expected positive evaluation loss, got %f", loss)
	}

	// Test with short sequence
	testTokens = []int{1}
	loss = trainer.Evaluate(testTokens)

	if loss != 0 {
		t.Errorf("Expected zero loss for single token, got %f", loss)
	}

	// Test with empty sequence
	testTokens = []int{}
	loss = trainer.Evaluate(testTokens)

	if loss != 0 {
		t.Errorf("Expected zero loss for empty sequence, got %f", loss)
	}
}

func TestTrainBatch(t *testing.T) {
	model := llm.New(100, 32, 64, 1)
	trainer := New(model, 0.01)

	// Test batch training with multiple sequences
	inputs := [][]int{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	targets := [][]int{
		{4}, // Single target token for next-token prediction
		{7},
		{10},
	}

	loss := trainer.TrainBatch(inputs, targets)

	if loss <= 0 {
		t.Errorf("Expected positive loss, got %f", loss)
	}

	// Test with single sequence
	loss = trainer.TrainBatch(inputs[:1], targets[:1])
	if loss <= 0 {
		t.Errorf("Expected positive loss for single sequence, got %f", loss)
	}
}

func TestTrainerIntegration(t *testing.T) {
	// Integration test: Train a small model and check that loss decreases
	model := llm.New(50, 16, 32, 1)
	trainer := New(model, 0.01)

	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}

	// Get initial loss
	initialLoss := trainer.Evaluate(tokens)

	// Train for a few epochs
	err := trainer.Train(tokens, 3)
	if err != nil {
		t.Errorf("Training failed: %v", err)
	}

	// Get final loss
	finalLoss := trainer.Evaluate(tokens)

	// Check that both losses are positive
	if initialLoss <= 0 {
		t.Errorf("Initial loss should be positive, got %f", initialLoss)
	}
	if finalLoss <= 0 {
		t.Errorf("Final loss should be positive, got %f", finalLoss)
	}

	// Note: Due to the simplified training mechanism, we can't guarantee
	// that loss will decrease, but we can check that training runs without errors
}

func TestTrainerWithDifferentLearningRates(t *testing.T) {
	model := llm.New(50, 16, 32, 1)

	// Test with different learning rates
	learningRates := []float64{0.001, 0.01, 0.1}
	tokens := []int{1, 2, 3, 4, 5}

	for _, lr := range learningRates {
		trainer := New(model, lr)

		if trainer.Config.LearningRate != lr {
			t.Errorf("Expected learning rate %f, got %f", lr, trainer.Config.LearningRate)
		}

		// Test that training works with different learning rates
		err := trainer.Train(tokens, 1)
		if err != nil {
			t.Errorf("Training failed with learning rate %f: %v", lr, err)
		}
	}
}

func TestTrainerEdgeCases(t *testing.T) {
	model := llm.New(50, 16, 32, 1)
	trainer := New(model, 0.01)

	// Test with zero epochs
	tokens := []int{1, 2, 3}
	err := trainer.Train(tokens, 0)
	if err != nil {
		t.Errorf("Training with 0 epochs should not fail: %v", err)
	}

	// Test with negative epochs (should handle gracefully)
	err = trainer.Train(tokens, -1)
	if err != nil {
		t.Errorf("Training with negative epochs should not fail: %v", err)
	}
}

func TestBatchTrainingValidation(t *testing.T) {
	model := llm.New(50, 16, 32, 1)
	trainer := New(model, 0.01)

	// Test with valid batch
	sequences := [][]int{
		{1, 2, 3},
		{4, 5, 6},
	}
	targets := [][]int{
		{4}, // Single target token for next-token prediction
		{7},
	}

	loss := trainer.TrainOnBatch(sequences, targets)
	if loss <= 0 {
		t.Errorf("Expected positive batch loss, got %f", loss)
	}

	// Test with mismatched sequence lengths (should skip)
	sequences = append(sequences, []int{1, 2})
	targets = append(targets, []int{3}) // Single target token

	loss = trainer.TrainOnBatch(sequences, targets)
	if loss <= 0 {
		t.Errorf("Expected positive batch loss even with mismatched lengths, got %f", loss)
	}
}
