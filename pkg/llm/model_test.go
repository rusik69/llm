package llm

import (
	"testing"

	"github.com/rusik69/shittyllm/pkg/matrix"
)

func TestNewLLM(t *testing.T) {
	vocabSize := 100
	embedSize := 32
	hiddenSize := 64
	numLayers := 2

	model := New(vocabSize, embedSize, hiddenSize, numLayers)

	if model.VocabSize != vocabSize {
		t.Errorf("Expected vocab size %d, got %d", vocabSize, model.VocabSize)
	}
	if model.EmbedSize != embedSize {
		t.Errorf("Expected embed size %d, got %d", embedSize, model.EmbedSize)
	}
	if model.HiddenSize != hiddenSize {
		t.Errorf("Expected hidden size %d, got %d", hiddenSize, model.HiddenSize)
	}
	if model.NumLayers != numLayers {
		t.Errorf("Expected %d layers, got %d", numLayers, model.NumLayers)
	}

	// Check that components are initialized
	if model.Embedding == nil {
		t.Error("Embedding layer should be initialized")
	}
	if model.PositionalEnc == nil {
		t.Error("Positional encoding should be initialized")
	}
	if model.OutputLayer == nil {
		t.Error("Output layer should be initialized")
	}
	if len(model.TransformerBlocks) != numLayers {
		t.Errorf("Expected %d transformer blocks, got %d", numLayers, len(model.TransformerBlocks))
	}
}

func TestLLMForward(t *testing.T) {
	model := New(50, 16, 32, 1)

	tokens := []int{1, 2, 3, 4}
	logits := model.Forward(tokens)

	// Check output dimensions
	if logits.Rows != len(tokens) {
		t.Errorf("Expected %d rows in output, got %d", len(tokens), logits.Rows)
	}
	if logits.Cols != model.VocabSize {
		t.Errorf("Expected %d columns in output, got %d", model.VocabSize, logits.Cols)
	}

	// Check that logits are not all zero
	nonZeroCount := 0
	for i := 0; i < logits.Rows; i++ {
		for j := 0; j < logits.Cols; j++ {
			if logits.Get(i, j) != 0 {
				nonZeroCount++
			}
		}
	}

	if nonZeroCount == 0 {
		t.Error("All logits are zero - model may not be working correctly")
	}
}

func TestLLMPredict(t *testing.T) {
	model := New(50, 16, 32, 1)

	tokens := []int{1, 2, 3}
	probs := model.Predict(tokens)

	// Check output dimensions
	if len(probs) != model.VocabSize {
		t.Errorf("Expected %d probabilities, got %d", model.VocabSize, len(probs))
	}

	// Check that probabilities sum to approximately 1
	sum := 0.0
	for _, prob := range probs {
		sum += prob
		if prob < 0 {
			t.Errorf("Probability should be non-negative, got %f", prob)
		}
	}

	if sum < 0.99 || sum > 1.01 {
		t.Errorf("Probabilities should sum to 1, got %f", sum)
	}
}

func TestLLMGenerate(t *testing.T) {
	model := New(50, 16, 32, 1)

	prompt := []int{1, 2, 3}
	generated := model.Generate(prompt, 5)

	// Check that output contains original prompt
	if len(generated) < len(prompt) {
		t.Errorf("Generated sequence should contain original prompt")
	}

	for i := 0; i < len(prompt); i++ {
		if generated[i] != prompt[i] {
			t.Errorf("Generated sequence should start with prompt")
		}
	}

	// Check that new tokens were added
	if len(generated) <= len(prompt) {
		t.Errorf("Expected generation to add new tokens")
	}
}

func TestLLMComputeLoss(t *testing.T) {
	model := New(50, 16, 32, 1)

	input := []int{1, 2, 3}
	target := []int{4} // Single target token for next-token prediction

	loss := model.ComputeLoss(input, target)

	// Loss should be positive
	if loss <= 0 {
		t.Errorf("Loss should be positive, got %f", loss)
	}

	// Loss should be finite
	if loss != loss { // NaN check
		t.Errorf("Loss should not be NaN")
	}
}

func TestNewEmbeddingLayer(t *testing.T) {
	vocabSize := 100
	embedSize := 32

	layer := NewEmbeddingLayer(vocabSize, embedSize)

	if layer.VocabSize != vocabSize {
		t.Errorf("Expected vocab size %d, got %d", vocabSize, layer.VocabSize)
	}
	if layer.EmbedSize != embedSize {
		t.Errorf("Expected embed size %d, got %d", embedSize, layer.EmbedSize)
	}

	// Check weight matrix dimensions
	if layer.Weights.Rows != vocabSize {
		t.Errorf("Expected %d rows in weight matrix, got %d", vocabSize, layer.Weights.Rows)
	}
	if layer.Weights.Cols != embedSize {
		t.Errorf("Expected %d columns in weight matrix, got %d", embedSize, layer.Weights.Cols)
	}
}

func TestEmbeddingLayerForward(t *testing.T) {
	layer := NewEmbeddingLayer(50, 16)

	tokens := []int{1, 2, 3, 4}
	embeddings := layer.Forward(tokens)

	// Check output dimensions
	if embeddings.Rows != len(tokens) {
		t.Errorf("Expected %d rows in embeddings, got %d", len(tokens), embeddings.Rows)
	}
	if embeddings.Cols != layer.EmbedSize {
		t.Errorf("Expected %d columns in embeddings, got %d", layer.EmbedSize, embeddings.Cols)
	}
}

func TestNewAttentionLayer(t *testing.T) {
	embedSize := 64
	numHeads := 4

	layer := NewAttentionLayer(embedSize, numHeads)

	if layer.EmbedSize != embedSize {
		t.Errorf("Expected embed size %d, got %d", embedSize, layer.EmbedSize)
	}
	if layer.NumHeads != numHeads {
		t.Errorf("Expected %d heads, got %d", numHeads, layer.NumHeads)
	}
	if layer.HeadSize != embedSize/numHeads {
		t.Errorf("Expected head size %d, got %d", embedSize/numHeads, layer.HeadSize)
	}
}

func TestAttentionLayerForward(t *testing.T) {
	layer := NewAttentionLayer(32, 4)

	// Create input matrix (sequence_length x embed_size)
	input := matrix.New(5, 32)
	input.Random()

	output := layer.Forward(input)

	// Check output dimensions
	if output.Rows != input.Rows {
		t.Errorf("Expected %d rows in output, got %d", input.Rows, output.Rows)
	}
	if output.Cols != input.Cols {
		t.Errorf("Expected %d columns in output, got %d", input.Cols, output.Cols)
	}
}

func TestNewFeedForwardLayer(t *testing.T) {
	inputSize := 32
	hiddenSize := 64

	layer := NewFeedForwardLayer(inputSize, hiddenSize)

	if layer.InputSize != inputSize {
		t.Errorf("Expected input size %d, got %d", inputSize, layer.InputSize)
	}
	if layer.HiddenSize != hiddenSize {
		t.Errorf("Expected hidden size %d, got %d", hiddenSize, layer.HiddenSize)
	}
}

func TestFeedForwardLayerForward(t *testing.T) {
	layer := NewFeedForwardLayer(32, 64)

	input := matrix.New(5, 32)
	input.Random()

	output := layer.Forward(input)

	// Check output dimensions
	if output.Rows != input.Rows {
		t.Errorf("Expected %d rows in output, got %d", input.Rows, output.Rows)
	}
	if output.Cols != input.Cols {
		t.Errorf("Expected %d columns in output, got %d", input.Cols, output.Cols)
	}
}

func TestNewTransformerBlock(t *testing.T) {
	embedSize := 32
	numHeads := 4
	hiddenSize := 64

	block := NewTransformerBlock(embedSize, numHeads, hiddenSize)

	if block.EmbedSize != embedSize {
		t.Errorf("Expected embed size %d, got %d", embedSize, block.EmbedSize)
	}
	if block.Attention == nil {
		t.Error("Attention layer should be initialized")
	}
	if block.FeedForward == nil {
		t.Error("Feed forward layer should be initialized")
	}
}

func TestTransformerBlockForward(t *testing.T) {
	block := NewTransformerBlock(32, 4, 64)

	input := matrix.New(5, 32)
	input.Random()

	output := block.Forward(input)

	// Check output dimensions
	if output.Rows != input.Rows {
		t.Errorf("Expected %d rows in output, got %d", input.Rows, output.Rows)
	}
	if output.Cols != input.Cols {
		t.Errorf("Expected %d columns in output, got %d", input.Cols, output.Cols)
	}
}

func TestNewPositionalEncoding(t *testing.T) {
	maxSeqLen := 100
	embedSize := 32

	pe := NewPositionalEncoding(maxSeqLen, embedSize)

	if pe.MaxSeqLen != maxSeqLen {
		t.Errorf("Expected max seq len %d, got %d", maxSeqLen, pe.MaxSeqLen)
	}
	if pe.EmbedSize != embedSize {
		t.Errorf("Expected embed size %d, got %d", embedSize, pe.EmbedSize)
	}

	// Check encoding matrix dimensions
	if pe.Encoding.Rows != maxSeqLen {
		t.Errorf("Expected %d rows in encoding matrix, got %d", maxSeqLen, pe.Encoding.Rows)
	}
	if pe.Encoding.Cols != embedSize {
		t.Errorf("Expected %d columns in encoding matrix, got %d", embedSize, pe.Encoding.Cols)
	}
}

func TestPositionalEncodingForward(t *testing.T) {
	pe := NewPositionalEncoding(100, 32)

	input := matrix.New(5, 32)
	input.Random()

	output := pe.Forward(input)

	// Check output dimensions
	if output.Rows != input.Rows {
		t.Errorf("Expected %d rows in output, got %d", input.Rows, output.Rows)
	}
	if output.Cols != input.Cols {
		t.Errorf("Expected %d columns in output, got %d", input.Cols, output.Cols)
	}

	// Check that output is different from input (due to positional encoding)
	different := false
	for i := 0; i < input.Rows; i++ {
		for j := 0; j < input.Cols; j++ {
			if input.Get(i, j) != output.Get(i, j) {
				different = true
				break
			}
		}
		if different {
			break
		}
	}

	if !different {
		t.Error("Positional encoding should modify the input")
	}
}

func TestSampleToken(t *testing.T) {
	model := New(10, 8, 16, 1)

	probs := []float64{0.1, 0.2, 0.3, 0.4}
	token := model.sampleToken(probs)

	// Should return the index of the maximum probability (index 3)
	if token != 3 {
		t.Errorf("Expected token 3 (highest probability), got %d", token)
	}
}

func TestSampleTokenEdgeCases(t *testing.T) {
	model := New(10, 8, 16, 1)

	// Test with single probability
	probs := []float64{1.0}
	token := model.sampleToken(probs)
	if token != 0 {
		t.Errorf("Expected token 0 for single probability, got %d", token)
	}

	// Test with equal probabilities
	probs = []float64{0.5, 0.5}
	token = model.sampleToken(probs)
	if token != 0 && token != 1 {
		t.Errorf("Expected token 0 or 1 for equal probabilities, got %d", token)
	}
}
