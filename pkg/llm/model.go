package llm

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/rusik69/shittyllm/pkg/matrix"
)

// LLM represents the complete language model
type LLM struct {
	VocabSize  int
	EmbedSize  int
	HiddenSize int
	NumLayers  int
	MaxSeqLen  int
	IsTraining bool

	// Model components
	Embedding         *EmbeddingLayer
	PositionalEnc     *PositionalEncoding
	TransformerBlocks []*TransformerBlock
	OutputLayer       *matrix.Matrix      // Linear layer for final predictions
	FinalLayerNorm    *LayerNormalization // Final layer normalization
}

// New creates a new LLM with the specified architecture
func New(vocabSize, embedSize, hiddenSize, numLayers int) *LLM {
	maxSeqLen := 512 // Default maximum sequence length

	model := &LLM{
		VocabSize:  vocabSize,
		EmbedSize:  embedSize,
		HiddenSize: hiddenSize,
		NumLayers:  numLayers,
		MaxSeqLen:  maxSeqLen,
		IsTraining: false, // Default to inference mode

		Embedding:      NewEmbeddingLayer(vocabSize, embedSize),
		PositionalEnc:  NewPositionalEncoding(maxSeqLen, embedSize),
		OutputLayer:    matrix.New(embedSize, vocabSize),
		FinalLayerNorm: NewLayerNormalization(embedSize),
	}

	// Initialize transformer blocks
	numHeads := 4 // Fixed number of attention heads
	model.TransformerBlocks = make([]*TransformerBlock, numLayers)
	for i := 0; i < numLayers; i++ {
		model.TransformerBlocks[i] = NewTransformerBlock(embedSize, numHeads, hiddenSize)
	}

	// Initialize output layer
	model.OutputLayer.Xavier()

	return model
}

// Forward performs a forward pass through the model
func (llm *LLM) Forward(tokenIDs []int) *matrix.Matrix {
	// Convert tokens to embeddings
	embeddings := llm.Embedding.Forward(tokenIDs)

	// Add positional encoding
	x := llm.PositionalEnc.Forward(embeddings)

	// Pass through transformer blocks
	for _, block := range llm.TransformerBlocks {
		x = block.Forward(x)
	}

	// Apply final layer normalization
	x = llm.FinalLayerNorm.Forward(x)

	// Apply output layer to get logits
	logits := x.Multiply(llm.OutputLayer)

	return logits
}

// SetTraining sets the model to training or inference mode
func (llm *LLM) SetTraining(training bool) {
	llm.IsTraining = training
	for _, block := range llm.TransformerBlocks {
		block.SetTraining(training)
	}
}

// Train sets the model to training mode
func (llm *LLM) Train() {
	llm.SetTraining(true)
}

// Eval sets the model to evaluation mode
func (llm *LLM) Eval() {
	llm.SetTraining(false)
}

// Predict gets the probability distribution for the next token
func (llm *LLM) Predict(tokenIDs []int) []float64 {
	logits := llm.Forward(tokenIDs)

	// Get the last time step's predictions
	lastStep := logits.Rows - 1
	probs := make([]float64, llm.VocabSize)

	// Extract the last row and apply softmax
	lastLogits := matrix.New(1, llm.VocabSize)
	for j := 0; j < llm.VocabSize; j++ {
		lastLogits.Set(0, j, logits.Get(lastStep, j))
	}

	softmaxProbs := lastLogits.Softmax()
	for j := 0; j < llm.VocabSize; j++ {
		probs[j] = softmaxProbs.Get(0, j)
	}

	return probs
}

// GenerationConfig controls text generation behavior
type GenerationConfig struct {
	MaxNewTokens      int
	Temperature       float64
	TopK              int
	TopP              float64
	DoSample          bool
	RepetitionPenalty float64 // New: penalty for repeating tokens
	NoRepeatNGramSize int     // New: prevent repeating n-grams
}

// DefaultGenerationConfig returns sensible defaults with improved diversity
func DefaultGenerationConfig() *GenerationConfig {
	return &GenerationConfig{
		MaxNewTokens:      50,
		Temperature:       1.0,  // Increased for more creativity
		TopK:              40,   // Reduced for more focused sampling
		TopP:              0.85, // Slightly reduced for better quality
		DoSample:          true,
		RepetitionPenalty: 1.5, // Stronger penalty for repetition
		NoRepeatNGramSize: 4,   // Prevent longer repetitive patterns
	}
}

// Generate produces new tokens autoregressively with advanced sampling
func (llm *LLM) Generate(promptTokens []int, maxNewTokens int) []int {
	config := DefaultGenerationConfig()
	config.MaxNewTokens = maxNewTokens
	return llm.GenerateWithConfig(promptTokens, config)
}

// GenerateWithConfig produces new tokens with custom generation config
func (llm *LLM) GenerateWithConfig(promptTokens []int, config *GenerationConfig) []int {
	generated := make([]int, len(promptTokens))
	copy(generated, promptTokens)

	// Track generated tokens for repetition penalty
	tokenCounts := make(map[int]int)

	// Count tokens in the prompt
	for _, token := range promptTokens {
		tokenCounts[token]++
	}

	for i := 0; i < config.MaxNewTokens; i++ {
		// Get probabilities for next token
		probs := llm.Predict(generated)

		// Apply repetition penalty
		if config.RepetitionPenalty > 0 {
			probs = llm.applyRepetitionPenalty(probs, tokenCounts, config.RepetitionPenalty)
		}

		// Apply n-gram repetition filtering
		if config.NoRepeatNGramSize > 0 {
			probs = llm.applyNGramFiltering(probs, generated, config.NoRepeatNGramSize)
		}

		var nextToken int
		if config.DoSample {
			// Apply temperature, top-k, and nucleus sampling
			nextToken = llm.sampleTokenAdvanced(probs, config)
		} else {
			// Use greedy decoding
			nextToken = llm.sampleToken(probs)
		}

		// Add to sequence
		generated = append(generated, nextToken)
		tokenCounts[nextToken]++

		// Early stopping for repetitive patterns
		if llm.isRepetitiveSequence(generated, 4) {
			break
		}

		// Stop if we exceed max sequence length
		if len(generated) >= llm.MaxSeqLen {
			break
		}

		// Simple check for end token (you might want to make this configurable)
		if nextToken == 3 { // Assuming EOS token is ID 3
			break
		}
	}

	return generated
}

// sampleToken samples a token from probability distribution (greedy decoding)
func (llm *LLM) sampleToken(probs []float64) int {
	// Implement simple greedy decoding (argmax)
	maxProb := probs[0]
	maxIdx := 0

	for i, prob := range probs {
		if prob > maxProb {
			maxProb = prob
			maxIdx = i
		}
	}

	return maxIdx
}

// sampleTokenAdvanced implements advanced sampling strategies
func (llm *LLM) sampleTokenAdvanced(probs []float64, config *GenerationConfig) int {
	// Make a copy to avoid modifying original
	samplingProbs := make([]float64, len(probs))
	copy(samplingProbs, probs)

	// Apply temperature scaling
	if config.Temperature != 1.0 && config.Temperature > 0 {
		for i := range samplingProbs {
			samplingProbs[i] = math.Pow(samplingProbs[i], 1.0/config.Temperature)
		}
		// Renormalize
		samplingProbs = normalize(samplingProbs)
	}

	// Apply top-k filtering
	if config.TopK > 0 && config.TopK < len(samplingProbs) {
		samplingProbs = applyTopK(samplingProbs, config.TopK)
	}

	// Apply nucleus (top-p) sampling
	if config.TopP > 0 && config.TopP < 1.0 {
		samplingProbs = applyTopP(samplingProbs, config.TopP)
	}

	// Sample from the filtered distribution
	return sampleFromDistribution(samplingProbs)
}

// normalize ensures probabilities sum to 1
func normalize(probs []float64) []float64 {
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if sum == 0 {
		// Uniform distribution if all probabilities are 0
		for i := range probs {
			probs[i] = 1.0 / float64(len(probs))
		}
		return probs
	}

	normalized := make([]float64, len(probs))
	for i, p := range probs {
		normalized[i] = p / sum
	}
	return normalized
}

// applyTopK keeps only the top-k highest probability tokens
func applyTopK(probs []float64, k int) []float64 {
	if k >= len(probs) {
		return probs
	}

	// Create pairs of (probability, index)
	type probIndex struct {
		prob  float64
		index int
	}

	pairs := make([]probIndex, len(probs))
	for i, p := range probs {
		pairs[i] = probIndex{p, i}
	}

	// Sort by probability (descending)
	for i := 0; i < len(pairs)-1; i++ {
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].prob > pairs[i].prob {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}

	// Zero out probabilities outside top-k
	result := make([]float64, len(probs))
	for i := 0; i < k; i++ {
		result[pairs[i].index] = pairs[i].prob
	}

	return normalize(result)
}

// applyTopP keeps tokens with cumulative probability <= p
func applyTopP(probs []float64, p float64) []float64 {
	// Create and sort probability indices
	type probIndex struct {
		prob  float64
		index int
	}

	pairs := make([]probIndex, len(probs))
	for i, prob := range probs {
		pairs[i] = probIndex{prob, i}
	}

	// Sort by probability (descending)
	for i := 0; i < len(pairs)-1; i++ {
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].prob > pairs[i].prob {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}

	// Find cutoff point
	cumSum := 0.0
	cutoff := len(pairs)
	for i, pair := range pairs {
		cumSum += pair.prob
		if cumSum >= p {
			cutoff = i + 1
			break
		}
	}

	// Zero out probabilities outside nucleus
	result := make([]float64, len(probs))
	for i := 0; i < cutoff; i++ {
		result[pairs[i].index] = pairs[i].prob
	}

	return normalize(result)
}

// sampleFromDistribution samples from a probability distribution with proper randomness
func sampleFromDistribution(probs []float64) int {
	// Normalize to ensure sum = 1
	probs = normalize(probs)

	// Use a simple pseudo-random approach based on cumulative distribution
	// This is better than the previous deterministic method

	// Calculate cumulative probabilities
	cumProbs := make([]float64, len(probs))
	cumProbs[0] = probs[0]
	for i := 1; i < len(probs); i++ {
		cumProbs[i] = cumProbs[i-1] + probs[i]
	}

	// Generate a pseudo-random number between 0 and 1
	// Using a simple linear congruential generator seeded with the hash of probabilities
	seed := uint64(0)
	for i, p := range probs {
		seed += uint64(p*1000000) + uint64(i*17)
	}

	// Simple LCG: (a * seed + c) % m
	// Using values from Numerical Recipes
	a := uint64(1664525)
	c := uint64(1013904223)
	m := uint64(1 << 32)

	seed = (a*seed + c) % m
	r := float64(seed) / float64(m)

	// Find the token corresponding to this random value
	for i, cumProb := range cumProbs {
		if r <= cumProb {
			return i
		}
	}

	// Fallback: return the last token
	return len(probs) - 1
}

// ComputeLoss calculates the cross-entropy loss
func (llm *LLM) ComputeLoss(tokenIDs []int, targets []int) float64 {
	// For next-token prediction, we predict the next token for each position
	// Input: [a, b, c] -> Targets: [b, c, d]
	// But we're getting input sequence and single target, so we need to adjust

	if len(targets) != 1 {
		panic("Expected single target token for next-token prediction")
	}

	if len(tokenIDs) == 0 {
		return 0.0
	}

	// Forward pass to get logits
	logits := llm.Forward(tokenIDs)

	// Get prediction for the last time step (this predicts the next token)
	lastStep := logits.Rows - 1
	target := targets[0]

	if target < 0 || target >= llm.VocabSize {
		return 0.0
	}

	// Get probabilities for the last time step
	timestepLogits := matrix.New(1, llm.VocabSize)
	for j := 0; j < llm.VocabSize; j++ {
		timestepLogits.Set(0, j, logits.Get(lastStep, j))
	}

	probs := timestepLogits.Softmax()
	targetProb := probs.Get(0, target)

	// Cross-entropy loss
	loss := -math.Log(targetProb + 1e-10)
	return loss
}

// ModelCheckpoint represents a saved model state
type ModelCheckpoint struct {
	VocabSize  int                  `json:"vocab_size"`
	EmbedSize  int                  `json:"embed_size"`
	HiddenSize int                  `json:"hidden_size"`
	NumLayers  int                  `json:"num_layers"`
	MaxSeqLen  int                  `json:"max_seq_len"`
	Weights    map[string][]float64 `json:"weights"`
}

// Enhanced checkpoint with layer normalization parameters
type ModelCheckpointV2 struct {
	VocabSize  int                  `json:"vocab_size"`
	EmbedSize  int                  `json:"embed_size"`
	HiddenSize int                  `json:"hidden_size"`
	NumLayers  int                  `json:"num_layers"`
	MaxSeqLen  int                  `json:"max_seq_len"`
	IsTraining bool                 `json:"is_training"`
	Weights    map[string][]float64 `json:"weights"`
	Version    int                  `json:"version"`
}

// SaveModel saves the model to disk
func (llm *LLM) SaveModel(modelDir string) error {
	// Create model directory
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return fmt.Errorf("failed to create model directory: %v", err)
	}

	checkpoint := &ModelCheckpoint{
		VocabSize:  llm.VocabSize,
		EmbedSize:  llm.EmbedSize,
		HiddenSize: llm.HiddenSize,
		NumLayers:  llm.NumLayers,
		MaxSeqLen:  llm.MaxSeqLen,
		Weights:    make(map[string][]float64),
	}

	// Save embedding weights
	checkpoint.Weights["embedding"] = matrixToSlice(llm.Embedding.Weights)

	// Save positional encoding
	checkpoint.Weights["positional_encoding"] = matrixToSlice(llm.PositionalEnc.Encoding)

	// Save transformer block weights
	for i, block := range llm.TransformerBlocks {
		prefix := fmt.Sprintf("transformer_%d", i)

		// Attention weights
		checkpoint.Weights[prefix+"_attn_query"] = matrixToSlice(block.Attention.WQuery)
		checkpoint.Weights[prefix+"_attn_key"] = matrixToSlice(block.Attention.WKey)
		checkpoint.Weights[prefix+"_attn_value"] = matrixToSlice(block.Attention.WValue)
		checkpoint.Weights[prefix+"_attn_output"] = matrixToSlice(block.Attention.WOutput)

		// Feed-forward weights
		checkpoint.Weights[prefix+"_ff_w1"] = matrixToSlice(block.FeedForward.W1)
		checkpoint.Weights[prefix+"_ff_w2"] = matrixToSlice(block.FeedForward.W2)
		checkpoint.Weights[prefix+"_ff_b1"] = matrixToSlice(block.FeedForward.B1)
		checkpoint.Weights[prefix+"_ff_b2"] = matrixToSlice(block.FeedForward.B2)

		// Layer normalization weights
		checkpoint.Weights[prefix+"_ln1_gamma"] = matrixToSlice(block.LayerNorm1.Gamma)
		checkpoint.Weights[prefix+"_ln1_beta"] = matrixToSlice(block.LayerNorm1.Beta)
		checkpoint.Weights[prefix+"_ln2_gamma"] = matrixToSlice(block.LayerNorm2.Gamma)
		checkpoint.Weights[prefix+"_ln2_beta"] = matrixToSlice(block.LayerNorm2.Beta)
	}

	// Save final layer normalization
	checkpoint.Weights["final_ln_gamma"] = matrixToSlice(llm.FinalLayerNorm.Gamma)
	checkpoint.Weights["final_ln_beta"] = matrixToSlice(llm.FinalLayerNorm.Beta)

	// Save output layer
	checkpoint.Weights["output_layer"] = matrixToSlice(llm.OutputLayer)

	// Save to JSON file
	filePath := filepath.Join(modelDir, "model.json")
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create model file: %v", err)
	}
	defer func() {
		if err := file.Close(); err != nil {
			fmt.Printf("Warning: failed to close model file: %v\n", err)
		}
	}()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(checkpoint); err != nil {
		return fmt.Errorf("failed to encode model: %v", err)
	}

	fmt.Printf("Model saved to %s\n", filePath)
	return nil
}

// LoadModel loads a model from disk
func LoadModel(modelDir string) (*LLM, error) {
	filePath := filepath.Join(modelDir, "model.json")

	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open model file: %v", err)
	}
	defer func() {
		if err := file.Close(); err != nil {
			fmt.Printf("Warning: failed to close model file: %v\n", err)
		}
	}()

	var checkpoint ModelCheckpoint
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&checkpoint); err != nil {
		return nil, fmt.Errorf("failed to decode model: %v", err)
	}

	// Create new model with same architecture
	model := New(checkpoint.VocabSize, checkpoint.EmbedSize, checkpoint.HiddenSize, checkpoint.NumLayers)
	model.MaxSeqLen = checkpoint.MaxSeqLen

	// Load embedding weights
	if weights, exists := checkpoint.Weights["embedding"]; exists {
		sliceToMatrix(weights, model.Embedding.Weights)
	}

	// Load positional encoding
	if weights, exists := checkpoint.Weights["positional_encoding"]; exists {
		sliceToMatrix(weights, model.PositionalEnc.Encoding)
	}

	// Load transformer block weights
	for i := range model.TransformerBlocks {
		prefix := fmt.Sprintf("transformer_%d", i)
		block := model.TransformerBlocks[i]

		// Attention weights
		if weights, exists := checkpoint.Weights[prefix+"_attn_query"]; exists {
			sliceToMatrix(weights, block.Attention.WQuery)
		}
		if weights, exists := checkpoint.Weights[prefix+"_attn_key"]; exists {
			sliceToMatrix(weights, block.Attention.WKey)
		}
		if weights, exists := checkpoint.Weights[prefix+"_attn_value"]; exists {
			sliceToMatrix(weights, block.Attention.WValue)
		}
		if weights, exists := checkpoint.Weights[prefix+"_attn_output"]; exists {
			sliceToMatrix(weights, block.Attention.WOutput)
		}

		// Feed-forward weights
		if weights, exists := checkpoint.Weights[prefix+"_ff_w1"]; exists {
			sliceToMatrix(weights, block.FeedForward.W1)
		}
		if weights, exists := checkpoint.Weights[prefix+"_ff_w2"]; exists {
			sliceToMatrix(weights, block.FeedForward.W2)
		}
		if weights, exists := checkpoint.Weights[prefix+"_ff_b1"]; exists {
			sliceToMatrix(weights, block.FeedForward.B1)
		}
		if weights, exists := checkpoint.Weights[prefix+"_ff_b2"]; exists {
			sliceToMatrix(weights, block.FeedForward.B2)
		}

		// Layer normalization weights (with backward compatibility)
		if weights, exists := checkpoint.Weights[prefix+"_ln1_gamma"]; exists {
			sliceToMatrix(weights, block.LayerNorm1.Gamma)
		}
		if weights, exists := checkpoint.Weights[prefix+"_ln1_beta"]; exists {
			sliceToMatrix(weights, block.LayerNorm1.Beta)
		}
		if weights, exists := checkpoint.Weights[prefix+"_ln2_gamma"]; exists {
			sliceToMatrix(weights, block.LayerNorm2.Gamma)
		}
		if weights, exists := checkpoint.Weights[prefix+"_ln2_beta"]; exists {
			sliceToMatrix(weights, block.LayerNorm2.Beta)
		}
	}

	// Load final layer normalization (with backward compatibility)
	if weights, exists := checkpoint.Weights["final_ln_gamma"]; exists {
		sliceToMatrix(weights, model.FinalLayerNorm.Gamma)
	}
	if weights, exists := checkpoint.Weights["final_ln_beta"]; exists {
		sliceToMatrix(weights, model.FinalLayerNorm.Beta)
	}

	// Load output layer
	if weights, exists := checkpoint.Weights["output_layer"]; exists {
		sliceToMatrix(weights, model.OutputLayer)
	}

	fmt.Printf("Model loaded from %s\n", filePath)
	return model, nil
}

// matrixToSlice converts a matrix to a flat slice
func matrixToSlice(m *matrix.Matrix) []float64 {
	slice := make([]float64, m.Rows*m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			slice[i*m.Cols+j] = m.Get(i, j)
		}
	}
	return slice
}

// sliceToMatrix converts a flat slice back to a matrix
func sliceToMatrix(slice []float64, m *matrix.Matrix) {
	if len(slice) != m.Rows*m.Cols {
		return // Skip if dimensions don't match
	}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Set(i, j, slice[i*m.Cols+j])
		}
	}
}

// applyRepetitionPenalty reduces probability of already generated tokens with enhanced logic
func (llm *LLM) applyRepetitionPenalty(probs []float64, tokenCounts map[int]int, penalty float64) []float64 {
	penalizedProbs := make([]float64, len(probs))
	copy(penalizedProbs, probs)

	for tokenID, count := range tokenCounts {
		if tokenID < len(penalizedProbs) && count > 0 {
			// Enhanced penalty: stronger for frequent tokens, with minimum threshold
			if count > 2 { // Only penalize if seen more than twice
				penaltyFactor := math.Pow(penalty, float64(count-1))
				penalizedProbs[tokenID] = penalizedProbs[tokenID] / penaltyFactor

				// Ensure minimum probability to prevent complete elimination
				minProb := 0.001
				if penalizedProbs[tokenID] < minProb {
					penalizedProbs[tokenID] = minProb
				}
			}
		}
	}

	return normalize(penalizedProbs)
}

// applyNGramFiltering prevents repeating n-grams with enhanced diversity
func (llm *LLM) applyNGramFiltering(probs []float64, generated []int, ngramSize int) []float64 {
	if len(generated) < ngramSize {
		return probs
	}

	filteredProbs := make([]float64, len(probs))
	copy(filteredProbs, probs)

	// Track recent tokens for diversity boost
	recentTokens := make(map[int]int)
	lookbackWindow := min(len(generated), 15) // Look back 15 tokens
	for i := len(generated) - lookbackWindow; i < len(generated); i++ {
		recentTokens[generated[i]]++
	}

	// Check each possible next token for n-gram repetition
	for tokenID := 0; tokenID < len(probs); tokenID++ {
		// Create the potential n-gram
		if len(generated) >= ngramSize-1 {
			potentialNgram := make([]int, ngramSize)
			for i := 0; i < ngramSize-1; i++ {
				potentialNgram[i] = generated[len(generated)-(ngramSize-1)+i]
			}
			potentialNgram[ngramSize-1] = tokenID

			// Check if this exact n-gram already exists
			ngramExists := false
			for i := 0; i <= len(generated)-ngramSize; i++ {
				match := true
				for j := 0; j < ngramSize; j++ {
					if generated[i+j] != potentialNgram[j] {
						match = false
						break
					}
				}
				if match {
					ngramExists = true
					break
				}
			}

			if ngramExists {
				// Significantly reduce probability for repeated n-grams
				filteredProbs[tokenID] *= 0.02
			}
		}

		// Apply diversity boost based on recent usage
		if count, exists := recentTokens[tokenID]; exists {
			// Reduce probability based on frequency in recent window
			reductionFactor := 1.0 / (1.0 + float64(count)*0.3)
			filteredProbs[tokenID] *= reductionFactor
		} else {
			// Boost tokens that haven't been used recently
			filteredProbs[tokenID] *= 1.3
		}
	}

	return normalize(filteredProbs)
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// isRepetitiveSequence checks if the sequence is getting repetitive
func (llm *LLM) isRepetitiveSequence(sequence []int, windowSize int) bool {
	if len(sequence) < windowSize*2 {
		return false
	}

	// Check if the last windowSize tokens repeat
	lastWindow := sequence[len(sequence)-windowSize:]
	prevWindow := sequence[len(sequence)-windowSize*2 : len(sequence)-windowSize]

	// Check if the last two windows are identical
	for i := 0; i < windowSize; i++ {
		if lastWindow[i] != prevWindow[i] {
			return false
		}
	}

	return true
}
