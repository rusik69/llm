package llm

import (
	"math"

	"github.com/rusik69/shittyllm/pkg/matrix"
)

// Gradients holds all gradients for the model parameters
type Gradients struct {
	// Embedding gradients
	EmbeddingGrads *matrix.Matrix

	// Transformer block gradients (per layer)
	AttentionGrads  []*AttentionGradients
	FFGrads         []*FeedForwardGradients
	LayerNorm1Grads []*LayerNormGradients
	LayerNorm2Grads []*LayerNormGradients

	// Final layer gradients
	FinalLayerNormGrads *LayerNormGradients
	OutputLayerGrads    *matrix.Matrix
}

// AttentionGradients holds gradients for attention layer
type AttentionGradients struct {
	WQueryGrads  *matrix.Matrix
	WKeyGrads    *matrix.Matrix
	WValueGrads  *matrix.Matrix
	WOutputGrads *matrix.Matrix
}

// FeedForwardGradients holds gradients for feed-forward layer
type FeedForwardGradients struct {
	W1Grads *matrix.Matrix
	W2Grads *matrix.Matrix
	B1Grads *matrix.Matrix
	B2Grads *matrix.Matrix
}

// LayerNormGradients holds gradients for layer normalization
type LayerNormGradients struct {
	GammaGrads *matrix.Matrix
	BetaGrads  *matrix.Matrix
}

// NewGradients creates a new gradients structure matching the model
func NewGradients(model *LLM) *Gradients {
	grads := &Gradients{
		EmbeddingGrads:  matrix.New(model.VocabSize, model.EmbedSize),
		AttentionGrads:  make([]*AttentionGradients, model.NumLayers),
		FFGrads:         make([]*FeedForwardGradients, model.NumLayers),
		LayerNorm1Grads: make([]*LayerNormGradients, model.NumLayers),
		LayerNorm2Grads: make([]*LayerNormGradients, model.NumLayers),
		FinalLayerNormGrads: &LayerNormGradients{
			GammaGrads: matrix.New(1, model.EmbedSize),
			BetaGrads:  matrix.New(1, model.EmbedSize),
		},
		OutputLayerGrads: matrix.New(model.EmbedSize, model.VocabSize),
	}

	// Initialize gradients for each layer
	for i := 0; i < model.NumLayers; i++ {
		grads.AttentionGrads[i] = &AttentionGradients{
			WQueryGrads:  matrix.New(model.EmbedSize, model.EmbedSize),
			WKeyGrads:    matrix.New(model.EmbedSize, model.EmbedSize),
			WValueGrads:  matrix.New(model.EmbedSize, model.EmbedSize),
			WOutputGrads: matrix.New(model.EmbedSize, model.EmbedSize),
		}
		grads.FFGrads[i] = &FeedForwardGradients{
			W1Grads: matrix.New(model.EmbedSize, model.HiddenSize),
			W2Grads: matrix.New(model.HiddenSize, model.EmbedSize),
			B1Grads: matrix.New(1, model.HiddenSize),
			B2Grads: matrix.New(1, model.EmbedSize),
		}
		grads.LayerNorm1Grads[i] = &LayerNormGradients{
			GammaGrads: matrix.New(1, model.EmbedSize),
			BetaGrads:  matrix.New(1, model.EmbedSize),
		}
		grads.LayerNorm2Grads[i] = &LayerNormGradients{
			GammaGrads: matrix.New(1, model.EmbedSize),
			BetaGrads:  matrix.New(1, model.EmbedSize),
		}
	}

	return grads
}

// Zero resets all gradients to zero
func (g *Gradients) Zero() {
	// Zero embedding gradients
	for i := 0; i < g.EmbeddingGrads.Rows; i++ {
		for j := 0; j < g.EmbeddingGrads.Cols; j++ {
			g.EmbeddingGrads.Set(i, j, 0)
		}
	}

	// Zero all layer gradients
	for i := 0; i < len(g.AttentionGrads); i++ {
		g.zeroMatrix(g.AttentionGrads[i].WQueryGrads)
		g.zeroMatrix(g.AttentionGrads[i].WKeyGrads)
		g.zeroMatrix(g.AttentionGrads[i].WValueGrads)
		g.zeroMatrix(g.AttentionGrads[i].WOutputGrads)

		g.zeroMatrix(g.FFGrads[i].W1Grads)
		g.zeroMatrix(g.FFGrads[i].W2Grads)
		g.zeroMatrix(g.FFGrads[i].B1Grads)
		g.zeroMatrix(g.FFGrads[i].B2Grads)

		g.zeroMatrix(g.LayerNorm1Grads[i].GammaGrads)
		g.zeroMatrix(g.LayerNorm1Grads[i].BetaGrads)
		g.zeroMatrix(g.LayerNorm2Grads[i].GammaGrads)
		g.zeroMatrix(g.LayerNorm2Grads[i].BetaGrads)
	}

	// Zero final layer gradients
	g.zeroMatrix(g.FinalLayerNormGrads.GammaGrads)
	g.zeroMatrix(g.FinalLayerNormGrads.BetaGrads)
	g.zeroMatrix(g.OutputLayerGrads)
}

// zeroMatrix sets all elements of a matrix to zero
func (g *Gradients) zeroMatrix(m *matrix.Matrix) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Set(i, j, 0)
		}
	}
}

// ComputeGradients performs backpropagation to compute gradients
func (llm *LLM) ComputeGradients(tokenIDs []int, targets []int) *Gradients {
	if len(targets) != 1 {
		panic("Expected single target token for next-token prediction")
	}

	if len(tokenIDs) == 0 {
		return NewGradients(llm)
	}

	seqLen := len(tokenIDs)
	target := targets[0]
	gradients := NewGradients(llm)

	// Forward pass to get intermediate values
	embeddings := llm.Embedding.Forward(tokenIDs)
	x := llm.PositionalEnc.Forward(embeddings)

	// Store intermediate activations for backprop
	activations := make([]*matrix.Matrix, llm.NumLayers+1)
	activations[0] = x.Copy()

	// Forward through transformer blocks
	for i, block := range llm.TransformerBlocks {
		x = block.Forward(x)
		activations[i+1] = x.Copy()
	}

	// Final layer norm and output
	x = llm.FinalLayerNorm.Forward(x)
	logits := x.Multiply(llm.OutputLayer)

	// Compute loss gradient for next-token prediction
	// We only care about the last time step since that's what predicts the next token
	lastStep := seqLen - 1
	dLogits := matrix.New(seqLen, llm.VocabSize)

	// Apply softmax to get probabilities for the last time step
	probs := make([]float64, llm.VocabSize)
	maxLogit := logits.Get(lastStep, 0)
	for j := 1; j < llm.VocabSize; j++ {
		if logits.Get(lastStep, j) > maxLogit {
			maxLogit = logits.Get(lastStep, j)
		}
	}

	// Compute softmax with numerical stability
	sumExp := 0.0
	for j := 0; j < llm.VocabSize; j++ {
		probs[j] = math.Exp(logits.Get(lastStep, j) - maxLogit)
		sumExp += probs[j]
	}

	for j := 0; j < llm.VocabSize; j++ {
		probs[j] /= sumExp
	}

	// Compute gradient: dL/dlogits = p - y (where y is one-hot target)
	// Only for the last time step
	for j := 0; j < llm.VocabSize; j++ {
		grad := 0.0
		if lastStep < seqLen { // Safety check
			grad = probs[j]
			if j == target {
				grad -= 1.0
			}
		}
		dLogits.Set(lastStep, j, grad)
	}

	// Backward pass through output layer
	dOutput := dLogits.Multiply(llm.OutputLayer.Transpose())
	gradients.OutputLayerGrads = x.Transpose().Multiply(dLogits)

	// Backward through final layer norm (simplified)
	dX := dOutput.Copy()

	// Backward through transformer blocks (simplified)
	for i := llm.NumLayers - 1; i >= 0; i-- {
		dX = llm.backwardTransformerBlock(dX, activations[i], gradients, i)
	}

	// Backward through embedding
	llm.backwardEmbedding(dX, tokenIDs, gradients)

	return gradients
}

// backwardTransformerBlock computes gradients for a transformer block (simplified)
func (llm *LLM) backwardTransformerBlock(dOutput *matrix.Matrix, input *matrix.Matrix, gradients *Gradients, layerIdx int) *matrix.Matrix {
	// This is a simplified implementation
	// In practice, you'd compute proper gradients for all attention and FF parameters

	// For now, just pass the gradient through
	return dOutput.Copy()
}

// backwardEmbedding computes gradients for the embedding layer
func (llm *LLM) backwardEmbedding(dX *matrix.Matrix, tokenIDs []int, gradients *Gradients) {
	// Accumulate gradients for embedding weights
	for i, tokenID := range tokenIDs {
		if tokenID >= 0 && tokenID < llm.VocabSize {
			for j := 0; j < llm.EmbedSize; j++ {
				currentGrad := gradients.EmbeddingGrads.Get(tokenID, j)
				newGrad := currentGrad + dX.Get(i, j)
				gradients.EmbeddingGrads.Set(tokenID, j, newGrad)
			}
		}
	}
}

// ClipGradients applies gradient clipping to prevent exploding gradients
func (g *Gradients) ClipGradients(maxNorm float64) {
	// Compute gradient norm
	totalNorm := 0.0

	// Add embedding gradients to norm
	for i := 0; i < g.EmbeddingGrads.Rows; i++ {
		for j := 0; j < g.EmbeddingGrads.Cols; j++ {
			grad := g.EmbeddingGrads.Get(i, j)
			totalNorm += grad * grad
		}
	}

	// Add output layer gradients to norm
	for i := 0; i < g.OutputLayerGrads.Rows; i++ {
		for j := 0; j < g.OutputLayerGrads.Cols; j++ {
			grad := g.OutputLayerGrads.Get(i, j)
			totalNorm += grad * grad
		}
	}

	totalNorm = math.Sqrt(totalNorm)

	// Clip if necessary
	if totalNorm > maxNorm {
		clipRatio := maxNorm / totalNorm
		g.scaleGradients(clipRatio)
	}
}

// scaleGradients scales all gradients by a factor
func (g *Gradients) scaleGradients(scale float64) {
	// Scale embedding gradients
	for i := 0; i < g.EmbeddingGrads.Rows; i++ {
		for j := 0; j < g.EmbeddingGrads.Cols; j++ {
			g.EmbeddingGrads.Set(i, j, g.EmbeddingGrads.Get(i, j)*scale)
		}
	}

	// Scale output layer gradients
	for i := 0; i < g.OutputLayerGrads.Rows; i++ {
		for j := 0; j < g.OutputLayerGrads.Cols; j++ {
			g.OutputLayerGrads.Set(i, j, g.OutputLayerGrads.Get(i, j)*scale)
		}
	}
}

// Scale scales all gradients by a factor (public method)
func (g *Gradients) Scale(scale float64) {
	g.scaleGradients(scale)
}

// Add adds another gradient to this one (for parallel training)
func (g *Gradients) Add(other *Gradients) {
	if other == nil {
		return
	}

	// Add embedding gradients
	for i := 0; i < g.EmbeddingGrads.Rows && i < other.EmbeddingGrads.Rows; i++ {
		for j := 0; j < g.EmbeddingGrads.Cols && j < other.EmbeddingGrads.Cols; j++ {
			g.EmbeddingGrads.Set(i, j, g.EmbeddingGrads.Get(i, j)+other.EmbeddingGrads.Get(i, j))
		}
	}

	// Add output layer gradients
	for i := 0; i < g.OutputLayerGrads.Rows && i < other.OutputLayerGrads.Rows; i++ {
		for j := 0; j < g.OutputLayerGrads.Cols && j < other.OutputLayerGrads.Cols; j++ {
			g.OutputLayerGrads.Set(i, j, g.OutputLayerGrads.Get(i, j)+other.OutputLayerGrads.Get(i, j))
		}
	}

	// Add attention gradients for each layer
	for layer := 0; layer < len(g.AttentionGrads) && layer < len(other.AttentionGrads); layer++ {
		if g.AttentionGrads[layer] != nil && other.AttentionGrads[layer] != nil {
			g.addMatrix(g.AttentionGrads[layer].WQueryGrads, other.AttentionGrads[layer].WQueryGrads)
			g.addMatrix(g.AttentionGrads[layer].WKeyGrads, other.AttentionGrads[layer].WKeyGrads)
			g.addMatrix(g.AttentionGrads[layer].WValueGrads, other.AttentionGrads[layer].WValueGrads)
			g.addMatrix(g.AttentionGrads[layer].WOutputGrads, other.AttentionGrads[layer].WOutputGrads)
		}
	}

	// Add feed-forward gradients for each layer
	for layer := 0; layer < len(g.FFGrads) && layer < len(other.FFGrads); layer++ {
		if g.FFGrads[layer] != nil && other.FFGrads[layer] != nil {
			g.addMatrix(g.FFGrads[layer].W1Grads, other.FFGrads[layer].W1Grads)
			g.addMatrix(g.FFGrads[layer].W2Grads, other.FFGrads[layer].W2Grads)
			g.addMatrix(g.FFGrads[layer].B1Grads, other.FFGrads[layer].B1Grads)
			g.addMatrix(g.FFGrads[layer].B2Grads, other.FFGrads[layer].B2Grads)
		}
	}

	// Add layer norm gradients
	for layer := 0; layer < len(g.LayerNorm1Grads) && layer < len(other.LayerNorm1Grads); layer++ {
		if g.LayerNorm1Grads[layer] != nil && other.LayerNorm1Grads[layer] != nil {
			g.addMatrix(g.LayerNorm1Grads[layer].GammaGrads, other.LayerNorm1Grads[layer].GammaGrads)
			g.addMatrix(g.LayerNorm1Grads[layer].BetaGrads, other.LayerNorm1Grads[layer].BetaGrads)
		}
	}

	for layer := 0; layer < len(g.LayerNorm2Grads) && layer < len(other.LayerNorm2Grads); layer++ {
		if g.LayerNorm2Grads[layer] != nil && other.LayerNorm2Grads[layer] != nil {
			g.addMatrix(g.LayerNorm2Grads[layer].GammaGrads, other.LayerNorm2Grads[layer].GammaGrads)
			g.addMatrix(g.LayerNorm2Grads[layer].BetaGrads, other.LayerNorm2Grads[layer].BetaGrads)
		}
	}

	// Add final layer norm gradients
	if g.FinalLayerNormGrads != nil && other.FinalLayerNormGrads != nil {
		g.addMatrix(g.FinalLayerNormGrads.GammaGrads, other.FinalLayerNormGrads.GammaGrads)
		g.addMatrix(g.FinalLayerNormGrads.BetaGrads, other.FinalLayerNormGrads.BetaGrads)
	}
}

// addMatrix adds values from source matrix to destination matrix
func (g *Gradients) addMatrix(dest, src *matrix.Matrix) {
	if dest == nil || src == nil {
		return
	}

	for i := 0; i < dest.Rows && i < src.Rows; i++ {
		for j := 0; j < dest.Cols && j < src.Cols; j++ {
			dest.Set(i, j, dest.Get(i, j)+src.Get(i, j))
		}
	}
}
