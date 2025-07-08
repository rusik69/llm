package llm

import (
	"math"

	"github.com/rusik69/shittyllm/pkg/matrix"
)

// AttentionLayer implements multi-head self-attention
type AttentionLayer struct {
	EmbedSize int
	NumHeads  int
	HeadSize  int

	// Weight matrices for query, key, value projections
	WQuery, WKey, WValue *matrix.Matrix
	WOutput              *matrix.Matrix
}

// NewAttentionLayer creates a new attention layer
func NewAttentionLayer(embedSize, numHeads int) *AttentionLayer {
	if embedSize%numHeads != 0 {
		panic("Embed size must be divisible by number of heads")
	}

	headSize := embedSize / numHeads

	layer := &AttentionLayer{
		EmbedSize: embedSize,
		NumHeads:  numHeads,
		HeadSize:  headSize,
		WQuery:    matrix.New(embedSize, embedSize),
		WKey:      matrix.New(embedSize, embedSize),
		WValue:    matrix.New(embedSize, embedSize),
		WOutput:   matrix.New(embedSize, embedSize),
	}

	// Initialize weights
	layer.WQuery.Xavier()
	layer.WKey.Xavier()
	layer.WValue.Xavier()
	layer.WOutput.Xavier()

	return layer
}

// Forward performs the attention computation
func (a *AttentionLayer) Forward(input *matrix.Matrix) *matrix.Matrix {
	seqLen := input.Rows

	// Compute Q, K, V matrices
	Q := input.Multiply(a.WQuery)
	K := input.Multiply(a.WKey)
	V := input.Multiply(a.WValue)

	// Reshape for multi-head attention
	// For simplicity, we'll implement single-head attention
	// In a full implementation, you'd split into multiple heads

	// Compute attention scores
	scores := Q.Multiply(K.Transpose())

	// Scale by sqrt(head_size)
	scaleFactor := 1.0 / math.Sqrt(float64(a.HeadSize))
	scores = scores.Scale(scaleFactor)

	// Apply causal mask (for autoregressive generation)
	for i := 0; i < seqLen; i++ {
		for j := i + 1; j < seqLen; j++ {
			scores.Set(i, j, -1e9) // Large negative value
		}
	}

	// Apply softmax to get attention weights
	attentionWeights := scores.Softmax()

	// Apply attention to values
	output := attentionWeights.Multiply(V)

	// Apply output projection
	return output.Multiply(a.WOutput)
}

// FeedForwardLayer implements the position-wise feed-forward network
type FeedForwardLayer struct {
	InputSize  int
	HiddenSize int

	W1, W2 *matrix.Matrix
	B1, B2 *matrix.Matrix
}

// NewFeedForwardLayer creates a new feed-forward layer
func NewFeedForwardLayer(inputSize, hiddenSize int) *FeedForwardLayer {
	layer := &FeedForwardLayer{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		W1:         matrix.New(inputSize, hiddenSize),
		W2:         matrix.New(hiddenSize, inputSize),
		B1:         matrix.New(1, hiddenSize),
		B2:         matrix.New(1, inputSize),
	}

	// Initialize weights
	layer.W1.Xavier()
	layer.W2.Xavier()
	layer.B1.Random()
	layer.B2.Random()

	return layer
}

// Forward performs the feed-forward computation
func (ff *FeedForwardLayer) Forward(input *matrix.Matrix) *matrix.Matrix {
	// First linear transformation + ReLU
	hidden := input.Multiply(ff.W1)

	// Add bias (broadcast across all rows)
	for i := 0; i < hidden.Rows; i++ {
		for j := 0; j < hidden.Cols; j++ {
			hidden.Set(i, j, hidden.Get(i, j)+ff.B1.Get(0, j))
		}
	}

	// Apply GeLU activation (better than ReLU for transformers)
	hidden = hidden.ApplyFunction(matrix.GeLU)

	// Second linear transformation
	output := hidden.Multiply(ff.W2)

	// Add bias
	for i := 0; i < output.Rows; i++ {
		for j := 0; j < output.Cols; j++ {
			output.Set(i, j, output.Get(i, j)+ff.B2.Get(0, j))
		}
	}

	return output
}

// LayerNormalization implements layer normalization with learnable parameters
type LayerNormalization struct {
	EmbedSize int
	Gamma     *matrix.Matrix // Scale parameters
	Beta      *matrix.Matrix // Shift parameters
	Epsilon   float64
}

// NewLayerNormalization creates a new layer normalization layer
func NewLayerNormalization(embedSize int) *LayerNormalization {
	gamma := matrix.New(1, embedSize)
	beta := matrix.New(1, embedSize)

	// Initialize gamma to 1 and beta to 0
	for i := 0; i < embedSize; i++ {
		gamma.Set(0, i, 1.0)
		beta.Set(0, i, 0.0)
	}

	return &LayerNormalization{
		EmbedSize: embedSize,
		Gamma:     gamma,
		Beta:      beta,
		Epsilon:   1e-6,
	}
}

// Forward applies layer normalization
func (ln *LayerNormalization) Forward(input *matrix.Matrix) *matrix.Matrix {
	// Apply basic layer normalization
	normalized := input.LayerNorm(ln.Epsilon)

	// Apply scale and shift (gamma and beta)
	result := matrix.New(input.Rows, input.Cols)
	for i := 0; i < input.Rows; i++ {
		for j := 0; j < input.Cols; j++ {
			scaled := normalized.Get(i, j) * ln.Gamma.Get(0, j)
			result.Set(i, j, scaled+ln.Beta.Get(0, j))
		}
	}

	return result
}

// TransformerBlock combines attention and feed-forward layers with layer normalization
type TransformerBlock struct {
	Attention   *AttentionLayer
	FeedForward *FeedForwardLayer
	LayerNorm1  *LayerNormalization
	LayerNorm2  *LayerNormalization
	EmbedSize   int
	DropoutRate float64
	IsTraining  bool
}

// NewTransformerBlock creates a new transformer block
func NewTransformerBlock(embedSize, numHeads, hiddenSize int) *TransformerBlock {
	return &TransformerBlock{
		Attention:   NewAttentionLayer(embedSize, numHeads),
		FeedForward: NewFeedForwardLayer(embedSize, hiddenSize),
		LayerNorm1:  NewLayerNormalization(embedSize),
		LayerNorm2:  NewLayerNormalization(embedSize),
		EmbedSize:   embedSize,
		DropoutRate: 0.1, // Default dropout rate
		IsTraining:  true,
	}
}

// Forward performs the transformer block computation with Pre-Layer Normalization
func (tb *TransformerBlock) Forward(input *matrix.Matrix) *matrix.Matrix {
	// Pre-Layer Normalization + Self-attention + Residual connection
	normalized1 := tb.LayerNorm1.Forward(input)
	attnOutput := tb.Attention.Forward(normalized1)
	attnOutput = attnOutput.Dropout(tb.DropoutRate, tb.IsTraining)
	attnOutput = attnOutput.Add(input) // Residual connection

	// Pre-Layer Normalization + Feed-forward + Residual connection
	normalized2 := tb.LayerNorm2.Forward(attnOutput)
	ffOutput := tb.FeedForward.Forward(normalized2)
	ffOutput = ffOutput.Dropout(tb.DropoutRate, tb.IsTraining)
	ffOutput = ffOutput.Add(attnOutput) // Residual connection

	return ffOutput
}

// SetTraining sets the training mode for dropout
func (tb *TransformerBlock) SetTraining(training bool) {
	tb.IsTraining = training
}

// EmbeddingLayer converts token IDs to embeddings
type EmbeddingLayer struct {
	VocabSize int
	EmbedSize int
	Weights   *matrix.Matrix
}

// NewEmbeddingLayer creates a new embedding layer
func NewEmbeddingLayer(vocabSize, embedSize int) *EmbeddingLayer {
	layer := &EmbeddingLayer{
		VocabSize: vocabSize,
		EmbedSize: embedSize,
		Weights:   matrix.New(vocabSize, embedSize),
	}

	layer.Weights.Xavier()
	return layer
}

// Forward converts token IDs to embeddings
func (e *EmbeddingLayer) Forward(tokenIDs []int) *matrix.Matrix {
	seqLen := len(tokenIDs)
	embeddings := matrix.New(seqLen, e.EmbedSize)

	for i, tokenID := range tokenIDs {
		if tokenID >= 0 && tokenID < e.VocabSize {
			for j := 0; j < e.EmbedSize; j++ {
				embeddings.Set(i, j, e.Weights.Get(tokenID, j))
			}
		}
	}

	return embeddings
}

// PositionalEncoding adds positional information to embeddings
type PositionalEncoding struct {
	MaxSeqLen int
	EmbedSize int
	Encoding  *matrix.Matrix
}

// NewPositionalEncoding creates positional encodings
func NewPositionalEncoding(maxSeqLen, embedSize int) *PositionalEncoding {
	pe := &PositionalEncoding{
		MaxSeqLen: maxSeqLen,
		EmbedSize: embedSize,
		Encoding:  matrix.New(maxSeqLen, embedSize),
	}

	// Compute positional encodings using sine and cosine functions
	for pos := 0; pos < maxSeqLen; pos++ {
		for i := 0; i < embedSize; i++ {
			if i%2 == 0 {
				// Use sine for even dimensions
				angle := float64(pos) / math.Pow(10000, float64(i)/float64(embedSize))
				pe.Encoding.Set(pos, i, math.Sin(angle))
			} else {
				// Use cosine for odd dimensions
				angle := float64(pos) / math.Pow(10000, float64(i-1)/float64(embedSize))
				pe.Encoding.Set(pos, i, math.Cos(angle))
			}
		}
	}

	return pe
}

// Forward adds positional encoding to input embeddings
func (pe *PositionalEncoding) Forward(embeddings *matrix.Matrix) *matrix.Matrix {
	seqLen := embeddings.Rows
	result := embeddings.Copy()

	for i := 0; i < seqLen && i < pe.MaxSeqLen; i++ {
		for j := 0; j < embeddings.Cols; j++ {
			current := result.Get(i, j)
			positional := pe.Encoding.Get(i, j)
			result.Set(i, j, current+positional)
		}
	}

	return result
}
