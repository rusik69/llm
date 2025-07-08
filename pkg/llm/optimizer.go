package llm

import (
	"math"

	"github.com/rusik69/shittyllm/pkg/matrix"
)

// OptimizerConfig holds configuration for the optimizer
type OptimizerConfig struct {
	LearningRate float64
	Beta1        float64 // Exponential decay rate for first moment estimates
	Beta2        float64 // Exponential decay rate for second moment estimates
	Epsilon      float64 // Small constant for numerical stability
	WeightDecay  float64 // L2 regularization strength
}

// DefaultOptimizerConfig returns sensible defaults for Adam optimizer
func DefaultOptimizerConfig() *OptimizerConfig {
	return &OptimizerConfig{
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  0.01,
	}
}

// AdamState holds the state for Adam optimizer
type AdamState struct {
	// First moment estimates (momentum)
	EmbeddingM      *matrix.Matrix
	AttentionM      []*AttentionMoments
	FFM             []*FeedForwardMoments
	LayerNorm1M     []*LayerNormMoments
	LayerNorm2M     []*LayerNormMoments
	FinalLayerNormM *LayerNormMoments
	OutputLayerM    *matrix.Matrix

	// Second moment estimates (RMSprop)
	EmbeddingV      *matrix.Matrix
	AttentionV      []*AttentionMoments
	FFV             []*FeedForwardMoments
	LayerNorm1V     []*LayerNormMoments
	LayerNorm2V     []*LayerNormMoments
	FinalLayerNormV *LayerNormMoments
	OutputLayerV    *matrix.Matrix

	// Step counter
	Step int64
}

// AttentionMoments holds momentum for attention parameters
type AttentionMoments struct {
	WQueryM, WKeyM, WValueM, WOutputM *matrix.Matrix
}

// FeedForwardMoments holds momentum for feed-forward parameters
type FeedForwardMoments struct {
	W1M, W2M, B1M, B2M *matrix.Matrix
}

// LayerNormMoments holds momentum for layer normalization parameters
type LayerNormMoments struct {
	GammaM, BetaM *matrix.Matrix
}

// NewAdamState creates a new Adam optimizer state
func NewAdamState(model *LLM) *AdamState {
	state := &AdamState{
		EmbeddingM:  matrix.New(model.VocabSize, model.EmbedSize),
		AttentionM:  make([]*AttentionMoments, model.NumLayers),
		FFM:         make([]*FeedForwardMoments, model.NumLayers),
		LayerNorm1M: make([]*LayerNormMoments, model.NumLayers),
		LayerNorm2M: make([]*LayerNormMoments, model.NumLayers),
		FinalLayerNormM: &LayerNormMoments{
			GammaM: matrix.New(1, model.EmbedSize),
			BetaM:  matrix.New(1, model.EmbedSize),
		},
		OutputLayerM: matrix.New(model.EmbedSize, model.VocabSize),

		EmbeddingV:  matrix.New(model.VocabSize, model.EmbedSize),
		AttentionV:  make([]*AttentionMoments, model.NumLayers),
		FFV:         make([]*FeedForwardMoments, model.NumLayers),
		LayerNorm1V: make([]*LayerNormMoments, model.NumLayers),
		LayerNorm2V: make([]*LayerNormMoments, model.NumLayers),
		FinalLayerNormV: &LayerNormMoments{
			GammaM: matrix.New(1, model.EmbedSize),
			BetaM:  matrix.New(1, model.EmbedSize),
		},
		OutputLayerV: matrix.New(model.EmbedSize, model.VocabSize),

		Step: 0,
	}

	// Initialize layer-specific states
	for i := 0; i < model.NumLayers; i++ {
		state.AttentionM[i] = &AttentionMoments{
			WQueryM:  matrix.New(model.EmbedSize, model.EmbedSize),
			WKeyM:    matrix.New(model.EmbedSize, model.EmbedSize),
			WValueM:  matrix.New(model.EmbedSize, model.EmbedSize),
			WOutputM: matrix.New(model.EmbedSize, model.EmbedSize),
		}
		state.FFM[i] = &FeedForwardMoments{
			W1M: matrix.New(model.EmbedSize, model.HiddenSize),
			W2M: matrix.New(model.HiddenSize, model.EmbedSize),
			B1M: matrix.New(1, model.HiddenSize),
			B2M: matrix.New(1, model.EmbedSize),
		}
		state.LayerNorm1M[i] = &LayerNormMoments{
			GammaM: matrix.New(1, model.EmbedSize),
			BetaM:  matrix.New(1, model.EmbedSize),
		}
		state.LayerNorm2M[i] = &LayerNormMoments{
			GammaM: matrix.New(1, model.EmbedSize),
			BetaM:  matrix.New(1, model.EmbedSize),
		}

		state.AttentionV[i] = &AttentionMoments{
			WQueryM:  matrix.New(model.EmbedSize, model.EmbedSize),
			WKeyM:    matrix.New(model.EmbedSize, model.EmbedSize),
			WValueM:  matrix.New(model.EmbedSize, model.EmbedSize),
			WOutputM: matrix.New(model.EmbedSize, model.EmbedSize),
		}
		state.FFV[i] = &FeedForwardMoments{
			W1M: matrix.New(model.EmbedSize, model.HiddenSize),
			W2M: matrix.New(model.HiddenSize, model.EmbedSize),
			B1M: matrix.New(1, model.HiddenSize),
			B2M: matrix.New(1, model.EmbedSize),
		}
		state.LayerNorm1V[i] = &LayerNormMoments{
			GammaM: matrix.New(1, model.EmbedSize),
			BetaM:  matrix.New(1, model.EmbedSize),
		}
		state.LayerNorm2V[i] = &LayerNormMoments{
			GammaM: matrix.New(1, model.EmbedSize),
			BetaM:  matrix.New(1, model.EmbedSize),
		}
	}

	return state
}

// AdamOptimizer implements the Adam optimization algorithm
type AdamOptimizer struct {
	Config *OptimizerConfig
	State  *AdamState
}

// NewAdamOptimizer creates a new Adam optimizer
func NewAdamOptimizer(model *LLM, config *OptimizerConfig) *AdamOptimizer {
	if config == nil {
		config = DefaultOptimizerConfig()
	}

	return &AdamOptimizer{
		Config: config,
		State:  NewAdamState(model),
	}
}

// Step performs one optimization step
func (opt *AdamOptimizer) Step(model *LLM, gradients *Gradients) {
	opt.State.Step++

	// Bias correction terms
	beta1t := math.Pow(opt.Config.Beta1, float64(opt.State.Step))
	beta2t := math.Pow(opt.Config.Beta2, float64(opt.State.Step))
	biasCorrection1 := 1.0 - beta1t
	biasCorrection2 := 1.0 - beta2t

	stepSize := opt.Config.LearningRate * math.Sqrt(biasCorrection2) / biasCorrection1

	// Update embedding parameters
	opt.updateMatrix(
		model.Embedding.Weights,
		gradients.EmbeddingGrads,
		opt.State.EmbeddingM,
		opt.State.EmbeddingV,
		stepSize,
	)

	// Update output layer parameters
	opt.updateMatrix(
		model.OutputLayer,
		gradients.OutputLayerGrads,
		opt.State.OutputLayerM,
		opt.State.OutputLayerV,
		stepSize,
	)

	// Update layer normalization parameters
	opt.updateMatrix(
		model.FinalLayerNorm.Gamma,
		gradients.FinalLayerNormGrads.GammaGrads,
		opt.State.FinalLayerNormM.GammaM,
		opt.State.FinalLayerNormV.GammaM,
		stepSize,
	)
	opt.updateMatrix(
		model.FinalLayerNorm.Beta,
		gradients.FinalLayerNormGrads.BetaGrads,
		opt.State.FinalLayerNormM.BetaM,
		opt.State.FinalLayerNormV.BetaM,
		stepSize,
	)

	// Update transformer block parameters (simplified - only the main matrices)
	for i := 0; i < model.NumLayers; i++ {
		// Update attention layer normalization
		opt.updateMatrix(
			model.TransformerBlocks[i].LayerNorm1.Gamma,
			gradients.LayerNorm1Grads[i].GammaGrads,
			opt.State.LayerNorm1M[i].GammaM,
			opt.State.LayerNorm1V[i].GammaM,
			stepSize,
		)
		opt.updateMatrix(
			model.TransformerBlocks[i].LayerNorm2.Gamma,
			gradients.LayerNorm2Grads[i].GammaGrads,
			opt.State.LayerNorm2M[i].GammaM,
			opt.State.LayerNorm2V[i].GammaM,
			stepSize,
		)
	}
}

// updateMatrix updates a single matrix using Adam algorithm
func (opt *AdamOptimizer) updateMatrix(params, grads, m, v *matrix.Matrix, stepSize float64) {
	for i := 0; i < params.Rows; i++ {
		for j := 0; j < params.Cols; j++ {
			grad := grads.Get(i, j)
			param := params.Get(i, j)

			// Add weight decay (L2 regularization)
			if opt.Config.WeightDecay > 0 {
				grad += opt.Config.WeightDecay * param
			}

			// Update first moment (momentum)
			mVal := opt.Config.Beta1*m.Get(i, j) + (1-opt.Config.Beta1)*grad
			m.Set(i, j, mVal)

			// Update second moment (RMSprop)
			vVal := opt.Config.Beta2*v.Get(i, j) + (1-opt.Config.Beta2)*grad*grad
			v.Set(i, j, vVal)

			// Update parameter
			update := stepSize * mVal / (math.Sqrt(vVal) + opt.Config.Epsilon)
			newParam := param - update

			// Optional: clamp parameters to prevent extreme values
			if newParam > 10.0 {
				newParam = 10.0
			} else if newParam < -10.0 {
				newParam = -10.0
			}

			params.Set(i, j, newParam)
		}
	}
}

// GetLearningRate returns the current effective learning rate
func (opt *AdamOptimizer) GetLearningRate() float64 {
	if opt.State.Step == 0 {
		return opt.Config.LearningRate
	}

	beta1t := math.Pow(opt.Config.Beta1, float64(opt.State.Step))
	beta2t := math.Pow(opt.Config.Beta2, float64(opt.State.Step))
	biasCorrection1 := 1.0 - beta1t
	biasCorrection2 := 1.0 - beta2t

	return opt.Config.LearningRate * math.Sqrt(biasCorrection2) / biasCorrection1
}

// SetLearningRate updates the learning rate
func (opt *AdamOptimizer) SetLearningRate(lr float64) {
	opt.Config.LearningRate = lr
}
