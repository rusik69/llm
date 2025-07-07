package matrix

import (
	"math"
	"math/rand"
)

// Matrix represents a 2D matrix
type Matrix struct {
	Rows, Cols int
	Data       [][]float64
}

// New creates a new matrix with given dimensions
func New(rows, cols int) *Matrix {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: data,
	}
}

// NewFromData creates a matrix from existing data
func NewFromData(data [][]float64) *Matrix {
	rows := len(data)
	if rows == 0 {
		return New(0, 0)
	}
	cols := len(data[0])

	m := New(rows, cols)
	for i := 0; i < rows; i++ {
		copy(m.Data[i], data[i])
	}
	return m
}

// Random initializes matrix with random values
func (m *Matrix) Random() {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] = rand.Float64()*2 - 1 // Random between -1 and 1
		}
	}
}

// Xavier initializes matrix with Xavier initialization
func (m *Matrix) Xavier() {
	limit := math.Sqrt(6.0 / float64(m.Rows+m.Cols))
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] = (rand.Float64()*2 - 1) * limit
		}
	}
}

// Set sets value at position (i, j)
func (m *Matrix) Set(i, j int, val float64) {
	if i >= 0 && i < m.Rows && j >= 0 && j < m.Cols {
		m.Data[i][j] = val
	}
}

// Get gets value at position (i, j)
func (m *Matrix) Get(i, j int) float64 {
	if i >= 0 && i < m.Rows && j >= 0 && j < m.Cols {
		return m.Data[i][j]
	}
	return 0
}

// Multiply performs matrix multiplication
func (m *Matrix) Multiply(other *Matrix) *Matrix {
	if m.Cols != other.Rows {
		panic("Matrix dimensions don't match for multiplication")
	}

	result := New(m.Rows, other.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < other.Cols; j++ {
			sum := 0.0
			for k := 0; k < m.Cols; k++ {
				sum += m.Data[i][k] * other.Data[k][j]
			}
			result.Data[i][j] = sum
		}
	}
	return result
}

// Add performs element-wise addition
func (m *Matrix) Add(other *Matrix) *Matrix {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic("Matrix dimensions don't match for addition")
	}

	result := New(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] + other.Data[i][j]
		}
	}
	return result
}

// Scale multiplies all elements by a scalar
func (m *Matrix) Scale(scalar float64) *Matrix {
	result := New(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] * scalar
		}
	}
	return result
}

// Transpose returns the transpose of the matrix
func (m *Matrix) Transpose() *Matrix {
	result := New(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[j][i] = m.Data[i][j]
		}
	}
	return result
}

// Copy creates a deep copy of the matrix
func (m *Matrix) Copy() *Matrix {
	result := New(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		copy(result.Data[i], m.Data[i])
	}
	return result
}

// ApplyFunction applies a function to each element
func (m *Matrix) ApplyFunction(fn func(float64) float64) *Matrix {
	result := New(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = fn(m.Data[i][j])
		}
	}
	return result
}

// Softmax applies softmax function row-wise
func (m *Matrix) Softmax() *Matrix {
	result := New(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		// Find max for numerical stability
		max := m.Data[i][0]
		for j := 1; j < m.Cols; j++ {
			if m.Data[i][j] > max {
				max = m.Data[i][j]
			}
		}

		// Compute softmax
		sum := 0.0
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = math.Exp(m.Data[i][j] - max)
			sum += result.Data[i][j]
		}

		// Normalize
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] /= sum
		}
	}
	return result
}

// ReLU applies ReLU activation function
func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// Sigmoid applies sigmoid activation function
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Tanh applies hyperbolic tangent activation function
func Tanh(x float64) float64 {
	return math.Tanh(x)
}

// GeLU applies Gaussian Error Linear Unit activation function
func GeLU(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*math.Pow(x, 3))))
}

// LayerNorm applies layer normalization to each row
func (m *Matrix) LayerNorm(eps float64) *Matrix {
	result := New(m.Rows, m.Cols)

	for i := 0; i < m.Rows; i++ {
		// Calculate mean
		mean := 0.0
		for j := 0; j < m.Cols; j++ {
			mean += m.Data[i][j]
		}
		mean /= float64(m.Cols)

		// Calculate variance
		variance := 0.0
		for j := 0; j < m.Cols; j++ {
			diff := m.Data[i][j] - mean
			variance += diff * diff
		}
		variance /= float64(m.Cols)

		// Apply normalization
		stddev := math.Sqrt(variance + eps)
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = (m.Data[i][j] - mean) / stddev
		}
	}

	return result
}

// Dropout applies dropout with given probability (for training)
func (m *Matrix) Dropout(dropoutRate float64, training bool) *Matrix {
	if !training || dropoutRate == 0 {
		return m.Copy()
	}

	result := New(m.Rows, m.Cols)
	scale := 1.0 / (1.0 - dropoutRate)

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			if rand.Float64() > dropoutRate {
				result.Data[i][j] = m.Data[i][j] * scale
			} else {
				result.Data[i][j] = 0
			}
		}
	}

	return result
}

// ElementwiseMultiply performs element-wise multiplication (Hadamard product)
func (m *Matrix) ElementwiseMultiply(other *Matrix) *Matrix {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic("Matrix dimensions don't match for element-wise multiplication")
	}

	result := New(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] * other.Data[i][j]
		}
	}
	return result
}
