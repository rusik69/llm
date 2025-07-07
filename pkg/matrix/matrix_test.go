package matrix

import (
	"math"
	"testing"
)

func TestNew(t *testing.T) {
	m := New(3, 4)

	if m.Rows != 3 {
		t.Errorf("Expected 3 rows, got %d", m.Rows)
	}
	if m.Cols != 4 {
		t.Errorf("Expected 4 columns, got %d", m.Cols)
	}

	// Check all elements are zero
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			if m.Data[i][j] != 0 {
				t.Errorf("Expected zero at [%d][%d], got %f", i, j, m.Data[i][j])
			}
		}
	}
}

func TestNewFromData(t *testing.T) {
	data := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}

	m := NewFromData(data)

	if m.Rows != 2 {
		t.Errorf("Expected 2 rows, got %d", m.Rows)
	}
	if m.Cols != 3 {
		t.Errorf("Expected 3 columns, got %d", m.Cols)
	}

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			if m.Data[i][j] != data[i][j] {
				t.Errorf("Expected %f at [%d][%d], got %f", data[i][j], i, j, m.Data[i][j])
			}
		}
	}
}

func TestSetGet(t *testing.T) {
	m := New(2, 2)

	m.Set(0, 0, 5.5)
	m.Set(1, 1, 3.14)

	if m.Get(0, 0) != 5.5 {
		t.Errorf("Expected 5.5, got %f", m.Get(0, 0))
	}
	if m.Get(1, 1) != 3.14 {
		t.Errorf("Expected 3.14, got %f", m.Get(1, 1))
	}

	// Test out of bounds
	m.Set(-1, 0, 99)
	m.Set(0, -1, 99)
	m.Set(2, 0, 99)
	m.Set(0, 2, 99)

	if m.Get(-1, 0) != 0 {
		t.Errorf("Expected 0 for out of bounds, got %f", m.Get(-1, 0))
	}
	if m.Get(2, 0) != 0 {
		t.Errorf("Expected 0 for out of bounds, got %f", m.Get(2, 0))
	}
}

func TestMultiply(t *testing.T) {
	// Test 2x3 * 3x2 = 2x2
	a := NewFromData([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})

	b := NewFromData([][]float64{
		{7, 8},
		{9, 10},
		{11, 12},
	})

	result := a.Multiply(b)

	// Expected: [58, 64; 139, 154]
	expected := [][]float64{
		{58, 64},
		{139, 154},
	}

	if result.Rows != 2 || result.Cols != 2 {
		t.Errorf("Expected 2x2 result, got %dx%d", result.Rows, result.Cols)
	}

	for i := 0; i < result.Rows; i++ {
		for j := 0; j < result.Cols; j++ {
			if result.Data[i][j] != expected[i][j] {
				t.Errorf("Expected %f at [%d][%d], got %f", expected[i][j], i, j, result.Data[i][j])
			}
		}
	}
}

func TestAdd(t *testing.T) {
	a := NewFromData([][]float64{
		{1, 2},
		{3, 4},
	})

	b := NewFromData([][]float64{
		{5, 6},
		{7, 8},
	})

	result := a.Add(b)

	expected := [][]float64{
		{6, 8},
		{10, 12},
	}

	for i := 0; i < result.Rows; i++ {
		for j := 0; j < result.Cols; j++ {
			if result.Data[i][j] != expected[i][j] {
				t.Errorf("Expected %f at [%d][%d], got %f", expected[i][j], i, j, result.Data[i][j])
			}
		}
	}
}

func TestScale(t *testing.T) {
	m := NewFromData([][]float64{
		{1, 2},
		{3, 4},
	})

	result := m.Scale(2.5)

	expected := [][]float64{
		{2.5, 5.0},
		{7.5, 10.0},
	}

	for i := 0; i < result.Rows; i++ {
		for j := 0; j < result.Cols; j++ {
			if result.Data[i][j] != expected[i][j] {
				t.Errorf("Expected %f at [%d][%d], got %f", expected[i][j], i, j, result.Data[i][j])
			}
		}
	}
}

func TestTranspose(t *testing.T) {
	m := NewFromData([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})

	result := m.Transpose()

	expected := [][]float64{
		{1, 4},
		{2, 5},
		{3, 6},
	}

	if result.Rows != 3 || result.Cols != 2 {
		t.Errorf("Expected 3x2 result, got %dx%d", result.Rows, result.Cols)
	}

	for i := 0; i < result.Rows; i++ {
		for j := 0; j < result.Cols; j++ {
			if result.Data[i][j] != expected[i][j] {
				t.Errorf("Expected %f at [%d][%d], got %f", expected[i][j], i, j, result.Data[i][j])
			}
		}
	}
}

func TestApplyFunction(t *testing.T) {
	m := NewFromData([][]float64{
		{-1, 2},
		{-3, 4},
	})

	result := m.ApplyFunction(ReLU)

	expected := [][]float64{
		{0, 2},
		{0, 4},
	}

	for i := 0; i < result.Rows; i++ {
		for j := 0; j < result.Cols; j++ {
			if result.Data[i][j] != expected[i][j] {
				t.Errorf("Expected %f at [%d][%d], got %f", expected[i][j], i, j, result.Data[i][j])
			}
		}
	}
}

func TestSoftmax(t *testing.T) {
	m := NewFromData([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})

	result := m.Softmax()

	// Check that each row sums to 1
	for i := 0; i < result.Rows; i++ {
		sum := 0.0
		for j := 0; j < result.Cols; j++ {
			sum += result.Data[i][j]
		}
		if math.Abs(sum-1.0) > 1e-10 {
			t.Errorf("Row %d sum is %f, expected 1.0", i, sum)
		}
	}

	// Check that probabilities are positive
	for i := 0; i < result.Rows; i++ {
		for j := 0; j < result.Cols; j++ {
			if result.Data[i][j] <= 0 {
				t.Errorf("Probability at [%d][%d] is %f, expected positive", i, j, result.Data[i][j])
			}
		}
	}
}

func TestActivationFunctions(t *testing.T) {
	// Test ReLU
	if ReLU(-5) != 0 {
		t.Errorf("ReLU(-5) = %f, expected 0", ReLU(-5))
	}
	if ReLU(5) != 5 {
		t.Errorf("ReLU(5) = %f, expected 5", ReLU(5))
	}

	// Test Sigmoid
	sig0 := Sigmoid(0)
	if math.Abs(sig0-0.5) > 1e-10 {
		t.Errorf("Sigmoid(0) = %f, expected 0.5", sig0)
	}

	// Test Tanh
	tanh0 := Tanh(0)
	if math.Abs(tanh0-0.0) > 1e-10 {
		t.Errorf("Tanh(0) = %f, expected 0.0", tanh0)
	}
}

func TestCopy(t *testing.T) {
	original := NewFromData([][]float64{
		{1, 2},
		{3, 4},
	})

	copy := original.Copy()

	// Modify original
	original.Set(0, 0, 999)

	// Check that copy is unchanged
	if copy.Get(0, 0) != 1 {
		t.Errorf("Copy was modified when original changed")
	}
}

func TestRandom(t *testing.T) {
	m := New(10, 10)
	m.Random()

	// Check that not all values are zero
	nonZeroCount := 0
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			if m.Data[i][j] != 0 {
				nonZeroCount++
			}
		}
	}

	if nonZeroCount < 90 { // Expect most values to be non-zero
		t.Errorf("Random initialization produced too many zeros: %d/%d", nonZeroCount, 100)
	}
}

func TestXavier(t *testing.T) {
	m := New(10, 10)
	m.Xavier()

	// Check that not all values are zero
	nonZeroCount := 0
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			if m.Data[i][j] != 0 {
				nonZeroCount++
			}
		}
	}

	if nonZeroCount < 90 { // Expect most values to be non-zero
		t.Errorf("Xavier initialization produced too many zeros: %d/%d", nonZeroCount, 100)
	}
}
