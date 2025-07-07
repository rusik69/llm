package tokenizer

import (
	"testing"
)

func TestNew(t *testing.T) {
	tokenizer := New()

	// Check that special tokens are present
	if tokenizer.GetTokenID(PAD_TOKEN) != 0 {
		t.Errorf("Expected PAD_TOKEN to have ID 0, got %d", tokenizer.GetTokenID(PAD_TOKEN))
	}
	if tokenizer.GetTokenID(UNK_TOKEN) != 1 {
		t.Errorf("Expected UNK_TOKEN to have ID 1, got %d", tokenizer.GetTokenID(UNK_TOKEN))
	}
	if tokenizer.GetTokenID(BOS_TOKEN) != 2 {
		t.Errorf("Expected BOS_TOKEN to have ID 2, got %d", tokenizer.GetTokenID(BOS_TOKEN))
	}
	if tokenizer.GetTokenID(EOS_TOKEN) != 3 {
		t.Errorf("Expected EOS_TOKEN to have ID 3, got %d", tokenizer.GetTokenID(EOS_TOKEN))
	}

	// Check initial vocabulary size
	if tokenizer.VocabSize() != 4 {
		t.Errorf("Expected initial vocab size 4, got %d", tokenizer.VocabSize())
	}
}

func TestTokenize(t *testing.T) {
	tokenizer := New()

	text := "Hello world!"
	tokens := tokenizer.Tokenize(text)

	// Should start with BOS and end with EOS
	if tokens[0] != tokenizer.GetTokenID(BOS_TOKEN) {
		t.Errorf("Expected first token to be BOS, got %d", tokens[0])
	}
	if tokens[len(tokens)-1] != tokenizer.GetTokenID(EOS_TOKEN) {
		t.Errorf("Expected last token to be EOS, got %d", tokens[len(tokens)-1])
	}

	// Check that tokens are added to vocabulary
	if tokenizer.VocabSize() < 6 { // At least BOS, EOS, PAD, UNK + hello + world + !
		t.Errorf("Expected vocab size to increase after tokenization, got %d", tokenizer.VocabSize())
	}
}

func TestTokenizeEmpty(t *testing.T) {
	tokenizer := New()

	tokens := tokenizer.Tokenize("")

	// Should only have BOS and EOS
	if len(tokens) != 2 {
		t.Errorf("Expected 2 tokens for empty string, got %d", len(tokens))
	}
	if tokens[0] != tokenizer.GetTokenID(BOS_TOKEN) {
		t.Errorf("Expected first token to be BOS, got %d", tokens[0])
	}
	if tokens[1] != tokenizer.GetTokenID(EOS_TOKEN) {
		t.Errorf("Expected second token to be EOS, got %d", tokens[1])
	}
}

func TestDetokenize(t *testing.T) {
	tokenizer := New()

	text := "Hello world!"
	tokens := tokenizer.Tokenize(text)
	result := tokenizer.Detokenize(tokens)

	// Should get back similar text (lowercase and possibly different punctuation handling)
	expected := "hello world!"
	if result != expected {
		t.Errorf("Expected '%s', got '%s'", expected, result)
	}
}

func TestGetTokenID(t *testing.T) {
	tokenizer := New()

	// Add a token
	tokenizer.Tokenize("test")

	// Should find the token
	testID := tokenizer.GetTokenID("test")
	if testID == tokenizer.GetTokenID(UNK_TOKEN) {
		t.Errorf("Expected 'test' to have its own ID, got UNK")
	}

	// Should return UNK for unknown token
	unknownID := tokenizer.GetTokenID("nonexistent")
	if unknownID != tokenizer.GetTokenID(UNK_TOKEN) {
		t.Errorf("Expected unknown token to return UNK ID, got %d", unknownID)
	}
}

func TestGetToken(t *testing.T) {
	tokenizer := New()

	// Test special tokens
	if tokenizer.GetToken(0) != PAD_TOKEN {
		t.Errorf("Expected PAD_TOKEN at ID 0, got %s", tokenizer.GetToken(0))
	}
	if tokenizer.GetToken(1) != UNK_TOKEN {
		t.Errorf("Expected UNK_TOKEN at ID 1, got %s", tokenizer.GetToken(1))
	}

	// Test invalid ID
	if tokenizer.GetToken(9999) != UNK_TOKEN {
		t.Errorf("Expected UNK_TOKEN for invalid ID, got %s", tokenizer.GetToken(9999))
	}
}

func TestPreprocessText(t *testing.T) {
	tokenizer := New()

	text := "Hello World! This is a test."
	words := tokenizer.preprocessText(text)

	// Should split into words and punctuation
	expected := []string{"hello", "world", "!", "this", "is", "a", "test", "."}

	if len(words) != len(expected) {
		t.Errorf("Expected %d words, got %d", len(expected), len(words))
	}

	for i, word := range words {
		if i < len(expected) && word != expected[i] {
			t.Errorf("Expected word %d to be '%s', got '%s'", i, expected[i], word)
		}
	}
}

func TestPadSequence(t *testing.T) {
	tokenizer := New()

	tokens := []int{1, 2, 3}

	// Test padding
	padded := tokenizer.PadSequence(tokens, 5)
	if len(padded) != 5 {
		t.Errorf("Expected length 5, got %d", len(padded))
	}

	// First 3 should be original
	for i := 0; i < 3; i++ {
		if padded[i] != tokens[i] {
			t.Errorf("Expected %d at position %d, got %d", tokens[i], i, padded[i])
		}
	}

	// Last 2 should be PAD tokens
	padID := tokenizer.GetTokenID(PAD_TOKEN)
	for i := 3; i < 5; i++ {
		if padded[i] != padID {
			t.Errorf("Expected PAD token at position %d, got %d", i, padded[i])
		}
	}

	// Test truncation
	truncated := tokenizer.PadSequence(tokens, 2)
	if len(truncated) != 2 {
		t.Errorf("Expected length 2, got %d", len(truncated))
	}
	if truncated[0] != tokens[0] || truncated[1] != tokens[1] {
		t.Errorf("Expected first 2 tokens to be preserved")
	}
}

func TestVocab(t *testing.T) {
	tokenizer := New()

	tokenizer.Tokenize("hello world")

	vocab := tokenizer.Vocab()

	// Should contain special tokens
	if _, exists := vocab[PAD_TOKEN]; !exists {
		t.Errorf("Vocab should contain PAD_TOKEN")
	}
	if _, exists := vocab[UNK_TOKEN]; !exists {
		t.Errorf("Vocab should contain UNK_TOKEN")
	}
	if _, exists := vocab[BOS_TOKEN]; !exists {
		t.Errorf("Vocab should contain BOS_TOKEN")
	}
	if _, exists := vocab[EOS_TOKEN]; !exists {
		t.Errorf("Vocab should contain EOS_TOKEN")
	}

	// Should contain added words
	if _, exists := vocab["hello"]; !exists {
		t.Errorf("Vocab should contain 'hello'")
	}
	if _, exists := vocab["world"]; !exists {
		t.Errorf("Vocab should contain 'world'")
	}
}

func TestVocabSize(t *testing.T) {
	tokenizer := New()

	initialSize := tokenizer.VocabSize()

	tokenizer.Tokenize("new word")

	newSize := tokenizer.VocabSize()

	if newSize <= initialSize {
		t.Errorf("Expected vocab size to increase from %d to %d", initialSize, newSize)
	}
}

func TestIsPunctuation(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
	}{
		{".", true},
		{",", true},
		{"!", true},
		{"?", true},
		{";", true},
		{":", true},
		{"a", false},
		{"hello", false},
		{"", false},
		{"ab", false},
	}

	for _, test := range tests {
		result := isPunctuation(test.input)
		if result != test.expected {
			t.Errorf("isPunctuation('%s') = %v, expected %v", test.input, result, test.expected)
		}
	}
}

func TestConsistentTokenization(t *testing.T) {
	tokenizer := New()

	text := "Hello world! This is a test."

	// Tokenize twice
	tokens1 := tokenizer.Tokenize(text)
	tokens2 := tokenizer.Tokenize(text)

	// Should get same tokens
	if len(tokens1) != len(tokens2) {
		t.Errorf("Inconsistent tokenization lengths: %d vs %d", len(tokens1), len(tokens2))
	}

	for i := 0; i < len(tokens1) && i < len(tokens2); i++ {
		if tokens1[i] != tokens2[i] {
			t.Errorf("Inconsistent tokenization at position %d: %d vs %d", i, tokens1[i], tokens2[i])
		}
	}
}

func TestRoundTripTokenization(t *testing.T) {
	tokenizer := New()

	text := "hello world"
	tokens := tokenizer.Tokenize(text)
	result := tokenizer.Detokenize(tokens)

	if result != text {
		t.Errorf("Round-trip failed: '%s' -> '%s'", text, result)
	}
}
