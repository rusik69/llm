package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Tokenizer handles text tokenization and vocabulary management
type Tokenizer struct {
	wordToID map[string]int
	idToWord map[int]string
	nextID   int
}

// Special tokens
const (
	UNK_TOKEN = "<UNK>"
	PAD_TOKEN = "<PAD>"
	BOS_TOKEN = "<BOS>" // Beginning of sequence
	EOS_TOKEN = "<EOS>" // End of sequence
)

// TokenizerCheckpoint represents a saved tokenizer state
type TokenizerCheckpoint struct {
	WordToID map[string]int `json:"word_to_id"`
	IDToWord map[int]string `json:"id_to_word"`
	NextID   int            `json:"next_id"`
}

// New creates a new tokenizer with special tokens
func New() *Tokenizer {
	t := &Tokenizer{
		wordToID: make(map[string]int),
		idToWord: make(map[int]string),
		nextID:   0,
	}

	// Add special tokens
	t.addToken(PAD_TOKEN)
	t.addToken(UNK_TOKEN)
	t.addToken(BOS_TOKEN)
	t.addToken(EOS_TOKEN)

	return t
}

// addToken adds a token to the vocabulary
func (t *Tokenizer) addToken(token string) int {
	if id, exists := t.wordToID[token]; exists {
		return id
	}

	id := t.nextID
	t.wordToID[token] = id
	t.idToWord[id] = token
	t.nextID++

	return id
}

// Tokenize converts text to token IDs
func (t *Tokenizer) Tokenize(text string) []int {
	// Simple tokenization: split by whitespace and punctuation
	words := t.preprocessText(text)

	tokens := []int{t.wordToID[BOS_TOKEN]} // Start with BOS token

	for _, word := range words {
		if word == "" {
			continue
		}

		id := t.addToken(word) // Add to vocab if not exists
		tokens = append(tokens, id)
	}

	tokens = append(tokens, t.wordToID[EOS_TOKEN]) // End with EOS token

	return tokens
}

// preprocessText cleans and splits text into words
func (t *Tokenizer) preprocessText(text string) []string {
	// Convert to lowercase
	text = strings.ToLower(text)

	// Split on whitespace and punctuation but keep punctuation
	re := regexp.MustCompile(`\w+|[^\w\s]`)
	words := re.FindAllString(text, -1)

	return words
}

// Detokenize converts token IDs back to text
func (t *Tokenizer) Detokenize(tokens []int) string {
	var words []string

	for _, tokenID := range tokens {
		if word, exists := t.idToWord[tokenID]; exists {
			// Skip special tokens in output except for readability
			if word == BOS_TOKEN || word == EOS_TOKEN || word == PAD_TOKEN {
				continue
			}
			words = append(words, word)
		} else {
			words = append(words, UNK_TOKEN)
		}
	}

	// Join words with spaces, but handle punctuation
	result := ""
	for i, word := range words {
		if i > 0 && !isPunctuation(word) {
			result += " "
		}
		result += word
	}

	return result
}

// isPunctuation checks if a word is punctuation
func isPunctuation(word string) bool {
	if len(word) != 1 {
		return false
	}

	punctuation := ".,!?;:"
	return strings.Contains(punctuation, word)
}

// GetTokenID returns the ID for a token, or UNK if not found
func (t *Tokenizer) GetTokenID(token string) int {
	if id, exists := t.wordToID[token]; exists {
		return id
	}
	return t.wordToID[UNK_TOKEN]
}

// GetToken returns the token for an ID
func (t *Tokenizer) GetToken(id int) string {
	if token, exists := t.idToWord[id]; exists {
		return token
	}
	return UNK_TOKEN
}

// Vocab returns the vocabulary map
func (t *Tokenizer) Vocab() map[string]int {
	return t.wordToID
}

// VocabSize returns the size of the vocabulary
func (t *Tokenizer) VocabSize() int {
	return len(t.wordToID)
}

// PadSequence pads or truncates a sequence to a specific length
func (t *Tokenizer) PadSequence(tokens []int, maxLen int) []int {
	padID := t.wordToID[PAD_TOKEN]

	if len(tokens) > maxLen {
		return tokens[:maxLen]
	}

	result := make([]int, maxLen)
	copy(result, tokens)

	// Fill remaining with pad tokens
	for i := len(tokens); i < maxLen; i++ {
		result[i] = padID
	}

	return result
}

// SaveTokenizer saves the tokenizer vocabulary to disk
func (t *Tokenizer) SaveTokenizer(modelDir string) error {
	// Create model directory
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return fmt.Errorf("failed to create model directory: %v", err)
	}

	checkpoint := &TokenizerCheckpoint{
		WordToID: t.wordToID,
		IDToWord: t.idToWord,
		NextID:   t.nextID,
	}

	filePath := filepath.Join(modelDir, "tokenizer.json")
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create tokenizer file: %v", err)
	}
	defer func() {
		if err := file.Close(); err != nil {
			fmt.Printf("Warning: failed to close tokenizer file: %v\n", err)
		}
	}()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(checkpoint); err != nil {
		return fmt.Errorf("failed to encode tokenizer: %v", err)
	}

	fmt.Printf("Tokenizer saved to %s\n", filePath)
	return nil
}

// LoadTokenizer loads a tokenizer from disk
func LoadTokenizer(modelDir string) (*Tokenizer, error) {
	filePath := filepath.Join(modelDir, "tokenizer.json")

	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open tokenizer file: %v", err)
	}
	defer func() {
		if err := file.Close(); err != nil {
			fmt.Printf("Warning: failed to close tokenizer file: %v\n", err)
		}
	}()

	var checkpoint TokenizerCheckpoint
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&checkpoint); err != nil {
		return nil, fmt.Errorf("failed to decode tokenizer: %v", err)
	}

	tokenizer := &Tokenizer{
		wordToID: checkpoint.WordToID,
		idToWord: checkpoint.IDToWord,
		nextID:   checkpoint.NextID,
	}

	fmt.Printf("Tokenizer loaded from %s\n", filePath)
	return tokenizer, nil
}
