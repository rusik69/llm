package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/rusik69/llm/pkg/data"
	"github.com/rusik69/llm/pkg/llm"
	"github.com/rusik69/llm/pkg/tokenizer"
	"github.com/rusik69/llm/pkg/training"
)

// version is set by the build process
var version = "dev"

func main() {
	var (
		mode        = flag.String("mode", "train", "Mode: train, generate, or download")
		text        = flag.String("text", "", "Text to process (for generation)")
		modelDir    = flag.String("model", "./model", "Directory to save/load model")
		epochs      = flag.Int("epochs", 10, "Number of training epochs")
		lr          = flag.Float64("lr", 0.0001, "Learning rate")
		dataFile    = flag.String("data", "", "Path to training data file")
		dataDir     = flag.String("datadir", "./data", "Directory to store downloaded data")
		useSample   = flag.Bool("sample", false, "Use sample data instead of full Wikipedia dump")
		maxArticles = flag.Int("maxarticles", 1000, "Maximum number of articles to process")

		// Generation parameters
		maxTokens         = flag.Int("maxtokens", 50, "Maximum number of tokens to generate")
		temperature       = flag.Float64("temperature", 0.8, "Temperature for sampling (0.1-2.0)")
		topK              = flag.Int("topk", 50, "Top-K sampling parameter")
		topP              = flag.Float64("topp", 0.9, "Top-P (nucleus) sampling parameter")
		useGreedy         = flag.Bool("greedy", false, "Use greedy decoding instead of sampling")
		repetitionPenalty = flag.Float64("reppenalty", 1.2, "Repetition penalty (1.0 = no penalty)")
		noRepeatNGram     = flag.Int("norep", 3, "Prevent repeating n-grams of this size")

		// Utility flags
		showVersion = flag.Bool("version", false, "Show version information")
	)
	flag.Parse()

	if *showVersion {
		fmt.Printf("LLM version %s\n", version)
		fmt.Println("A simple Large Language Model implementation in Go")
		fmt.Println("GitHub: https://github.com/rusik69/llm")
		os.Exit(0)
	}

	switch *mode {
	case "download":
		fmt.Println("Downloading Wikipedia data...")
		if err := downloadWikipedia(*dataDir, *useSample, *maxArticles); err != nil {
			log.Fatal("Download failed:", err)
		}
		fmt.Println("Download completed!")

	case "train":
		fmt.Println("Starting LLM training...")
		if err := trainModel(*modelDir, *epochs, *lr, *dataFile); err != nil {
			log.Fatal("Training failed:", err)
		}
		fmt.Println("Training completed!")

	case "generate":
		if *text == "" {
			fmt.Println("Please provide text with -text flag")
			os.Exit(1)
		}
		fmt.Println("Generating text...")
		genConfig := &llm.GenerationConfig{
			MaxNewTokens:      *maxTokens,
			Temperature:       *temperature,
			TopK:              *topK,
			TopP:              *topP,
			DoSample:          !*useGreedy,
			RepetitionPenalty: *repetitionPenalty,
			NoRepeatNGramSize: *noRepeatNGram,
		}
		if err := generateText(*text, *modelDir, genConfig); err != nil {
			log.Fatal("Generation failed:", err)
		}

	default:
		fmt.Println("Invalid mode. Use 'train', 'generate', or 'download'")
		os.Exit(1)
	}
}

func downloadWikipedia(dataDir string, useSample bool, maxArticles int) error {
	downloader := data.NewWikipediaDownloader(dataDir, maxArticles)

	articles, err := downloader.ProcessWikipediaData(useSample)
	if err != nil {
		return fmt.Errorf("failed to process Wikipedia data: %v", err)
	}

	fmt.Printf("Successfully processed %d articles\n", len(articles))
	return nil
}

func trainModel(modelDir string, epochs int, lr float64, dataFile string) error {
	// Initialize tokenizer
	tokenizer := tokenizer.New()

	var tokens []int
	var err error

	if dataFile != "" {
		// Load training data from file
		tokens, err = training.LoadTrainingDataFromFile(dataFile, tokenizer)
		if err != nil {
			return fmt.Errorf("failed to load training data: %v", err)
		}
	} else {
		// Use default training data
		trainingText := `Hello world. This is a simple language model.
The quick brown fox jumps over the lazy dog.
Machine learning is fascinating and powerful.
Go is a great programming language for building systems.
Natural language processing enables computers to understand human language.
Artificial intelligence is transforming how we interact with technology.
Deep learning models can learn complex patterns from data.
Programming languages provide tools for creating software applications.`

		tokens = tokenizer.Tokenize(trainingText)
	}

	// Create model
	model := llm.New(len(tokenizer.Vocab()), 64, 128, 2) // vocab_size, embed_size, hidden_size, num_layers

	// Train model with saving
	trainer := training.New(model, lr)
	return trainer.TrainWithSave(tokens, epochs, modelDir, tokenizer)
}

func generateText(prompt string, modelDir string, config *llm.GenerationConfig) error {
	// Try to load existing model and tokenizer
	model, err := llm.LoadModel(modelDir)
	if err != nil {
		fmt.Printf("Could not load model from %s, using new model: %v\n", modelDir, err)
		// Fall back to creating a new model (for backward compatibility)
		fallbackTokenizer := tokenizer.New()
		model = llm.New(len(fallbackTokenizer.Vocab()), 64, 128, 2)
	}

	tok, err := tokenizer.LoadTokenizer(modelDir)
	if err != nil {
		fmt.Printf("Could not load tokenizer from %s, using new tokenizer: %v\n", modelDir, err)
		tok = tokenizer.New()
	}

	// Tokenize prompt
	tokens := tok.Tokenize(prompt)

	// Generate with configuration
	generated := model.GenerateWithConfig(tokens, config)

	// Convert back to text
	result := tok.Detokenize(generated)

	fmt.Printf("Input: %s\n", prompt)
	fmt.Printf("Generated: %s\n", result)
	fmt.Printf("Generation config: temp=%.2f, top_k=%d, top_p=%.2f, sampling=%t, rep_penalty=%.2f, no_repeat_ngram=%d\n",
		config.Temperature, config.TopK, config.TopP, config.DoSample, config.RepetitionPenalty, config.NoRepeatNGramSize)

	return nil
}
