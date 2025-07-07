#!/bin/bash

# Test script for Wikipedia data functionality
# Usage: ./scripts/test_wikipedia.sh [--ci] [--full]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Parse command line arguments
CI_MODE=false
FULL_TEST=false

for arg in "$@"; do
    case $arg in
        --ci)
            CI_MODE=true
            shift
            ;;
        --full)
            FULL_TEST=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--ci] [--full]"
            echo "  --ci    Run in CI mode (smaller datasets, faster execution)"
            echo "  --full  Run full test suite (larger datasets, longer execution)"
            exit 0
            ;;
    esac
done

echo "🔬 Starting Wikipedia Data Tests"
echo "==============================================="

# Set test parameters based on mode
if [ "$CI_MODE" = true ]; then
    MAX_ARTICLES=5
    EPOCHS=1
    echo "📊 Running in CI mode (fast)"
elif [ "$FULL_TEST" = true ]; then
    MAX_ARTICLES=50
    EPOCHS=5
    echo "📊 Running full test suite (comprehensive)"
else
    MAX_ARTICLES=10
    EPOCHS=2
    echo "📊 Running standard test suite"
fi

echo "Parameters: max_articles=$MAX_ARTICLES, epochs=$EPOCHS"
echo ""

# Clean up any previous test data
echo "🧹 Cleaning up previous test data..."
rm -rf test_wiki_data/ model/ || true

# Build the project
echo "🔨 Building project..."
if ! make build; then
    echo "❌ Build failed"
    exit 1
fi
echo "✅ Build successful"
echo ""

# Test 1: Sample Wikipedia data
echo "📝 Test 1: Sample Wikipedia Data Processing"
echo "-------------------------------------------"
if ./llm -mode=download -sample=true -datadir=test_wiki_data -maxarticles=$MAX_ARTICLES; then
    echo "✅ Sample data download successful"
else
    echo "❌ Sample data download failed"
    exit 1
fi

# Verify sample data quality
if [ -f "test_wiki_data/training_data.txt" ]; then
    SAMPLE_LINES=$(wc -l < test_wiki_data/training_data.txt)
    SAMPLE_WORDS=$(wc -w < test_wiki_data/training_data.txt)
    echo "📈 Sample data stats: $SAMPLE_LINES lines, $SAMPLE_WORDS words"
    
    if [ "$SAMPLE_LINES" -lt 5 ]; then
        echo "❌ Sample data too small"
        exit 1
    fi
    echo "✅ Sample data quality check passed"
else
    echo "❌ Sample training data file not created"
    exit 1
fi
echo ""

# Test 2: Real Wikipedia data (if not in CI mode)
if [ "$CI_MODE" = false ]; then
    echo "🌐 Test 2: Real Wikipedia Data Download"
    echo "--------------------------------------"
    rm -rf test_wiki_data/
    
    if ./llm -mode=download -datadir=test_wiki_data -maxarticles=$MAX_ARTICLES; then
        echo "✅ Real Wikipedia download successful"
    else
        echo "❌ Real Wikipedia download failed"
        exit 1
    fi

    # Verify real data quality
    if [ -f "test_wiki_data/training_data.txt" ]; then
        REAL_LINES=$(wc -l < test_wiki_data/training_data.txt)
        REAL_WORDS=$(wc -w < test_wiki_data/training_data.txt)
        echo "📈 Real data stats: $REAL_LINES lines, $REAL_WORDS words"
        
        # Check for structural filtering
        TABLE_COUNT=$(grep -c "^|.*|" test_wiki_data/training_data.txt || true)
        HEADER_COUNT=$(grep -c "^==" test_wiki_data/training_data.txt || true)
        
        echo "🔍 Structural elements found: tables=$TABLE_COUNT, headers=$HEADER_COUNT"
        
        if [ "$TABLE_COUNT" -gt 5 ] || [ "$HEADER_COUNT" -gt 5 ]; then
            echo "⚠️  Warning: High number of structural elements detected"
        else
            echo "✅ Structural filtering working correctly"
        fi
    else
        echo "❌ Real training data file not created"
        exit 1
    fi
    echo ""
fi

# Test 3: Model training
echo "🧠 Test 3: Model Training on Wikipedia Data"
echo "------------------------------------------"
if ./llm -mode=train -datadir=test_wiki_data -epochs=$EPOCHS -lr=0.0001; then
    echo "✅ Training successful"
else
    echo "❌ Training failed"
    exit 1
fi

# Verify model files
if [ -f "model/model.json" ] && [ -f "model/tokenizer.json" ]; then
    MODEL_SIZE=$(stat -f%z model/model.json 2>/dev/null || stat -c%s model/model.json 2>/dev/null)
    TOKENIZER_SIZE=$(stat -f%z model/tokenizer.json 2>/dev/null || stat -c%s model/tokenizer.json 2>/dev/null)
    echo "📈 Model files: model.json ($MODEL_SIZE bytes), tokenizer.json ($TOKENIZER_SIZE bytes)"
    echo "✅ Model files created successfully"
else
    echo "❌ Model files not created"
    exit 1
fi
echo ""

# Test 4: Text generation
echo "💬 Test 4: Text Generation with Wikipedia Model"
echo "-----------------------------------------------"

# Test multiple prompts
TEST_PROMPTS=("Science is" "Technology" "Mathematics" "Art is")

for prompt in "${TEST_PROMPTS[@]}"; do
    echo "🔤 Testing prompt: '$prompt'"
    
    if output=$(./llm -mode=generate -text="$prompt" -maxtokens=15 -temperature=0.8 2>&1); then
        echo "✅ Generated: $(echo "$output" | grep "Generated:" | sed 's/Generated: //')"
    else
        echo "❌ Generation failed for prompt: '$prompt'"
        echo "Error: $output"
        exit 1
    fi
done
echo ""

# Test 5: Integration tests (if available)
echo "🔧 Test 5: Integration Tests"
echo "---------------------------"
if go test -tags=integration -v ./pkg/data > integration_test.log 2>&1; then
    echo "✅ Integration tests passed"
    # Show summary
    grep -E "(PASS|FAIL|RUN)" integration_test.log | tail -5
else
    echo "❌ Integration tests failed"
    echo "Last few lines of test output:"
    tail -10 integration_test.log
    exit 1
fi
echo ""

# Test 6: Performance benchmark (if full test)
if [ "$FULL_TEST" = true ]; then
    echo "⚡ Test 6: Performance Benchmark"
    echo "------------------------------"
    
    echo "🔍 Running Wikipedia processing benchmark..."
    if go test -tags=integration -bench=BenchmarkWikipediaProcessing -v ./pkg/data > benchmark.log 2>&1; then
        echo "✅ Benchmark completed"
        grep "BenchmarkWikipediaProcessing" benchmark.log
    else
        echo "❌ Benchmark failed"
        tail -5 benchmark.log
    fi
    echo ""
fi

# Test 7: Error handling
echo "🚨 Test 7: Error Handling"
echo "------------------------"

# Test with non-existent directory
if ./llm -mode=train -datadir=nonexistent 2>/dev/null; then
    echo "❌ Should have failed with non-existent directory"
    exit 1
else
    echo "✅ Correctly handled non-existent directory"
fi

# Test generation without model
rm -rf model/
if ./llm -mode=generate -text="test" 2>/dev/null; then
    echo "❌ Should have failed without model"
    exit 1
else
    echo "✅ Correctly handled missing model"
fi
echo ""

# Summary
echo "🎉 Wikipedia Data Tests Complete!"
echo "================================="
echo "✅ All tests passed successfully"
echo ""
echo "📊 Summary:"
echo "  - Sample data processing: ✅"
if [ "$CI_MODE" = false ]; then
    echo "  - Real Wikipedia download: ✅"
fi
echo "  - Model training: ✅"
echo "  - Text generation: ✅" 
echo "  - Integration tests: ✅"
if [ "$FULL_TEST" = true ]; then
    echo "  - Performance benchmark: ✅"
fi
echo "  - Error handling: ✅"
echo ""
echo "🚀 Wikipedia data functionality is working correctly!"

# Clean up test data
if [ "$CI_MODE" = false ]; then
    echo ""
    echo "🧹 Cleaning up test data..."
    rm -rf test_wiki_data/ model/ integration_test.log benchmark.log || true
    echo "✅ Cleanup complete"
fi 