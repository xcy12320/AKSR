<div align="center">

# AKSR: Adaptive Knowledge Selection and Multi-Strategy Reasoning for Alzheimer's Disease Question Answering

</div>

## Overview

<p align="center">
  <img src="framework.png" width="750" title="Overview of our framework" alt="">
</p>

## Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

## Configuration

### 1. Environment Setup

Copy the example configuration file:

```bash
cp .env.example .env
```

### 2. Configure API Keys and Database

Edit `.env` file and fill in your credentials:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# UMLS API Configuration
UMLS_API_KEY=your_umls_api_key_here

# Neo4j Database Configuration
NEO4J_URI=bolt://localhost:7688
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here

# Ablation Configuration (optional)
ABLATION_CONFIG=full_model
```

### 3. Prepare Knowledge Graph Data

Place your knowledge graph triples in:
```
./Alzheimers/train_s2s.txt
```

Format: `head\trelation\ttail` (tab-separated)

## Running the Code

### Basic Execution

```bash
python Main.py
```

The script will:
1. Initialize Neo4j knowledge graph database
2. Load entity embeddings and keyword embeddings
3. Process medical questions from datasets
4. Generate answers using KG-enhanced reasoning

### Output

Results will be saved in:
```
./Alzheimers/result_chatgpt_mindmap/
```

Files generated:
- `{dataset}_{config}_ablation_results.json` - Detailed results
- `{dataset}_{config}_performance_stats.json` - Performance statistics
- `ablation_experiment_report.json` - Overall experiment summary

## Required Data Files

Ensure the following files exist:
```
./Alzheimers/
├── train_s2s.txt              # Knowledge graph triples
├── entity_embeddings.pkl       # Entity embeddings
├── keyword_embeddings.pkl      # Keyword embeddings
└── result_filter/              # Dataset questions (optional)
```

## Notes

- First run will initialize the Neo4j database (may take some time)
- The system uses file locks to prevent concurrent initialization
- API rate limits may affect processing speed
- Results are cached to improve performance

## Troubleshooting

**Q: "UMLS API key not set" error**
- Make sure you've set `UMLS_API_KEY` in your `.env` file

**Q: Neo4j connection failed**
- Verify Neo4j is running on the specified URI
- Check username and password are correct

**Q: Out of memory errors**
- Adjust `CLEANUP_FREQUENCY` and `MAX_CACHE_SIZE` in the code
- Use a machine with more RAM for large datasets

