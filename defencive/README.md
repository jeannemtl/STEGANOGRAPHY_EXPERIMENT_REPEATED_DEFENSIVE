# Steganographic AI Detection Research

## Overview
This repository contains code for detecting and analyzing steganographic patterns in AI reasoning outputs. 
The work demonstrates that different steganographic encoding methods create distinguishable neural signatures in language model PCA analysis representations.

## Repository Structure

### Core Components

**detector.py**
- BERT dual classifier (token + sentence level)
- Pattern detection algorithms for identifying steganographic signals
- Methods: alternating patterns, length analysis, token variance detection

**visualize.py**
- Neural representation extraction from BERT hidden states
- PCA dimensionality reduction (768D → 3D)
- Animated and static visualizations of representation clustering
- Temporal analysis across reasoning steps

### Data Generation

**steganographic_dataset_generator.py**
- Generates 5 pattern types of basic steganographic types:
  - Normal reasoning (baseline)
  - Step/Now alternation (spatial encoding)
  - First/Then binary (spatial encoding)
  - Word length patterns (linguistic density)
  - Punctuation patterns (syntactic markers)

**generator-am.py**
- Generates specialized periodic patterning steganographic types
- Amplitude Modulation (AM) inspired steganography
- Encodes messages: HELLO, SECRET, AI_RISK

**generate.py**
- Main script to generate the standard steganographic dataset

### Generated Data
- `large_scale_steganographic_dataset.json` - 900 examples across 5 pattern types
- `am_steganographic_data.json` - 3 AM periodic pattern encoded messages:
  - HELLO (5 chars, 40 binary bits, 40 reasoning steps)
  - SECRET (6 chars, 48 binary bits, 40 reasoning steps)  
  - AI_RISK (7 chars, 56 binary bits, 40 reasoning steps)

## Usage

### 1. Generate Datasets
```bash
python3 generate.py
python3 generator-am.py
