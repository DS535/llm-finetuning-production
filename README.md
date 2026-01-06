# LLM Finetuning Production Project

A comprehensive, production-ready LLM finetuning educational project optimized for Google Colab with 8GB GPU VRAM. Learn multiple finetuning techniques (LoRA, QLoRA, Full Finetuning, Prefix Tuning) across various tasks (Instruction following, Text classification, Question answering, Text generation) with rigorous evaluation frameworks.

## 🎯 Project Overview

This project provides hands-on experience with:
- **4 Finetuning Techniques**: Full Finetuning, LoRA, QLoRA, Prefix Tuning
- **4 Task Types**: Instruction/Chat, Classification, Question Answering, Text Generation
- **Production-Ready Code**: Modular, reusable, memory-optimized components
- **Comprehensive Evaluation**: Automated metrics, base vs finetuned comparisons
- **Real-World Skills**: Industry-standard HuggingFace stack

## 🚀 Quick Start

### Local Setup (Windows)

1. **Clone this repository**:
```bash
git clone https://github.com/YOUR_USERNAME/llm-finetuning-production.git
cd llm-finetuning-production
```

2. **Prepare for Colab**: All code is ready to run in Google Colab (no local Python environment needed)

### Google Colab Setup

1. **Open Google Colab**: [colab.research.google.com](https://colab.research.google.com)

2. **Start with the first notebook**:
   - Upload or clone this repo in Colab
   - Open `notebooks/01_foundations/01_environment_setup.ipynb`
   - This notebook will:
     - Install all dependencies
     - Mount Google Drive
     - Verify GPU access
     - Configure experiment tracking

3. **Execute notebooks in sequence** (01 → 16)

## 📚 Notebook Structure

### Phase 1: Foundations (Week 1)
- **01_environment_setup.ipynb** - Colab setup, GPU validation
- **02_data_exploration.ipynb** - Dataset loading & EDA
- **03_baseline_evaluation.ipynb** - Zero-shot baseline metrics

### Phase 2: Basic Finetuning (Week 1-2)
- **04_full_finetuning_gpt2.ipynb** - Traditional full parameter training
- **05_classification_bert.ipynb** - BERT text classification

### Phase 3: Efficient Methods (Week 2)
- **06_lora_finetuning.ipynb** - LoRA on Llama/Phi-2
- **07_qlora_large_models.ipynb** - QLoRA on Mistral-7B
- **08_prefix_tuning.ipynb** - Prefix/P-Tuning v2

### Phase 4: Task-Specific (Week 3)
- **09_instruction_tuning.ipynb** - Chat/assistant training
- **10_question_answering.ipynb** - Extractive QA (SQuAD)
- **11_text_classification.ipynb** - Multi-class classification
- **12_text_generation.ipynb** - Creative generation/summarization

### Phase 5: Advanced (Week 4)
- **13_evaluation_framework.ipynb** - Comprehensive metrics suite
- **14_model_comparison.ipynb** - Base vs finetuned comparison
- **15_inference_optimization.ipynb** - Deployment optimization
- **16_production_deployment.ipynb** - HF Hub, API server, Docker

## 🔧 Technology Stack

- **Deep Learning**: PyTorch 2.1+
- **Transformers**: HuggingFace Transformers 4.36+
- **Efficient Finetuning**: PEFT, TRL
- **Quantization**: BitsAndBytes, Optimum
- **Evaluation**: Evaluate, ROUGE, BERTScore
- **Tracking**: Weights & Biases, TensorBoard

## 💾 Hardware Requirements

- **GPU**: Google Colab Free Tier (T4 with 15GB RAM, 8GB VRAM) or higher
- **Storage**: ~10GB Google Drive space (for checkpoints)
- **RAM**: 12GB+ system RAM

## 🎓 Models Used (8GB VRAM Optimized)

### Small Models (Full Finetuning)
- GPT-2 (124M/355M params)
- BERT-base (110M params)
- DistilGPT-2 (82M params)

### Medium Models (LoRA)
- Phi-2 (2.7B params)
- Llama-3.2-1B/3B
- Gemma-2B
- TinyLlama-1.1B

### Large Models (QLoRA Only)
- Mistral-7B (requires 4-bit quantization)

## 📊 Datasets

- **Instruction/Chat**: Dolly-15k, OpenAssistant, Alpaca
- **Classification**: AG News, Emotion, IMDB, Financial Phrasebank
- **Question Answering**: SQuAD, SQuAD v2, Adversarial QA
- **Text Generation**: TinyStories, DailyDialog, CNN/DailyMail

## 🛠️ Project Structure

```
llm-finetuning-production/
├── notebooks/          # All 16 Jupyter notebooks
├── src/                # Reusable Python modules
│   ├── data/          # Dataset loading utilities
│   ├── models/        # Model loading with quantization
│   ├── training/      # Custom trainers & callbacks
│   ├── evaluation/    # Metrics & benchmarking
│   └── utils/         # Memory optimization, logging
├── configs/           # YAML configuration files
├── experiments/       # Experiment tracking & results
├── tests/             # Unit tests
└── docs/              # Documentation
```

## 📖 Learning Path

**Total Time**: 26-34 hours

1. **Week 1** (6-8 hrs): Setup, data exploration, basic finetuning
2. **Week 2** (8-10 hrs): LoRA, QLoRA, Prefix Tuning
3. **Week 3** (6-8 hrs): Task-specific implementations
4. **Week 4** (6-8 hrs): Evaluation, comparison, deployment

## 🔑 Key Features

✅ Memory-optimized for 8GB GPU (finetune Mistral-7B with QLoRA!)
✅ Production-ready code with modular components
✅ Comprehensive evaluation framework
✅ Experiment tracking with W&B/TensorBoard
✅ Complete documentation
✅ Real-world deployment examples

## 📝 Memory Optimization Techniques

- Gradient checkpointing (30-50% memory reduction)
- Mixed precision training (fp16/bf16)
- 4-bit NF4 quantization
- Gradient accumulation
- 8-bit AdamW optimizer
- LoRA with r=8-16 (train <1% of parameters)
- Sequence length management

## 🤝 Workflow: Local → GitHub → Colab

1. **Local Development**: Edit notebooks/code on your machine
2. **Push to GitHub**: Version control your changes
3. **Run in Colab**: Clone repo and execute in Colab with GPU
4. **Save Results**: Store to Google Drive or HuggingFace Hub
5. **Iterate**: Pull results back, update documentation

## 📚 Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design
- [TECHNIQUES.md](docs/TECHNIQUES.md) - Finetuning methods explained
- [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - Common issues & solutions
- [RESULTS.md](docs/RESULTS.md) - Experiment results & insights

## 🎯 Success Criteria

By completing this project, you'll have:

✅ Working implementations of 4 finetuning techniques
✅ Experience with 4 different task types
✅ Production-ready, modular codebase
✅ Comprehensive evaluation framework
✅ Model deployment knowledge
✅ Real-world LLM engineering skills

## 🐛 Troubleshooting

**Out of Memory (OOM) Error**:
- Reduce batch size
- Enable gradient checkpointing
- Use smaller model or QLoRA
- Reduce sequence length

**Colab Disconnection**:
- Notebooks include auto-save to Google Drive
- Resume from checkpoints

**GPU Not Detected**:
- Runtime → Change runtime type → GPU (T4)
- Verify with `nvidia-smi` in first notebook

## 📜 License

MIT License - Feel free to use for learning and commercial projects

## 🙏 Acknowledgments

Built with:
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [TRL](https://github.com/huggingface/trl)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Ready to start?** Open `notebooks/01_foundations/01_environment_setup.ipynb` in Google Colab!
