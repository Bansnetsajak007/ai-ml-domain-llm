"""
RAMESH Curated Paper Collection 📚
====================================
Hand-picked landmark papers for AI/ML LLM training dataset.

These are the MUST-HAVE papers that defined the field.
Organized by topic with specific search queries.

Target: 750-800 high-quality papers
- 300 Landmark papers (curated queries)
- 260 Survey/Review papers (filtered search)
- 190 Recent advances (2022-2024)
"""

# ============================================
# CURATED TOPIC LIST
# Each topic has specific queries designed to find CORE papers
# Papers per topic: 25-35 (controlled, no redundancy)
# ============================================

CURATED_TOPICS = {
    # ===========================================
    # 🏛️ FOUNDATIONS (150 papers)
    # ===========================================
    "foundations": {
        "Neural Network Fundamentals": {
            "queries": [
                "backpropagation neural network learning",
                "universal approximation theorem neural",
                "deep learning representation learning"
            ],
            "papers_per_query": 10,
            "total_target": 30,
            "categories": ["cs.LG", "cs.NE"],
            "year_range": (2010, 2024)
        },
        "Optimization in Deep Learning": {
            "queries": [
                "stochastic gradient descent convergence",
                "Adam optimizer adaptive learning",
                "batch normalization training deep"
            ],
            "papers_per_query": 10,
            "total_target": 30,
            "categories": ["cs.LG", "stat.ML"],
            "year_range": (2014, 2024)
        },
        "Regularization Generalization": {
            "queries": [
                "dropout regularization neural network",
                "weight decay deep learning",
                "generalization deep neural network"
            ],
            "papers_per_query": 10,
            "total_target": 30,
            "categories": ["cs.LG", "stat.ML"],
            "year_range": (2012, 2024)
        },
        "Loss Functions Training": {
            "queries": [
                "cross entropy loss classification",
                "contrastive learning representation",
                "self-supervised learning pretext"
            ],
            "papers_per_query": 10,
            "total_target": 30,
            "categories": ["cs.LG", "cs.CV"],
            "year_range": (2015, 2024)
        },
        "Mathematical Foundations ML": {
            "queries": [
                "information theory deep learning",
                "statistical learning theory neural",
                "kernel methods support vector"
            ],
            "papers_per_query": 10,
            "total_target": 30,
            "categories": ["cs.LG", "stat.ML", "cs.IT"],
            "year_range": (2010, 2024)
        },
    },

    # ===========================================
    # 🏗️ CORE ARCHITECTURES (180 papers)
    # ===========================================
    "architectures": {
        "Convolutional Neural Networks": {
            "queries": [
                "AlexNet ImageNet classification",
                "ResNet deep residual learning",
                "VGGNet very deep convolutional"
            ],
            "papers_per_query": 10,
            "total_target": 30,
            "categories": ["cs.CV", "cs.LG"],
            "year_range": (2012, 2024)
        },
        "Recurrent Neural Networks": {
            "queries": [
                "LSTM long short-term memory",
                "GRU gated recurrent unit",
                "sequence to sequence learning"
            ],
            "papers_per_query": 10,
            "total_target": 30,
            "categories": ["cs.CL", "cs.LG", "cs.NE"],
            "year_range": (2014, 2022)
        },
        "Transformer Architecture": {
            "queries": [
                "attention is all you need transformer",
                "self-attention mechanism neural",
                "multi-head attention transformer"
            ],
            "papers_per_query": 12,
            "total_target": 35,
            "categories": ["cs.CL", "cs.LG"],
            "year_range": (2017, 2024)
        },
        "Generative Adversarial Networks": {
            "queries": [
                "generative adversarial network GAN",
                "StyleGAN image synthesis",
                "conditional GAN image generation"
            ],
            "papers_per_query": 10,
            "total_target": 30,
            "categories": ["cs.CV", "cs.LG"],
            "year_range": (2014, 2024)
        },
        "Autoencoders VAE": {
            "queries": [
                "variational autoencoder VAE",
                "autoencoder representation learning",
                "disentangled representation learning"
            ],
            "papers_per_query": 8,
            "total_target": 25,
            "categories": ["cs.LG", "stat.ML"],
            "year_range": (2013, 2024)
        },
        "Graph Neural Networks": {
            "queries": [
                "graph neural network GNN",
                "graph convolutional network",
                "message passing neural network"
            ],
            "papers_per_query": 10,
            "total_target": 30,
            "categories": ["cs.LG", "cs.SI"],
            "year_range": (2016, 2024)
        },
    },

    # ===========================================
    # 📝 LLMs & NLP (200 papers) - MAIN FOCUS
    # ===========================================
    "llm_nlp": {
        "Word Embeddings": {
            "queries": [
                "word2vec distributed representations",
                "GloVe global vectors word",
                "fastText subword embeddings"
            ],
            "papers_per_query": 8,
            "total_target": 25,
            "categories": ["cs.CL"],
            "year_range": (2013, 2020)
        },
        "BERT Encoder Models": {
            "queries": [
                "BERT pre-training deep bidirectional",
                "RoBERTa robustly optimized BERT",
                "ALBERT lite BERT self-supervised"
            ],
            "papers_per_query": 10,
            "total_target": 30,
            "categories": ["cs.CL"],
            "year_range": (2018, 2024)
        },
        "GPT Decoder Models": {
            "queries": [
                "GPT generative pre-training",
                "language models are few-shot learners",
                "GPT-4 technical report"
            ],
            "papers_per_query": 12,
            "total_target": 35,
            "categories": ["cs.CL", "cs.AI"],
            "year_range": (2018, 2024)
        },
        "Encoder-Decoder Models": {
            "queries": [
                "T5 text-to-text transfer transformer",
                "BART denoising sequence-to-sequence",
                "mT5 multilingual T5"
            ],
            "papers_per_query": 8,
            "total_target": 25,
            "categories": ["cs.CL"],
            "year_range": (2019, 2024)
        },
        "Instruction Tuning RLHF": {
            "queries": [
                "instruction tuning language models",
                "reinforcement learning human feedback",
                "InstructGPT training language models"
            ],
            "papers_per_query": 12,
            "total_target": 35,
            "categories": ["cs.CL", "cs.AI", "cs.LG"],
            "year_range": (2021, 2024)
        },
        "Prompt Engineering ICL": {
            "queries": [
                "chain of thought prompting reasoning",
                "in-context learning language models",
                "prompt tuning soft prompts"
            ],
            "papers_per_query": 10,
            "total_target": 30,
            "categories": ["cs.CL", "cs.AI"],
            "year_range": (2020, 2024)
        },
        "LLM Scaling Emergent": {
            "queries": [
                "scaling laws neural language models",
                "emergent abilities large language",
                "chinchilla compute optimal training"
            ],
            "papers_per_query": 8,
            "total_target": 25,
            "categories": ["cs.CL", "cs.LG"],
            "year_range": (2020, 2024)
        },
    },

    # ===========================================
    # 👁️ VISION & MULTIMODAL (100 papers)
    # ===========================================
    "vision_multimodal": {
        "Vision Transformers": {
            "queries": [
                "vision transformer ViT image",
                "DeiT data-efficient image transformers",
                "Swin transformer shifted windows"
            ],
            "papers_per_query": 8,
            "total_target": 25,
            "categories": ["cs.CV"],
            "year_range": (2020, 2024)
        },
        "Object Detection": {
            "queries": [
                "YOLO real-time object detection",
                "DETR detection transformer",
                "Faster R-CNN region-based"
            ],
            "papers_per_query": 8,
            "total_target": 25,
            "categories": ["cs.CV"],
            "year_range": (2015, 2024)
        },
        "Diffusion Models": {
            "queries": [
                "denoising diffusion probabilistic",
                "stable diffusion latent",
                "score-based generative models"
            ],
            "papers_per_query": 8,
            "total_target": 25,
            "categories": ["cs.CV", "cs.LG"],
            "year_range": (2020, 2024)
        },
        "Vision Language Models": {
            "queries": [
                "CLIP contrastive language image",
                "LLaVA visual instruction tuning",
                "multimodal large language models"
            ],
            "papers_per_query": 8,
            "total_target": 25,
            "categories": ["cs.CV", "cs.CL"],
            "year_range": (2021, 2024)
        },
    },

    # ===========================================
    # ⚡ TRAINING & EFFICIENCY (100 papers)
    # ===========================================
    "training_efficiency": {
        "Optimizers": {
            "queries": [
                "Adam optimizer deep learning",
                "learning rate scheduling warmup",
                "AdamW decoupled weight decay"
            ],
            "papers_per_query": 8,
            "total_target": 25,
            "categories": ["cs.LG"],
            "year_range": (2014, 2024)
        },
        "Distributed Training": {
            "queries": [
                "distributed deep learning training",
                "data parallelism model parallelism",
                "ZeRO memory optimization"
            ],
            "papers_per_query": 8,
            "total_target": 25,
            "categories": ["cs.DC", "cs.LG"],
            "year_range": (2018, 2024)
        },
        "Model Compression": {
            "queries": [
                "quantization neural network inference",
                "pruning deep neural networks",
                "low-rank adaptation LoRA"
            ],
            "papers_per_query": 8,
            "total_target": 25,
            "categories": ["cs.LG", "cs.CV"],
            "year_range": (2016, 2024)
        },
        "Knowledge Distillation": {
            "queries": [
                "knowledge distillation neural network",
                "teacher student network compression",
                "distilling knowledge smaller models"
            ],
            "papers_per_query": 8,
            "total_target": 25,
            "categories": ["cs.LG", "cs.CV"],
            "year_range": (2015, 2024)
        },
    },

    # ===========================================
    # 🔬 EVALUATION & SAFETY (50 papers)
    # ===========================================
    "evaluation_safety": {
        "NLP Benchmarks": {
            "queries": [
                "GLUE benchmark natural language",
                "SuperGLUE benchmark understanding",
                "MMLU massive multitask language"
            ],
            "papers_per_query": 8,
            "total_target": 25,
            "categories": ["cs.CL"],
            "year_range": (2018, 2024)
        },
        "LLM Safety Alignment": {
            "queries": [
                "AI alignment safety language models",
                "red teaming language models",
                "constitutional AI harmlessness"
            ],
            "papers_per_query": 8,
            "total_target": 25,
            "categories": ["cs.CL", "cs.AI", "cs.CY"],
            "year_range": (2021, 2024)
        },
    },
}

# ============================================
# SURVEY PAPER QUERIES
# These specifically target comprehensive reviews
# ============================================

SURVEY_QUERIES = [
    {"query": "survey deep learning neural network", "max": 20, "categories": ["cs.LG"]},
    {"query": "survey transformer attention mechanism", "max": 15, "categories": ["cs.CL", "cs.LG"]},
    {"query": "survey large language models LLM", "max": 20, "categories": ["cs.CL"]},
    {"query": "survey natural language processing NLP", "max": 15, "categories": ["cs.CL"]},
    {"query": "survey computer vision deep learning", "max": 15, "categories": ["cs.CV"]},
    {"query": "survey generative models", "max": 10, "categories": ["cs.LG", "cs.CV"]},
    {"query": "survey reinforcement learning", "max": 15, "categories": ["cs.LG", "cs.AI"]},
    {"query": "survey graph neural network", "max": 10, "categories": ["cs.LG"]},
    {"query": "survey pre-trained language models", "max": 15, "categories": ["cs.CL"]},
    {"query": "survey prompt learning language models", "max": 10, "categories": ["cs.CL"]},
]

# Total survey target: ~145 papers


def get_collection_plan():
    """Generate a complete collection plan with paper counts."""
    plan = {
        "categories": {},
        "total_papers": 0,
        "survey_papers": 0
    }
    
    for category_name, topics in CURATED_TOPICS.items():
        category_total = 0
        plan["categories"][category_name] = {"topics": {}, "total": 0}
        
        for topic_name, config in topics.items():
            target = config["total_target"]
            plan["categories"][category_name]["topics"][topic_name] = target
            category_total += target
        
        plan["categories"][category_name]["total"] = category_total
        plan["total_papers"] += category_total
    
    # Add survey papers
    for survey in SURVEY_QUERIES:
        plan["survey_papers"] += survey["max"]
    
    plan["total_papers"] += plan["survey_papers"]
    
    return plan


def print_collection_plan():
    """Print a formatted collection plan."""
    plan = get_collection_plan()
    
    print("\n" + "=" * 60)
    print("📚 RAMESH CURATED COLLECTION PLAN")
    print("=" * 60)
    
    for cat_name, cat_data in plan["categories"].items():
        print(f"\n🏷️  {cat_name.upper().replace('_', ' ')} ({cat_data['total']} papers)")
        print("-" * 40)
        for topic, count in cat_data["topics"].items():
            print(f"   • {topic}: {count} papers")
    
    print(f"\n📖 SURVEY PAPERS: {plan['survey_papers']} papers")
    print("=" * 60)
    print(f"📊 TOTAL TARGET: {plan['total_papers']} papers")
    print("=" * 60)


if __name__ == "__main__":
    print_collection_plan()
