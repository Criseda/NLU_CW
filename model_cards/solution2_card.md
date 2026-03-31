---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
language:
- en
license: other
tags:
- authorship-verification
- transformer
- ensemble
- nlu
- deberta
- roberta
- xlnet
- electra
model_name: Solution 2 (Large Neural Meta-Ensemble)
---

# Model Card for Solution 2 (Large Neural Meta-Ensemble)

This model is a state-of-the-art neural meta-ensemble designed for the COMP34812 NLU Shared Task (Authorship Verification). It optimally blends the probabilistic outputs of four fine-tuned transformer cross-encoders.

## Model Details

### Model Description

The architecture uses a 4-layer fully-connected neural network with Batch Normalization and Dropout to ensemble the predictions from DeBERTa-v3-large, RoBERTa-large, ELECTRA-large-discriminator, and XLNet-large-cased.

- **Developed by:** Group 17 (NLU Coursework Track C)
- **Model type:** Deep Learning based (Transformer Neural Meta-Ensemble)
- **Language(s) (NLP):** English
- **License:** Individual academic use for COMP34812 coursework.
- **Finetuned from model:** DeBERTa-v3-large, RoBERTa-large, ELECTRA-large-discriminator, XLNet-large-cased.

### Model Sources

- **DeBERTa Base:** [View on Google Drive](https://drive.google.com/file/d/1IfzDd87FARs4CE7rEviKcvI2UL9sKH4s/view?usp=sharing)
- **XLNet Base:** [View on Google Drive](https://drive.google.com/file/d/1HWnOv36f-VOe92tKl2saQ1xs4k7hwbty/view?usp=sharing)
- **RoBERTa Base:** [View on Google Drive](https://drive.google.com/file/d/15M0kqGWtfLrO5F2mB3O4MH3Yt44dZDTZ/view?usp=sharing)
- **ELECTRA Base:** [View on Google Drive](https://drive.google.com/file/d/1ce5tgkHNuZzLhpoata9FiBDsPAtxGPKe/view?usp=sharing)
- **Ensemble Meta-Learner:** [View on Google Drive](https://drive.google.com/file/d/1jf9hXJtY0LCHW-dOiEBgo-SdnrZK0Hri/view?usp=sharing)

## Uses

### Direct Use

Pairwise Authorship Verification (AV) where deep semantic understanding is prioritized over structural stylometrics.

### Out-of-Scope Use

Real-time inference on mobile or low-power devices due to high parameter count (avg 335M+ per head).

## Bias, Risks, and Limitations

Performance depends on the context length (max 512 tokens). There is an inherent risk of inheriting biases from the massive pre-training corpora of the base models.

### Recommendations

For optimal results, ensure target texts are at least 100 tokens long. Requires significant VRAM for efficient inference.

## How to Get Started with the Model

Refer to the demo notebook `notebooks/demo_solution2.ipynb` for instructions on loading the fine-tuned base models and the neural meta-ensemble.

## Training Details

### Training Data

The model was trained on the **Official NLU Authorship Verification Track (C) Training set**.

### Training Procedure

#### Preprocessing

Hugging Face default tokenizers for each transformer architecture were used.

#### Training Hyperparameters

- **Base Transformers:** BCEWithLogitsLoss, AdamW, early stopping.
- **Ensemble Meta-Learner:** 4-layer fully-connected network with Batch Normalization and Dropout.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model was evaluated on the **Official NLU Authorship Verification Track (C) Development set**.

#### Metrics

- **Macro F1 Score:** Primary metric.
- **Accuracy:** Secondary metric.
- **Calibration Error:** Evaluated to ensure probabilities realistically represent authorship confidence.

### Results

The ensemble consistently achieves competitive performance by correcting base model disagreements through the 4-layer neural meta-learner.

## Environmental Impact

Training 4 large transformer architectures has a non-negligible carbon footprint. 

- **Hardware Type:** GPU (H100/A100) used for training.
- **Inference Requirement:** High-end GPUs (V100+) recommended for batch processing.

## Technical Specifications

### Compute Infrastructure

#### Software

- Python 3.14
- PyTorch
- Transformers
- Scikit-learn

## Model Card Authors

- Group 17

## Model Card Contact

For coursework-related inquiries, contact the Group 17 owners.
