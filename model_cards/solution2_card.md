# Model Card for Solution 2 (Large Neural Meta-Ensemble)

## Model Details
- **Architecture**: A deep neural meta-learner trained to optimally blend the probabilistic outputs of four fine-tuned transformer cross-encoders.
  - **Base Models**:
    - **DeBERTa-v3-large** (Cross-encoder fine-tuned for sequence pair classification)
    - **RoBERTa-large** (Cross-encoder fine-tuned for sequence pair classification)
    - **ELECTRA-large-discriminator** (Cross-encoder fine-tuned for sequence pair classification)
    - **XLNet-large-cased** (Cross-encoder fine-tuned for sequence pair classification)
  - **Ensemble Meta-Learner**: A 4-layer fully-connected neural network with Batch Normalization and Dropout. Takes in 14 aggregate statistical and raw predictive features extracted from base probabilities, outputting a highly calibrated final likelihood.

## Training Procedures
The base transformers were fine-tuned using `BCEWithLogitsLoss` equipped with AdamW optimizer, warmup scheduling, mixed precision (FP16), and early stopping on dev F1 scores. 
After obtaining base model predictions, the ensemble meta-learner was trained using Focal Loss to correct heavily disagreed inputs. Isotonic regression calibration was further applied to the final probabilities to smooth the predictive distribution.

## Performance
- Competes aggressively on the evaluation constraints using State-of-the-Art base representations. Dev metrics align strongly closely due to early stopping integration against the official dev setup constraints.

## Intended Use
Designed to produce state-of-the-art pairwise sequence classification performance on text pairs requiring deep semantic understanding rather than structural metrics capability (see Solution 1).

## Limitations
Incredibly computationally heavy to train. Standard inference requires sequential passes through 4 separate >350M parameter transformer architecture heads (or concurrent multi-GPU distribution). Does not encode hard stylometric metrics unless the attention head implicitly captures them.
