


A multimodal deep learning project for detecting hate speech in memes by combining visual and linguistic features.

![Project Banner](https://via.placeholder.com/800x200/4A90E2/FFFFFF?text=Hateful+Meme+Detection+Project)

## 👥 Team Members

- **Jay Gala** - jaygala@seas
- **Saurabh Raut** - sraut@seas
- **Rajnish Gupta** - rajnishg@seas

## 📋 Abstract

Detecting hate speech in multimodal memes requires combining visual and linguistic features to understand the intrinsic meaning of memes. This project explores various multimodal fusion techniques including:

- **Early Fusion**
- **Mid Fusion**
- **Late Fusion**

We compare these approaches against unimodal models using only images or text, demonstrating that multimodal approaches significantly outperform single-modality models.

### 🎯 Best Results

- **Accuracy**: 66.8%
- **AUROC**: 0.78

## 🎯 Motivation

Social media has enabled rapid spread of memes, which can range from light-hearted jokes to hateful content targeting individuals or communities. The challenge lies in the fact that memes combine images and text, where the meaning emerges from their combination rather than either modality alone.

### Example: Benign Confounders

![Benign Confounder Example](https://via.placeholder.com/600x300/E8E8E8/333333?text=Image+%2B+Text+%3D+Different+Meaning)

_The combination of text and image conveys different meaning compared to either modality individually._

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT MODALITIES                      │
├──────────────────────────┬──────────────────────────────┤
│         IMAGE            │           TEXT               │
│    (Meme Visual)         │     (Embedded Text)          │
└──────────┬───────────────┴──────────────┬───────────────┘
           │                               │
           ▼                               ▼
    ┌──────────────┐              ┌──────────────┐
    │   ResNeXt    │              │   RoBERTa    │
    │  (2048-dim)  │              │   (768-dim)  │
    └──────┬───────┘              └──────┬───────┘
           │                               │
           └───────────┬───────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  CLIP Embeddings │
              │    (512-dim)     │
              └────────┬─────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   Fusion Strategy    │
            │  (Early/Mid/Late)    │
            └──────────┬───────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  MLP Classifier  │
              └────────┬─────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Hateful / Not  │
              └─────────────────┘
```

## 🔬 Approaches

### 1. Unimodal Models

#### Text Unimodal

- Uses RoBERTa embeddings
- 3-layer MLP with ReLU and Sigmoid activations
- **Best Test Accuracy**: 59.1%

#### Image Unimodal

- Uses ResNeXt embeddings
- Same architecture as text model
- **Best Test Accuracy**: 52.9%

### 2. Multimodal Models

#### Early Fusion ⭐ (Best Performance)

```
Image Embedding ──┐
                  ├──> Concatenate ──> MLP ──> Output
Text Embedding  ──┘
```

- Concatenates embeddings at input level
- **Test Accuracy**: 66.8%
- **Test AUROC**: 0.78

#### Mid Fusion

```
Image ──> Linear ──┐
                   ├──> Concatenate ──> Linear ──> Output
Text  ──> Linear ──┘
```

- Processes each modality separately
- Fuses in middle layers
- **Test Accuracy**: 64.8%

#### Late Fusion

```
Image ──> MLP ──> Score ──┐
                           ├──> Average ──> Output
Text  ──> MLP ──> Score ──┘
```

- Trains separate models
- Combines predictions
- **Test Accuracy**: 63.0%

## 📊 Experimental Results

### Performance Comparison (RoBERTa + ResNeXt)

|Architecture|Epochs|Val Acc.|Val AUROC|Test Acc.|Test AUROC|
|---|---|---|---|---|---|
|Text unimodal|100|56.60%|0.61|59.10%|0.63|
|Image unimodal|50|52.80%|0.54|52.90%|0.54|
|Early fusion|100|61.00%|0.72|**63.90%**|**0.73**|
|Mid fusion|100|59.20%|0.63|61.30%|0.64|
|Late fusion|100|58.20%|0.62|59.50%|0.63|

### Performance Comparison (CLIP Embeddings)

|Architecture|Epochs|Val Acc.|Val AUROC|Test Acc.|Test AUROC|
|---|---|---|---|---|---|
|Text unimodal|100|57.40%|0.61|58.40%|0.63|
|Image unimodal|100|59.20%|0.62|60.20%|0.64|
|**Early fusion**|100|64.20%|0.74|**66.80%**|**0.78**|
|Mid fusion|170|63.20%|0.67|64.80%|0.71|
|Late fusion|170|62.60%|0.66|63.00%|0.69|

### Early Fusion Ablation Study

|Embeddings|Optimizer|Epochs|Val Acc.|Val AUROC|Test Acc.|Test AUROC|
|---|---|---|---|---|---|---|
|ResNext + RoBERTa|Adam|100|61.0%|0.72|63.9%|0.73|
|ResNext + RoBERTa|SGD|200|63.0%|0.70|64.5%|0.72|
|ResNext + RoBERTa + CLIP|Adam|150|58.4%|0.66|63.5%|0.73|
|ResNext + RoBERTa + CLIP|SGD|150|59.6%|0.69|63.8%|0.72|
|**CLIP Only**|**Adam**|100|**64.2%**|**0.74**|**66.8%**|**0.78**|
|CLIP Only|SGD|250|60.2%|0.69|62.2%|0.73|

## 🔑 Key Findings

1. **Multimodal > Unimodal**: All multimodal architectures significantly outperform unimodal approaches
2. **Early Fusion Wins**: Combining features at the input level produces the best results
3. **CLIP is Superior**: CLIP embeddings outperform separate ResNeXt + RoBERTa embeddings
4. **Adam > SGD**: Adam optimizer consistently produces better results than SGD

## 🛠️ Technical Details

### Pre-trained Models Used

- **RoBERTa**: Text embedding generation (768-dimensional)
- **ResNeXt**: Image embedding generation (2048-dimensional)
- **CLIP**: Joint image-text embeddings (512-dimensional each)

### Training Configuration

- **Framework**: PyTorch
- **Optimizers**: Adam (lr=1e-4), SGD
- **Activation Functions**: ReLU, Sigmoid
- **Normalization**: BatchNorm
- **Architecture**: Multi-Layer Perceptron (MLP)

### Dataset

**Hateful Memes Challenge Dataset** by Meta AI

- Specifically designed to fail on unimodal approaches
- Contains multimodal memes with labels
- Includes "benign confounders" that require both modalities

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision
pip install transformers
pip install clip
pip install numpy pandas matplotlib
```

### Usage

```python
# Load pre-trained embeddings
from transformers import RobertaModel
import clip
import torch

# Generate embeddings
text_embeddings = roberta_model(text_input)
image_embeddings = resnext_model(image_input)

# Early fusion
combined = torch.cat([text_embeddings, image_embeddings], dim=1)

# Pass through classifier
output = mlp_classifier(combined)
```

## 📈 Future Work

- Experiment with additional architectures and deeper networks
- Implement various dropout strategies to reduce overfitting
- Fine-tune pre-trained models rather than using frozen embeddings
- Explore attention mechanisms for better fusion
- Augment dataset with additional meme-related databases
- Investigate transformer-based fusion techniques

## 🔍 Challenges

1. **Overfitting**: High-dimensional feature vectors (2048 + 768 = 2816) lead to overfitting on limited data
2. **Dataset Size**: Limited training data makes it difficult to train deeper networks
3. **Complexity**: Memes contain cultural references and context that are difficult to encode
4. **Benign Confounders**: Images/text that are benign individually but hateful together

## 📚 References

1. [Facebook Hateful Memes Challenge](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/)
2. Kiela et al., "The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes" (arXiv:2005.04790)
3. Deshpande & Mani, "An Interpretable Approach to Hateful Meme Detection" (arXiv:2108.10069)
4. MIT 6.S191: Multimodal Deep Learning Lecture

## 📄 License

This project was completed as part of ESE 5460 coursework.

## 🙏 Acknowledgments

- Meta AI for the Hateful Memes Challenge dataset
- OpenAI for CLIP embeddings
- Hugging Face for pre-trained models

---

**Course**: ESE 5460  
**Institution**: University of Pennsylvania  
**Year**: 2024
