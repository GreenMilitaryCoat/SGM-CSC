# Bridging Semantic and Geometry: A Consistent Subspace Clustering Framework with Multi-Scale Graph Fusion
This is the official PyTorch implementation of SGM-CSC, a framework designed to bridge semantic features and geometric structures for robust image clustering.
# Overview

SGM-CSC is a novel deep subspace clustering framework designed to address the limitations of existing methods that overemphasize semantic features while neglecting latent geometric structures. By integrating multi-scale graph fusion and a dual self-expression consistency mechanism, our method achieves superior clustering stability and discriminability.

---
- Key Features:Multi-scale Feature Extraction:
- Leverages intermediate features from ConvNeXt stages (Stage 1-4) to capture both local textures and global semantics. 
- Attention-Residual Fusion (AR-Fusion): An adaptive fusion strategy that integrates multi-level geometric details while preserving fine-grained information. 
- Dual Self-Expressiveness: Jointly models subspace structures in both semantic and geometric spaces.
- Consistency Constraint: Uses a Smooth L1 loss to align the subspace representations from different views, ensuring structural consistency.
  
---

## Prepare

### ** Install dependencies**

```bash
git clone https://github.com/DMSSC-123/DeepSubspaceClustering.git
cd SGM-CSC
pip install -r requirements.txt

```
### **2. feature extraction**
```bash
 feature_extract.py 
```
### **3. Train **
```bash
 main.py 
