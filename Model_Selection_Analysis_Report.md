# Phishing Detection Model Suitability Analysis Report

**Project**: Phishing Detection Project
**Dataset**: structured_legitimate_data.csv + structured_phishing_data.csv
**Generated**: 2025-11-17

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Samples** | 45,930 (Legitimate: 22,093 / Phishing: 23,837) |
| **Class Balance** | Good (48.1% : 51.9%) |
| **Total Features** | 43 features + URL + label |
| **Feature Types** | 19 boolean features + 22 numerical features + 2 text columns |
| **Data Type** | **Tabular Structured Data** |
| **Feature Source** | HTML DOM structure feature extraction |
| **Missing Values** | None detected (requires further validation) |

### Key Feature Categories
1. **HTML Element Presence** (Boolean): has_title, has_form, has_password, has_iframe, etc.
2. **Element Count Statistics** (Numerical): number_of_inputs, number_of_scripts, number_of_links, etc.
3. **Content Length Metrics** (Numerical): length_of_title, length_of_text
4. **Identifier Information**: URL, label

---

## Model Suitability Assessment Matrix

Evaluating models mentioned in Models.docx against dataset characteristics:

### Rating Scale
- [5-STAR] Highly Recommended (Best Fit)
- [4-STAR] Strongly Recommended
- [3-STAR] Recommended
- [2-STAR] Optional
- [1-STAR] Not Recommended

---

## Detailed Model Analysis

### (1) Classic Lightweight Models

#### 1.1 Logistic Regression
**Suitability Rating**: [4-STAR]

**Advantages**:
- **High Feature Compatibility**: Boolean + numerical features are ideal for linear models
- **Fast Training**: 46K samples can be trained in seconds
- **Strong Interpretability**: Direct coefficient inspection reveals DOM feature contributions
- **Perfect Baseline**: Quick validation of feature engineering effectiveness
- **Low Overfitting Risk**: Easy regularization control

**Disadvantages**:
- **Linear Assumption Limitation**: Cannot capture feature interactions (e.g., "has_password AND has_form" combinations)
- **Performance Ceiling**: Limited capability for complex patterns

**Dataset Match**: 89%
**Implementation Recommendation**: **Mandatory as first baseline model** for rapid data quality and feature effectiveness validation

---

#### 1.2 k-Nearest Neighbors (kNN)
**Suitability Rating**: [2-STAR]

**Advantages**:
- **No Training Required**: Quick prototype validation
- **Non-parametric**: Automatically adapts to data distribution

**Disadvantages**:
- **Curse of Dimensionality**: Distance metrics may fail in 43-dimensional feature space
- **Slow Inference**: 46K samples × 43 dimensions requires traversing massive data for each prediction
- **High Memory Consumption**: Must store entire training dataset
- **Sensitive to Feature Scaling**: Boolean (0/1) vs numerical (0-1000+) requires careful standardization
- **Noise Sensitive**: Dataset may contain boundary-ambiguous samples

**Dataset Match**: 42%
**Implementation Recommendation**: **Not recommended**. If experimenting, use only for small-scale validation; scikit-learn implementation sufficient, no PyTorch needed

---

#### 1.3 Decision Tree
**Suitability Rating**: [3-STAR]

**Advantages**:
- **Feature Interaction**: Automatically learns feature combination rules (e.g., "if has_password=1 and number_of_inputs<2 → phishing")
- **Non-linear**: Handles complex decision boundaries
- **Strong Interpretability**: Visualizable decision paths
- **Handles Mixed Types**: Boolean + numerical requires no special preprocessing
- **No Feature Scaling Needed**: Based on split rules, scale-invariant

**Disadvantages**:
- **Extreme Overfitting Risk**: Single tree easily memorizes training data
- **Poor Generalization**: Unless strictly pruned (max_depth, min_samples_split)
- **Instability**: Minor data changes may result in completely different tree structures

**Dataset Match**: 73%
**Implementation Recommendation**: **Optional**. Mainly for feature importance analysis and visualization; not recommended as final model. Prefer ensemble tree methods

---

### (2) Ensemble Tree Methods [TOP TIER]

#### 2.1 Random Forest
**Suitability Rating**: [5-STAR]

**Advantages**:
- **Tabular Data Champion**: Excels on structured features
- **Overfitting Resistant**: Multi-tree ensemble with strong generalization
- **Robustness**: Insensitive to noise and outliers
- **Feature Importance**: Quantifies each DOM feature's contribution
- **No Feature Scaling**: Handles boolean + numerical mixing well
- **Parallel Training**: Multi-core CPU significantly accelerates training
- **Handles Non-linearity**: Automatically learns complex interaction patterns

**Disadvantages**:
- **Model Size**: Storing hundreds of trees requires storage space
- **Inference Speed**: Slower than single models, but acceptable for 46K samples

**Dataset Match**: 98%
**Implementation Recommendation**: **Strongly recommended as primary model**. Suggested starting parameters:
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    class_weight='balanced'
)
```

---

#### 2.2 Gradient Boosting (XGBoost / LightGBM / CatBoost)
**Suitability Rating**: [5-STAR]

**Advantages**:
- **Best Performance**: Typically wins tabular data competitions
- **Efficient Training**: LightGBM trains 46K samples extremely fast
- **Feature Interaction**: Automatically learns high-order feature combinations
- **Regularization**: Built-in L1/L2, dropout prevents overfitting
- **Handles Imbalance**: Built-in class weights and focal loss options
- **Incremental Learning**: Early stopping saves training time
- **CatBoost Advantage**: Native categorical feature handling (for future URL domain features)

**Disadvantages**:
- **Complex Tuning**: Requires adjusting learning_rate, tree depth, regularization hyperparameters
- **Slightly Less Interpretable**: Though SHAP/feature importance available, more complex than single tree

**Dataset Match**: 99%
**Implementation Recommendation**: **Prefer LightGBM or XGBoost**. Suggested starting configuration:
```python
# LightGBM
lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    class_weight='balanced',
    early_stopping_rounds=50
)
```

**Model Comparison**:
- **XGBoost**: Mature, stable, strong community support
- **LightGBM**: Fastest training, recommended for large-scale experiments
- **CatBoost**: Best if adding categorical features (e.g., domain suffix, protocol type)

---

### (3) Linear + Nonlinear Combination

#### 3.1 MLP (Multilayer Perceptron)
**Suitability Rating**: [4-STAR]

**Advantages**:
- **Non-linear Learning**: Hidden layers learn complex feature combinations
- **Feature Fusion**: Easy to extend with TF-IDF/embeddings (for future URL text features)
- **PyTorch Native**: Meets project requirements, facilitates deep learning extensions
- **End-to-End Training**: Jointly optimizes all layer parameters

**Disadvantages**:
- **Feature Engineering Dependent**: Usually underperforms tree models on tabular data
- **Requires Feature Normalization**: Boolean and numerical types need standardization
- **Complex Tuning**: Layers, neurons, dropout, learning rate, etc.
- **Training Instability**: Requires careful initialization and optimizer setup
- **Overfitting Risk**: 43-dimensional input easily overfits in deep networks

**Dataset Match**: 76%
**Implementation Recommendation**: **Recommended as deep learning baseline**. Suggested architecture:
```python
# Simple effective MLP structure
nn.Sequential(
    nn.Linear(43, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 1),
    nn.Sigmoid()
)
```

**Important Notes**:
1. **Must Standardize**: Use StandardScaler or MinMaxScaler
2. **Class Balance**: Use BCEWithLogitsLoss + pos_weight
3. **Early Stopping**: Monitor validation set to prevent overfitting

---

### (4) Support Vector Machine (SVM)
**Suitability Rating**: [3-STAR]

**Advantages**:
- **Medium Dimension Fit**: 43 dimensions within SVM acceptable range
- **Kernel Functions**: RBF kernel captures non-linear patterns
- **Theoretical Guarantees**: Maximum margin principle, strong generalization

**Disadvantages**:
- **Slow Training**: 46K samples may require minutes to hours (kernel-dependent)
- **Memory Consumption**: Kernel matrix computation requires significant memory
- **Difficult Tuning**: C, gamma parameters sensitive, requires grid search
- **Probability Output**: Requires additional calibration (Platt scaling)
- **No Incremental Learning**: New data requires complete retraining

**Dataset Match**: 68%
**Implementation Recommendation**: **Optional for comparative experiments**, not recommended as primary model. If using, suggest:
```python
SVC(kernel='rbf', C=1.0, gamma='scale',
    class_weight='balanced', probability=True)
```

---

### (5) Naive Bayes
**Suitability Rating**: [2-STAR]

**Advantages**:
- **Extremely Fast Training**: Linear time complexity
- **Small Memory Footprint**: Only stores probability tables
- **Good for Text**: If extracting URL text features later, useful for text channel

**Disadvantages**:
- **Feature Independence Assumption**: DOM features clearly not independent (e.g., has_form and has_password highly correlated)
- **Performance Limitation**: Usually underperforms on complex data
- **Numerical Feature Handling**: Requires distribution assumption (Gaussian NB) or discretization

**Dataset Match**: 45%
**Implementation Recommendation**: **Not recommended for current dataset**. If extracting URL string TF-IDF features later, can serve as text submodule baseline

---

### (6) Deep / Multimodal Models

#### 6.1 CNN / Transformer on Rendered Screenshots
**Suitability Rating**: [1-STAR] (for current dataset)

**Advantages**:
- **Visual Information**: Captures logo, layout, color scheme visual phishing cues
- **Transfer Learning**: Can use pretrained ViT/ResNet

**Disadvantages**:
- **Dataset Has No Images**: Current dataset contains only DOM statistics, **no screenshot data**
- **Additional Collection Cost**: Need to revisit 46K URLs and capture screenshots (time-consuming, may fail)
- **High Computational Cost**: Requires GPU training and inference
- **Storage Cost**: 46K images require several GB storage

**Dataset Match**: **0%** (current dataset not applicable)
**Implementation Recommendation**: **Not recommended**, unless:
1. Future project phases require adding visual channel
2. Sufficient time and resources to recollect screenshot data

---

#### 6.2 Transformer / BERT-like Models on Text + DOM Sequences
**Suitability Rating**: [2-STAR] (for current dataset)

**Advantages**:
- **Semantic Understanding**: Captures semantics of URL, title, visible text
- **Context Modeling**: Understands text phishing keywords ("verify account", "urgent")

**Disadvantages**:
- **Dataset Has No Raw Text**: Currently only has length_of_text, length_of_title statistics, **no original text content**
- **Over-engineering**: For already extracted statistical features, Transformer is overkill
- **Training Cost**: Requires large text corpora for pretraining or fine-tuning
- **Slow Inference**: Not suitable for real-time detection

**Dataset Match**: **15%** (current dataset not applicable)
**Implementation Recommendation**: **Not recommended currently**. If recrawling and saving original HTML content later, consider:
- Using DistilBERT to process title + visible text
- Fusing with tabular features for multimodal model

---

#### 6.3 Graph Neural Networks (GNN) on DOM Trees
**Suitability Rating**: [1-STAR] (for current dataset)

**Advantages**:
- **Structural Modeling**: Fully preserves DOM tree hierarchical relationships
- **Node Relationships**: Captures parent-child, sibling node patterns
- **Research Value**: Academic innovation potential

**Disadvantages**:
- **Dataset Has No DOM Tree**: Current data is **flattened statistical features**, **no tree structure data**
- **Complex Implementation**: Requires PyTorch Geometric, graph construction, node feature engineering
- **Training Difficulty**: Many hyperparameters, complex batch processing
- **Poor Interpretability**: Hard to explain learned patterns

**Dataset Match**: **0%** (current dataset not applicable)
**Implementation Recommendation**: **Not recommended**, unless:
1. Project goal is exploratory research
2. Willing to recollect and parse complete DOM tree structures

---

### (7) Model Combination (Stacking / Hybrid)
**Suitability Rating**: [4-STAR]

**Advantages**:
- **Performance Boost**: Combines different model strengths, typically 1-3% accuracy improvement
- **Robustness**: Reduces single model failure risk
- **Competition-ready**: Common technique in Kaggle competitions

**Implementation Recommendation**:
**Recommended for later stages** as performance optimization:
1. **Level 1**: Train Random Forest + LightGBM + MLP
2. **Level 2**: Use Level 1 prediction probabilities as features, train Logistic Regression meta-classifier

**Example Workflow**:
```python
# Level 1 models
rf_probs = random_forest.predict_proba(X_val)[:, 1]
lgb_probs = lightgbm.predict_proba(X_val)[:, 1]
mlp_probs = mlp.predict_proba(X_val)[:, 1]

# Level 2 meta features
meta_X = np.column_stack([rf_probs, lgb_probs, mlp_probs])

# Level 2 meta classifier
meta_clf = LogisticRegression().fit(meta_X, y_val)
```

---

## Final Recommendations (Priority Ranked)

### Phase 1: Rapid Baseline Validation (Days 1-2)
**Mandatory Models**:
1. **Logistic Regression** [4-STAR]
   - Purpose: Validate data loading, feature engineering pipeline
   - Expected: Quick 70-75% accuracy baseline
   - Tools: scikit-learn

2. **Random Forest** [5-STAR]
   - Purpose: Establish strong baseline, validate feature importance
   - Expected: 80-85% accuracy
   - Tools: scikit-learn

---

### Phase 2: Performance Optimization (Days 3-5)
**Primary Models**:
3. **LightGBM / XGBoost** [5-STAR]
   - Purpose: Pursue best performance
   - Expected: 85-90% accuracy, likely final best model
   - Tuning Focus: learning_rate, max_depth, num_leaves
   - Tools: LightGBM (recommended) or XGBoost

4. **MLP (PyTorch)** [4-STAR]
   - Purpose: Meet PyTorch requirement, explore deep learning potential
   - Expected: 78-83% accuracy (may underperform tree models)
   - Note: Must perform feature standardization
   - Tools: PyTorch

---

### Phase 3: Model Ensemble (Days 6-7)
**Optional Optimization**:
5. **Stacking Ensemble** [4-STAR]
   - Combination: Random Forest + LightGBM + MLP
   - Meta-learner: Logistic Regression or LightGBM
   - Expected: 1-3 percentage point improvement

---

### NOT Recommended Models (for Current Dataset)
- **kNN**: Curse of dimensionality, slow inference
- **Naive Bayes**: Violates independence assumption
- **Single Decision Tree**: Severe overfitting
- **SVM**: Slow training, low cost-effectiveness
- **CNN/ViT**: No image data
- **BERT/Transformer**: No raw text data
- **GNN**: No DOM tree structure data

---

## Performance Prediction Table

| Model | Expected Accuracy | Training Time | Inference Speed | Interpretability | Implementation Difficulty | Overall Score |
|-------|------------------|---------------|-----------------|------------------|--------------------------|---------------|
| **LightGBM** | 85-90% | Medium (1-5 min) | Fast | High (SHAP) | Low | [5-STAR] |
| **Random Forest** | 80-85% | Medium (2-10 min) | Medium | High | Low | [5-STAR] |
| **XGBoost** | 85-90% | Medium (2-8 min) | Fast | High (SHAP) | Low | [5-STAR] |
| **MLP** | 78-83% | Fast (10-60 sec) | Very Fast | Low | Medium | [4-STAR] |
| **Logistic Reg** | 70-75% | Very Fast (<10 sec) | Very Fast | Very High | Very Low | [4-STAR] |
| **Stacking** | 86-92% | Long (cumulative) | Slow | Low | High | [4-STAR] |
| **SVM** | 75-80% | Long (10-60 min) | Slow | Medium | Medium | [3-STAR] |
| **Decision Tree** | 65-75% | Fast | Fast | Very High | Low | [3-STAR] |
| **kNN** | 70-78% | None | Very Slow | Low | Low | [2-STAR] |
| **Naive Bayes** | 60-70% | Very Fast | Very Fast | Medium | Very Low | [2-STAR] |

---

## Implementation Roadmap

### Week 1: Core Model Development
```
Day 1-2: Data preprocessing + Logistic Regression + Random Forest
Day 3-4: LightGBM hyperparameter tuning
Day 5-6: PyTorch MLP implementation
Day 7:   Model comparison evaluation, select best approach
```

### Week 2: Optimization & Extension (Optional)
```
Day 8-9:  Stacking ensemble
Day 10-11: SHAP interpretability analysis
Day 12-13: Hyperparameter fine-tuning (Optuna/Grid Search)
Day 14:    Final report & deployment preparation
```

---

## Feature Engineering Recommendations

While current dataset features are extracted, further enhancement possible:

### High-Value Feature Engineering (Recommended)

1. **URL Feature Extraction** (from URL column):
   - URL length
   - Domain length
   - Subdomain depth
   - Contains IP address (boolean)
   - Is URL shortener (bit.ly, tinyurl, etc.)
   - HTTPS vs HTTP
   - Brand keyword matching (google, paypal, bank, etc.)
   - Special character count (@, -, _)



### Evaluation Metrics
Recommended metric priority:
1. **AUC-ROC**: Overall classification capability
2. **F1-Score**: Balance precision and recall
3. **Recall**: Reduce false negatives (phishing misclassified as legitimate)
4. **Precision**: Reduce false positives (legitimate misclassified as phishing)

Business Consideration: In phishing detection, **high Recall is more important** (better to flag safe than miss phishing)

---


```python
# Data processing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Baseline models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Best models
import lightgbm as lgb
# import xgboost as xgb  # Alternative

# Deep learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Interpretability
import shap

# Hyperparameter tuning
import optuna  # or use GridSearchCV
```