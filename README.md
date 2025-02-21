# Analysis-of-Industrial-Accidents-with-the-ARIA-Database
This project aims to analyze the ARIA database to identify the root causes of industrial accidents and improve risk management. The approach combines descriptive analysis, statistical tests, NLP, and supervised classification models to predict the root causes of incidents.

## 📊 Exploratory Analysis
We extracted the key columns relevant to the analysis:
- **Type of accident**: 4.55% of values are 'not reported'
- **Event type**: 2.09% of values are 'not reported'
- **Materials**: 5.65% of values are 'not reported'
- **Equipment**: 47.14% of values are 'not reported'
- **CLP hazard class**: 13.04% of values are 'not reported'
- **Root causes**: 53.29% of values are 'not reported'
- **Disturbances**: 47.68% of values are 'not reported'
- **Consequences**: 6.71% of values are 'not reported'
- **12% of materials** are marked as "unknown"

## 🔬 Correlation Analysis (Cramér’s V)
Correlation tests show **weak associations** between variables. The strongest correlations are observed between:
- **Accident type and Event type** (0.37)
- **Materials and CLP hazard class** (0.37)
- **Accident type and Materials** (0.31)
These results indicate that **relationships between variables are limited**, and no dominant factor explains the incidents.

## 🧠 Multiple Correspondence Analysis (MCA)
- **70% of variance** is spread over **230 axes**, indicating **high data diversity**.
- To simplify interpretation, we kept the **top 180 axes**, explaining **60% of inertia**.
- Results show that **accidents are influenced by a wide combination of factors**, with no small subset of axes capturing all information.

## 📝 NLP Analysis: Term Frequency
We analyzed the **most frequent terms** in accident descriptions:
- **Most common bigrams**:
  - "fire declare" (7555 occurrences)
  - "set up" (5252 occurrences)
  - "safety perimeter" (3149 occurrences)
  - "technical unemployment" (2529 occurrences)
  - "extinguish fire" (2525 occurrences)
- **Main themes identified**:
  - 🔥 **Fires and explosions** ("fire", "incident", "declare", "extinguish")
  - 🚧 **Risk management and procedures** ("safety perimeter", "procedure guideline", "risk identification")
  - ⛽ **Hazardous substances** ("gas", "domestic fuel", "gas leak")
  - 💼 **Economic consequences** ("technical unemployment")

## 🔍 Root Cause Analysis
- **The dominant root cause is "risk management"**, representing a large portion of incidents.
- **A significant class imbalance** was observed, with some root causes being underrepresented.
- **Frequently cited causes include:**
  - "Procedure guideline"
  - "Insufficient feedback"
  - "Control organization"
  - "Personnel training"
- **Need for class rebalancing** by refining root cause classification.

## 🤖 Modeling and Prediction of Root Causes
### 🔹 Initial Model
- **Standard Naïve Bayes**: Accuracy = **18%**
- Issue: The model was biased toward the majority class and ignored minority classes.

### 🔹 Class Rebalancing
- **Grouping of minor classes (<20 occurrences)**
- **Undersampling of the majority class (max 5000 occurrences)**
- **Oversampling of minority classes using SMOTE (min 500 occurrences)**
- **Feature Engineering: TF-IDF Vectorization**

### 🔹 Final Model (ComplementNB)
- **Accuracy: 68.4%**
- **Macro F1-score: 70%**
- **Weighted F1-score: 64%**
- **Better balance between classes**

## 🚀 Conclusions and Future Improvements
✅ **Improved detection of root causes** by balancing classes.
✅ **More robust model after rebalancing**, but still room for improvement.
✅ **Next steps:**
  - Test **other models** (Random Forest, XGBoost, Transformers NLP)
  - Explore a **multi-label approach** to capture multiple causes.
  - Analyze **feature importance** with SHAP to better understand predictions.

🔥 **Project under continuous improvement!** 🚀
