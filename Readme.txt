================================================================================
    CUSTOMER CHURN PREDICTION MODEL - QBUS3820 Group Assignment
================================================================================

PROJECT OVERVIEW
================================================================================
This project predicts which customers (households) are likely to churn and stop 
shopping with a retailer in the next few weeks. By identifying at-risk customers 
early, the retailer can deploy targeted retention offers to bring them back and 
reduce customer acquisition costs.

Dataset Source: Complete Journey retail dataset
Reference: https://cunningjames.github.io/completejourney_py/user-guide/datasets/

Business Problem:
Without a predictive model, retailers have no way of knowing when a customer is 
drifting away until it's too late to act. This project develops a machine learning 
solution to identify churn risk customers based on transaction history, demographics, 
and behavioral patterns.


KEY FEATURES & APPROACH
================================================================================
Churn Definition:
- FIXED: A household is classified as churned if there are NO TRANSACTIONS 
  in the past 4 weeks
- This consistent definition is applied uniformly across all households for modeling

Key Predictive Variables:
1. RECENCY: Weeks/days since last purchase
2. FREQUENCY: Number of purchases in recent period (last 4-12 weeks)
3. MONETARY VALUE: Spend trends, average order value, weekly/monthly spend
4. VISIT DECLINE RATE: (Recent visits - Past visits) / Past visits
5. SPEND TRENDS: (Recent spend - Past spend) / Past spend
6. DEMOGRAPHICS: Income, household size (for segmentation)
7. COUPON ENGAGEMENT: Redemption rates, time since last redemption

Data Split:
- Training set: 70% of household data
- Validation set: 15% of household data
- Test set: 15% of household data
- Split by HOUSEHOLD ID (not transactions) for proper generalization


PYTHON FILES DESCRIPTION
================================================================================

1. EDA.py - EXPLORATORY DATA ANALYSIS
   Purpose: Initial data exploration and visualization
   Outputs: 
   - Summary statistics of transactions, demographics, and coupons
   - Distribution of purchase frequency, spending patterns
   - Churn rate by customer segment
   - Identifies which variables are most predictive of churn
   Usage: python EDA.py

2. Feature_eng.py - FEATURE ENGINEERING
   Purpose: Create predictive features for modeling
   Key Features Engineered:
   - Recency: Days/weeks since last purchase
   - Frequency: Transaction counts over rolling windows (7d, 30d, 90d)
   - Monetary aggregates: Total spend, AOV, weekly/monthly averages
   - Spend trends: Recent vs past spending comparison
   - Visit decline rates: Change in purchase frequency
   - Behavioral indicators: Coupon responsiveness, category diversity
   Outputs: Processed feature dataframe (train/val/test splits)
   Usage: python Feature_eng.py

3. CT_model_training.py - CLASSIFICATION TREE
   Purpose: Train decision tree classifier for churn prediction
   Type: Baseline model for performance comparison
   Hyperparameters: Tuned to minimize validation loss
   Outputs: Trained tree model, accuracy metrics, feature importance
   Usage: python CT_model_training.py

4. CNN.py - DEEP LEARNING MODEL (CNN/Neural Network)
   Purpose: Train deep neural network for churn classification
   Architecture: Fully connected layers with appropriate activation functions
   Outputs: Trained neural network model, loss curves, predictions
   Notes: Explores modern deep learning approaches for classification
   Usage: python CNN.py
   

JUPYTER NOTEBOOK
================================================================================

RUN.ipynb - MAIN EXECUTION NOTEBOOK
   Purpose: Integrated workflow running the complete analysis pipeline
   Structure:
   - Data loading and preprocessing
   - Call EDA.py for exploratory analysis
   - Call Feature_eng.py for feature creation
   - Model training and evaluation (CT, neural networks, etc.)
   - Model comparison and performance analysis
   - Generate predictions on test set
   - Business insights and recommendations
   Usage: jupyter notebook RUN.ipynb
   
   This is the PRIMARY file to run for complete analysis.


DOCUMENTATION FILES
================================================================================

Assignment.txt - Project Requirements
   - Business problem statement
   - Data split requirements
   - Churn definition details (4 weeks no transaction rule)
   - Report structure requirements
   - Model development plan

Feature_eng.txt - Feature Engineering Guide
   - Detailed feature descriptions and derivations
   - Rationale for including each feature
   - Feature priority ranking
   - Examples of feature calculations

tree02 - Decision Tree Model Output
   Directory containing trained tree model files/visualizations


PACKAGE: completejourney_py/
================================================================================
This is a Python package for accessing and working with the Complete Journey data.

Key Files:
- get_data.py: Functions to load and access dataset
- data/: Contains the raw dataset files (transactions, demographics, etc.)
- docs/: Documentation and example notebooks
- tests/: Unit tests for the package

Key Datasets Available:
- Transactions: purchase history with timestamps, amounts, products
- Demographics: household info (income, household size, location)
- Products: product details and categories
- Coupons: coupon offers and redemptions
- Campaigns: marketing campaigns sent to households


WORKFLOW
================================================================================

Step 1: EXPLORE DATA
   → Run EDA.py to understand data distributions and patterns
   
Step 2: ENGINEER FEATURES
   → Run Feature_eng.py to create predictive features
   → Output: Processed train/val/test datasets
   
Step 3: BUILD MODELS
   → Run CT_model_training.py for baseline classification tree
   → Run CNN.py for deep learning approach
   → Tune hyperparameters to minimize validation loss
   
Step 4: EVALUATE & COMPARE
   → Compare model performance on validation and test sets
   → Analyze feature importance
   → Select best performing model
   
Step 5: GENERATE INSIGHTS
   → Profile at-risk customers
   → Estimate business impact of retention efforts
   → Make recommendations


EXPECTED OUTPUTS
================================================================================
Model Deliverables:
✓ Trained classification tree model (baseline)
✓ Trained deep learning model
✓ Hyperparameter optimization results
✓ Test set predictions with churn probability scores
✓ Feature importance rankings
✓ Performance metrics (accuracy, precision, recall, F1, AUC)

Analysis Deliverables:
✓ Exploratory data analysis visualizations
✓ Feature engineering documentation
✓ Model comparison report
✓ Business insights and churn risk profiles
✓ Implementation recommendations with ROI analysis


REQUIREMENTS & DEPENDENCIES
================================================================================
Core Libraries:
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning models (trees, forests, regression)
- tensorflow/keras: Deep learning models
- matplotlib/seaborn: Data visualization

Installation:
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn

For Jupyter Notebooks:
pip install jupyter


NOTES & BEST PRACTICES
================================================================================
1. TRAIN/VAL/TEST SPLIT
   - Always split by HOUSEHOLD ID, not individual transactions
   - This prevents data leakage (model learning household signatures)
   - Ensures model generalizes to new households

2. FIXED CHURN DEFINITION
   - All households use the same churn threshold: no transactions in past 4 weeks
   - Provides consistent, interpretable churn classification
   - Enables straightforward model training and business communication

3. FEATURE IMPORTANCE
   - Monitor which features most influence churn predictions
   - Recency and frequency typically most important
   - Spend trends and visit decline rates capture momentum

4. CLASS IMBALANCE
   - Churn is often rare event (< 20% of customers)
   - Use appropriate metrics: precision, recall, F1-score, AUC
   - Consider class weight balancing in models

5. BUSINESS VALIDATION
   - Validate model predictions with domain expertise
   - Check if flagged at-risk customers make sense contextually
   - Estimate retention offer costs vs customer lifetime value


CONTACT & NOTES
================================================================================
Last Updated: May 26, 2026
Course: QBUS3820 - Machine Learning in Business Analytics
University of Sydney

For questions about the dataset, see:
https://cunningjames.github.io/completejourney_py/

================================================================================
