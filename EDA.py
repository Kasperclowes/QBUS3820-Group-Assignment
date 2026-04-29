import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# Plot settings
import seaborn as sns

#---------------------------------------------------------------------------------
#RETRIEVING DATA FROM COMPLETEJOURNEY_Py
import os

from xgboost import train

# Load data directly from parquet files
data_dir = 'completejourney_py/completejourney_py/data'
transactions = pd.read_parquet(os.path.join(data_dir, 'transactions.parquet'))
demographics = pd.read_parquet(os.path.join(data_dir, 'demographics.parquet'))
products = pd.read_parquet(os.path.join(data_dir, 'products.parquet'))
campaigns = pd.read_parquet(os.path.join(data_dir, 'campaigns.parquet'))
campaign_descriptions = pd.read_parquet(os.path.join(data_dir, 'campaign_descriptions.parquet'))
promotions = pd.read_parquet(os.path.join(data_dir, 'promotions.parquet'))
#coupons = pd.read_parquet(os.path.join(data_dir, 'coupons.parquet'))
#coupon_redemptions = pd.read_parquet(os.path.join(data_dir, 'coupon_redemptions.parquet'))

def retrieve_data():
    return transactions, demographics, products, campaigns, campaign_descriptions, promotions

#---------------------------------------------------------------------------------
#HISTOGRAMS, TRANSFORMATIONS AND PLOTS 

sns.set_style('ticks') # set default plot style
colors = ['#4E79A7','#F28E2C','#E15759','#76B7B2','#59A14F', 
          '#EDC949','#AF7AA1','#FF9DA7','#9C755F','#BAB0AB']
sns.set_palette(colors) # set custom color scheme
plt.rcParams['figure.figsize'] = (9, 6)

def plot_feature_distributions(df, features, bins=30):
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
    if len(features) == 1:
        axes = [axes]
    for ax, feature in zip(axes, features):
        sns.histplot(df[feature], bins=bins, kde=True, ax=ax)
        ax.set_title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.show()

def plot_feature_vs_target(df, features, target):
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
    if len(features) == 1:
        axes = [axes]
    for ax, feature in zip(axes, features):
        sns.regplot(x=df[feature], y=target, lowess=True, line_kws={'color': 'black', 'alpha': 0.6}, ax=ax)
        ax.set_title(f'{feature} vs. {target.name}')
    plt.tight_layout()
    plt.show()

def plot_churn_rate(churn_train, colours):
    
    plt.figure(figsize=(6, 4))
    sns.barplot(x=churn_train.value_counts().index, 
                y=churn_train.value_counts(normalize=True) * 100,
                palette=colours[:2])

    plt.title("Customer Churn Rate", fontsize=14)
    plt.ylabel("Percentage of Households (%)")
    plt.xlabel("Churn (0 = Retained, 1 = Churned)")
    plt.ylim(0, 100)
    plt.show()

def distplots(X, kde=True):

    labels = list(X.columns)
    
    N, p = X.shape

    rows = int(np.ceil(p/3)) 

    fig, axes = plt.subplots(rows, 3, figsize=(12, rows*(12/4)))

    for i, ax in enumerate(fig.axes):
        if i < p:
            sns.histplot(X.iloc[:,i], ax=ax, stat='density', kde=False, alpha= 0.9, edgecolor ='black')
            sns.kdeplot(X.iloc[:,i], ax=ax, alpha= 0.0, color='#333333')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(labels[i])
            ax.set_yticks([])
        else:
            fig.delaxes(ax)

    sns.despine()
    plt.tight_layout()
    
    return fig, axes

def regplots(X, y):
    colors = ['#4E79A7','#F28E2C','#E15759','#76B7B2','#59A14F', 
          '#EDC949','#AF7AA1','#FF9DA7',"#956D57",'#BAB0AB']
    sns.set_palette(colors) # set custom color scheme

    labels = list(X.columns)
    
    N, p = X.shape

    rows = int(np.ceil(p/3)) 

    fig, axes = plt.subplots(rows, 3, figsize=(12, rows*(11/4)))

    for i, ax in enumerate(fig.axes):
        if i < p:          
            sns.regplot(x = X.iloc[:,i], y = y,  ci=None, logistic=True, y_jitter=0.05, 
                        scatter_kws={'s': 25, 'alpha':.5},  color=colors[i % 10], ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(labels[i])
            ax.set_xlim(X.iloc[:,i].min(),X.iloc[:,i].max())
        else:
            fig.delaxes(ax)

    sns.despine()
    plt.tight_layout()

    return fig, axes

def boxplots(X, y):
    sns.set_palette(colors) # set custom color scheme

    labels = list(X.columns)
    
    N, p = X.shape

    rows = int(np.ceil(p/3)) 

    fig, axes = plt.subplots(rows, 3, figsize=(12, rows*(11/4)))

    for i, ax in enumerate(fig.axes):
        if i < p:          
            sns.boxplot(x=y, y=X.iloc[:,i], ax=ax)
            ax.set_xlabel(f'')
            ax.set_ylabel(f'')
            ax.set_yticks([])
            ax.set_title(labels[i])
        else:
            fig.delaxes(ax)

    sns.despine()
    plt.tight_layout()

    return fig, axes

def churn_stack_plot(X, y, features=None, column_orders=None):
    if features is None:
        features = list(X.columns)
    column_orders = column_orders or {}

    cols = min(3, len(features))
    rows = int(np.ceil(len(features) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        table = pd.crosstab(X[feature], y)
        if feature in column_orders:
            table = table.reindex(column_orders[feature])
        else:
            table = table.sort_index()
        table.columns = ['Retained', 'Churned']
        table.plot(kind='bar', stacked=True, ax=ax,
                   color=[colors[0], colors[2]], alpha=0.85, edgecolor='white')
        ax.set_title(feature)
        ax.set_xlabel('')
        ax.set_ylabel('N households')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=8)

    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    sns.despine()
    plt.tight_layout()
    return fig, axes

def scatterplot(selected_features, train, y_train, string_target):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, feature in enumerate(selected_features):
        sns.regplot(x=train[feature], y=y_train, lowess=True, line_kws={'color':'black', 'alpha':0.6}, ax=axes[i])
        axes[i].set_title(f'{feature} vs. {string_target}')
    plt.tight_layout()
    plt.show()

def merge_rare_categories(feature_train, feature_valid, feature_test, threshold):
    # Count occurrences of each category
    counts = feature_train.value_counts()

    # Replace rare categories with "Other"
    feature_train = feature_train.apply(
        lambda x: x if counts[x] >= threshold else 'Other'
    )
    feature_valid = feature_valid.apply(
        lambda x: x if x in feature_train.unique() else 'Other'
    )
    feature_test = feature_test.apply(
        lambda x: x if x in feature_train.unique() else 'Other'
    )
    return feature_train, feature_valid, feature_test

def crosstabplots(X, y, column_orders=None):
    colors = sns.color_palette()
    column_orders = column_orders or {}

    labels = list(X.columns)
    N, p = X.shape
    rows = int(np.ceil(p/3))

    fig, axes = plt.subplots(rows, 3, figsize=(12, rows*(12/4)))

    for i, ax in enumerate(fig.axes):
        if i < p:
            col_name = labels[i]
            table = pd.crosstab(y, X.iloc[:,i])
            table = (table/table.sum()).iloc[1,:]
            if col_name in column_orders:
                table = table.reindex(column_orders[col_name])
            else:
                table = table.sort_index()
            table.T.plot(kind='bar', alpha=0.8, ax=ax, color=colors[i % len(colors)])
            ax.set_title(col_name)
            ax.set_ylabel('')
            ax.set_xlabel('')
        else:
            fig.delaxes(ax)

    sns.despine()
    plt.tight_layout()
    return fig, axes

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

def mutual_information_table(data, target, continuous=None, discrete=None, categorical=None, binary=None):
    """
    Compute mutual information between features and target variable.

    Parameters:
    - data: DataFrame with all features and target
    - target: Target variable (Series or array)
    - continuous: List of continuous feature names
    - discrete: List of discrete feature names
    - categorical: List of categorical feature names
    - binary: List of binary feature names

    Returns:
    - DataFrame: MI scores for all features, sorted by importance
    """
    # Convert pandas Index to list if necessary
    continuous = list(continuous) if continuous is not None else []
    discrete = list(discrete) if discrete is not None else []
    categorical = list(categorical) if categorical is not None else []
    binary = list(binary) if binary is not None else []

    frames = []

    if len(continuous) > 0:
        mi = mutual_info_classif(data[continuous], target, random_state=1)
        frames.append(pd.DataFrame(mi, index=continuous, columns=['MI']))

    if len(discrete + categorical + binary) > 0:
        features = OrdinalEncoder().fit_transform(data[discrete + categorical + binary])
        mi = mutual_info_classif(features, target, discrete_features=True, random_state=1)
        frames.append(pd.DataFrame(mi, index=discrete + categorical + binary, columns=['MI']))
    
    mi_results = pd.concat(frames).sort_values('MI', ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=mi_results['MI'], y=mi_results.index, palette='viridis')
    plt.title('Mutual Information Scores for Features')
    plt.xlabel('Mutual Information Score')
    plt.ylabel('Feature')
    plt.show()

    return mi_results



def rocplot(y_test, y_probs, labels, sample_weight=None):
    
    fig, ax= plt.subplots(figsize=(9,6))

    N, M=  y_probs.shape

    for i in range(M):
        fpr, tpr, _ = roc_curve(y_test, y_probs[:,i], sample_weight=sample_weight)
        auc = roc_auc_score(y_test, y_probs[:,i], sample_weight=sample_weight)
        ax.plot(1-fpr, tpr, label=labels.iloc[i] + ' (AUC = {:.3f})'.format(auc))
    
    ax.plot([0,1],[1,0], linestyle='--', color='black', alpha=0.6)

    ax.set_xlabel('Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('ROC curves', fontsize=14)
    sns.despine()

    plt.legend(fontsize=13, loc ='lower left' )
    
    return fig, ax

def coefplot(model, labels):
    coef = model.coef_
    table = pd.Series(coef.ravel(), index = labels).sort_values(ascending=True, inplace=False)
    
    all_ = True
    if len(table) > 20:
        reference = pd.Series(np.abs(coef.ravel()), index = labels).sort_values(ascending=False, inplace=False)
        reference = reference.iloc[:20]
        table = table[reference.index]
        table = table.sort_values(ascending=True, inplace=False)
        all_ = False
        

    fig, ax = fig, ax = plt.subplots()
    table.T.plot(kind='barh', edgecolor='black', width=0.7, linewidth=.8, alpha=0.9, ax=ax)
    ax.tick_params(axis=u'y', length=0) 
    if all_:
        ax.set_title('Estimated coefficients', fontsize=14)
    else: 
        ax.set_title('Estimated coefficients (20 largest in absolute value)', fontsize=14)
    sns.despine()
    return fig, ax

from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, log_loss

def evaluate_models(models, model_names, X_valid, y_valid, decision_threshold, loss_fn=1, loss_fp=1):
    """
    Evaluates multiple classification models on the validation set using various performance metrics.

    Parameters:
    - models (list): List of trained classification models with `predict_proba` method.
    - model_names (list): List of model names
    - X_valid (DataFrame): Validation feature set.
    - y_valid (array): True labels for validation.
    - tau (float): Decision threshold for classification.

    Returns:
    - DataFrame: Comparison table with key metrics.
    - y_prob (array): Predicted probabilities for all models (for plotting).
    """
    
    columns = ['Decision Loss', 'SE', 'Sensitivity', 'Specificity', 'Precision', 'AUC', 'Cross-entropy']
    results = pd.DataFrame(0.0, columns=columns, index=model_names)

    y_valid = np.ravel(y_valid)  # Ensure correct shape
    y_prob = np.zeros((len(y_valid), len(models)))  # Store probabilities

    for i, model in enumerate(models):    
        y_prob[:, i] = model.predict_proba(X_valid)[:, 1]  # Extract default probability
        y_pred = (y_prob[:, i] > decision_threshold).astype(int)  # Apply threshold

        # Compute loss using predefined loss matrix (False Negative = 5, False Positive = 1)
        fn_mask = (y_pred != y_valid) & (y_pred == 0)  # False Negatives
        fp_mask = (y_pred != y_valid) & (y_pred == 1)  # False Positives
        loss = (loss_fn * fn_mask + loss_fp * fp_mask) # Average loss per loan

        # Compute confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()

        # Store results
        results.iloc[i, 0] = np.mean(loss)  # Average loss
        results.iloc[i, 1] = np.std(loss) / np.sqrt(len(y_valid))  # Standard error
        results.iloc[i, 2] = tp / (tp + fn)  # Sensitivity (Recall)
        results.iloc[i, 3] = tn / (tn + fp)  # Specificity
        results.iloc[i, 4] = precision_score(y_valid, y_pred)  # Precision
        results.iloc[i, 5] = roc_auc_score(y_valid, y_prob[:, i])  # AUC
        results.iloc[i, 6] = log_loss(y_valid, y_prob[:, i])  # Cross-entropy loss

    return results.round(3), y_prob



#---------------------------------------------------------------------------------
#GENERAL DATA CLEANING (EDA)

def household_split(transactions):
    unique_households = transactions['household_id'].unique()
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(unique_households)

    # First split: 70% train, 30% temp (valid + test combined)
    train_size = int(0.7 * len(unique_households))
    train_households = unique_households[:train_size]

    # Second split: Split the remaining 30% into 50% valid, 50% test
    temp_households = unique_households[train_size:]
    temp_size = len(temp_households)
    valid_size = int(0.5 * temp_size)
    valid_households = temp_households[:valid_size]
    test_households = temp_households[valid_size:]
    return train_households, valid_households, test_households

def clean_transactions(transactions):
    #GENERAL DATA CLEANING
    transactions.info()
    missing_counts = transactions.isnull().sum()
    print("Missing values in transactions: ", missing_counts)
    #retail disc, coupon disc and coupon match disc have alot of zero values
    continous_transactions = ["sales_value","retail_disc","coupon_disc","coupon_match_disc"]
    discrete_transactions = ["quantity"]
    nominal_categorical_transactions = ["household_id", "store_id", "basket_id", "product_id"]
    ordinal_categorical_transactions = ["week"]
    duplicates= transactions.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")

    for col in ordinal_categorical_transactions:
        print(f"\n{col.upper()} - Unique Values:")
        print(transactions[col].unique())

    print("Week number- Unique Values:")
    print(transactions['week'].unique())

#-------------------------------------------------------------------------
#FURTHER DATA CLEANING

#-------------------------------------------------------------------------
#SPLITTING TRANSACTION DATA BY HOUSHOLD ID


    train_households, valid_households, test_households = household_split(transactions)

    # Create transaction datasets based on household splits
    transactions_train = transactions[transactions['household_id'].isin(train_households)].copy()
    transactions_valid = transactions[transactions['household_id'].isin(valid_households)].copy()
    transactions_test = transactions[transactions['household_id'].isin(test_households)].copy()

    print(f"\nTransactions Split Summary:")
    print(f"\nTransaction counts:")
    print(f"Train transactions: {len(transactions_train)}")
    print(f"Valid transactions: {len(transactions_valid)}")
    print(f"Test transactions: {len(transactions_test)}")
    
    return transactions_train, transactions_valid, transactions_test

def plot_transactions(transactions): 
    continous_transactions = ["sales_value","retail_disc","coupon_disc","coupon_match_disc"]
    discrete_transactions = ["quantity"]
    nominal_categorical_transactions = ["household_id", "store_id", "basket_id", "product_id"]
    ordinal_categorical_transactions = ["week"]
    duplicates= transactions.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")

    #CONTINOUS VARIABLES: 
    print(transactions["sales_value"].max())
    # Print count of zero values in continuous columns
    print("\n" + "="*60)
    print("Zero Values Count in Continuous Columns:")
    print("="*60)
    for col in continous_transactions:
        zero_count = (transactions[col] == 0).sum()
        print(f"{col}: {zero_count} zero values")
    
    # Remove rows with zero values in continuous columns
    initial_rows = len(transactions)
    transactions = transactions[(transactions[continous_transactions] != 0).all(axis=1)]
    rows_removed = initial_rows - len(transactions)
    print(f"\nRows removed with zero values: {rows_removed}")
    print(f"Remaining transactions: {len(transactions)}")

    print(distplots(transactions[continous_transactions]))



def clean_demographics(demographics):
    
    demographics.info()
    missing_counts = demographics.isnull().sum()
    print(missing_counts)
    demographics = demographics.drop(['home_ownership', 'marital_status'], axis=1)
    ordinal_categorical_demographics = ['age', 'income', 'household_size', 'kids_count']
    nominal_categorical_demographics = ['household_comp']
    categorical_demographics = ordinal_categorical_demographics + nominal_categorical_demographics
    duplicates= demographics.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")

    #Checking unique values in ordinal categorical variables and nominal categorical variable
    for col in ordinal_categorical_demographics:
        print(f"\n{col.upper()} - Unique Values:")
        print(demographics[col].unique())

    print("HOUSEHOLD_COMP - Unique Values:")
    print(demographics['household_comp'].unique())

    train_households, valid_households, test_households = household_split(transactions)
    demographics_train = demographics[demographics['household_id'].isin(train_households)]
    demographics_valid = demographics[demographics['household_id'].isin(valid_households)]
    demographics_test  = demographics[demographics['household_id'].isin(test_households)]

    demographics_train.set_index('household_id', inplace=True)
    demographics_valid.set_index('household_id', inplace=True)
    demographics_test.set_index('household_id', inplace=True)

    return demographics_train, demographics_valid, demographics_test
    


def clean_promotions(promotions): 
#promotions doesnt include household ids, only product ids which could be traced back to 
#a household id, although this dataset has low relevance to our assignment.
    promotions.info()
    missing_counts = promotions.isnull().sum()
    print("Missing values in promotions : ", missing_counts)
    nominal_categorical_promotions = ["product_id", "store_id", "display_location", "mailer_location"]
    ordinal_categorical_promotions = ["week"]
    duplicates= promotions.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")

    for col in ordinal_categorical_promotions:
        print(f"\n{col.upper()} - Unique Values:")
        print(transactions[col].unique())

        print("Week number- Unique Values:")
        print(transactions['week'].unique())

