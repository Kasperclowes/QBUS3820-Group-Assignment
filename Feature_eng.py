import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# Plot settings
import seaborn as sns
import EDA as eda

#---------------------------------------------------------------------------------
#RETRIEVING DATA FROM COMPLETEJOURNEY_Py
import os

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

def calculate_days_between_trips(transactions):
    """Calculate days between consecutive shopping trips for a household"""
    # Sort by household and date to ensure chronological order within each household
    household_trips = transactions.sort_values(['household_id', 'transaction_timestamp']).copy()
    
    # Convert to datetime for proper date arithmetic
    household_trips['datetime_date'] = pd.to_datetime(household_trips['transaction_timestamp'])
    
    # Calculate days since last trip WITHIN each household
    household_trips['days_since_last_trip'] = household_trips.groupby('household_id')['datetime_date'].diff().dt.days
    
    return household_trips

def transactions_cut(transactions, cutoff):
    cutoff_date = transactions['transaction_timestamp'].max() - pd.Timedelta(weeks=cutoff)
    transactions_cut = transactions[transactions['transaction_timestamp'] <= cutoff_date]
    return transactions_cut


def churn(transactions, threshold_days=21):
    """
    Adds a binary churn column to the transactions dataframe.
    
    Parameters:
        transactions   : the transactions dataframe
        threshold_days : number of days without a purchase to classify as churn
                         defaults to 21 days (3 weeks)
    
    Returns:
        transactions with a new 'churn' column (1 = churned, 0 = not churned)
    """
    transactions = transactions.copy()
    transactions['transaction_datetime'] = pd.to_datetime(transactions['transaction_timestamp'])

    # Find the last purchase date per household
    last_purchase = transactions.groupby('household_id')['transaction_datetime'].max().reset_index()
    last_purchase.columns = ['household_id', 'last_purchase_date']

    # Find the end of the dataset
    dataset_end_date = transactions['transaction_datetime'].max()

    # Calculate days since last purchase
    last_purchase['days_since_last_purchase'] = (
        dataset_end_date - last_purchase['last_purchase_date']
    ).dt.days

    # Assign churn: 1 if days since last purchase >= threshold, else 0
    last_purchase['churn'] = (
        last_purchase['days_since_last_purchase'] >= threshold_days
    ).astype(int)

    train_households, valid_households, test_households = eda.household_split(transactions)
    churn = last_purchase.set_index('household_id')['churn']
    churn_train = churn[churn.index.isin(train_households)]
    churn_valid = churn[churn.index.isin(valid_households)]
    churn_test  = churn[churn.index.isin(test_households)]

    return churn, churn_train, churn_valid, churn_test

def adaptive_churn(transactions, k=1): 
    houshold_trips = calculate_days_between_trips(transactions)
    # find the average days between trips for each household and put into a new column
    household_avg_days = houshold_trips.groupby('household_id')['days_since_last_trip'].mean().reset_index()

    transactions = transactions.copy()
    transactions['transaction_datetime'] = pd.to_datetime(transactions['transaction_timestamp'])

    # Find the last purchase date per household
    last_purchase = transactions.groupby('household_id')['transaction_datetime'].max().reset_index()
    last_purchase.columns = ['household_id', 'last_purchase_date']

    # Find the end of the dataset
    dataset_end_date = transactions['transaction_datetime'].max()

    # Calculate days since last purchase
    last_purchase['days_since_last_purchase'] = (
        dataset_end_date - last_purchase['last_purchase_date']
    ).dt.days

    # Merge to match each household with its own average days between trips
    last_purchase = last_purchase.merge(household_avg_days, on='household_id', how='left')

    # Assign churn: 1 if days since last purchase >= average days between trips + k*std_dev, else 0
    last_purchase['churn'] = (
        last_purchase['days_since_last_purchase'] >= last_purchase['days_since_last_trip'] + k * last_purchase['days_since_last_trip'].std()
    ).astype(int)

    train_households, valid_households, test_households = eda.household_split(transactions)
    churn = last_purchase.set_index('household_id')['churn']
    churn_train = churn[churn.index.isin(train_households)]
    churn_valid = churn[churn.index.isin(valid_households)]
    churn_test  = churn[churn.index.isin(test_households)]

    return churn, churn_train, churn_valid, churn_test

def collapse_income_categories(demographics_train, demographics_valid, demographics_test):
    income_collapse = {
        '100-124K': '100K+',
        '125-149K': '100K+',
        '150-174K': '100K+',
        '175-199K': '100K+',
        '200-249K': '100K+',
        '250K+':    '100K+',
    }
    demographics_train['income'] = demographics_train['income'].replace(income_collapse)
    demographics_valid['income'] = demographics_valid['income'].replace(income_collapse)
    demographics_test['income']  = demographics_test['income'].replace(income_collapse)

    return demographics_train['income'], demographics_valid['income'], demographics_test['income']  

def total_spend(transactions):
    total_spend = transactions.groupby('household_id')['sales_value'].sum()
    return total_spend

def total_transactions(transactions):
    transaction_freq = transactions.groupby('household_id').size()
    return transaction_freq

def average_basket_size(transactions):
    basket_size = (transactions.groupby(['household_id', 'basket_id'])['sales_value']
               .sum()
               .groupby('household_id')
               .mean())
    return basket_size

def department_diversity(transactions, products):
    transactions = transactions.merge(products[['product_id', 'department']], on='product_id', how='left')
    department_diversity = transactions.groupby('household_id')['department'].nunique()
    return department_diversity

def days_since_last_purchase(transactions):
    """
    Calculate days since last purchase for each household.
    
    Parameters:
        transactions : transactions dataframe
    
    Returns:
        Series with days since last purchase indexed by household_id
    """
    transactions = transactions.copy()
    transactions['transaction_datetime'] = pd.to_datetime(transactions['transaction_timestamp'])
    
    # Find the end of the dataset
    dataset_end_date = transactions['transaction_datetime'].max()
    
    # Find the last purchase date per household
    last_purchase = transactions.groupby('household_id')['transaction_datetime'].max()
    
    # Calculate days since last purchase
    days_since_last = (dataset_end_date - last_purchase).dt.days
    
    return days_since_last


def spend_trend(transactions):
    transactions['transaction_datetime'] = pd.to_datetime(transactions['transaction_timestamp'])
    
    # Get all unique household_ids in the dataset
    all_households = transactions['household_id'].unique()
    
    # Define recent and past periods (transactions already have past 3 weeks removed)
    # Recent: last 0-4 weeks
    # Past: 4-8 weeks ago
    recent_period = (transactions['transaction_datetime'] >= (transactions['transaction_datetime'].max() - pd.Timedelta(weeks=4)))
    past_period = (transactions['transaction_datetime'] >= (transactions['transaction_datetime'].max() - pd.Timedelta(weeks=8))) & (transactions['transaction_datetime'] < (transactions['transaction_datetime'].max() - pd.Timedelta(weeks=4)))
    # Calculate spend in recent and past periods
    recent_spend = transactions[recent_period].groupby('household_id')['sales_value'].sum()
    past_spend = transactions[past_period].groupby('household_id')['sales_value'].sum()
    # Calculate spend trend
    spend_trend = (recent_spend - past_spend) / past_spend.replace(0, np.nan)  # Avoid division by zero
    spend_trend = spend_trend.fillna(0)  # Replace NaN with 0 (no change if past spend is zero)
    spend_trend = spend_trend.to_frame()
    spend_trend.columns = ['spend_trend']
    
    # Find households with no transactions in the period (churned) and assign -1
    households_with_trends = set(spend_trend.index)
    missing_households = set(all_households) - households_with_trends
    if len(missing_households) > 0:
        missing_df = pd.DataFrame({'spend_trend': -1.0}, index=pd.Index(missing_households, name='household_id'))
        spend_trend = pd.concat([spend_trend, missing_df])
    
    return spend_trend

#spend trend is a measure of how much a household's spending has changed recently compared to the past. 
#A positive trend indicates increased spending, while a negative trend indicates decreased spending.

def visit_trend(transactions):
    transactions['transaction_datetime'] = pd.to_datetime(transactions['transaction_timestamp'])
    
    # Get all unique household_ids in the dataset
    all_households = transactions['household_id'].unique()
    
    # Define recent and past periods (transactions already have past 3 weeks removed)
    # Recent: last 0-4 weeks
    # Past: 4-8 weeks ago
    recent_period = (transactions['transaction_datetime'] >= (transactions['transaction_datetime'].max() - pd.Timedelta(weeks=4)))
    past_period = (transactions['transaction_datetime'] >= (transactions['transaction_datetime'].max() - pd.Timedelta(weeks=8))) & (transactions['transaction_datetime'] < (transactions['transaction_datetime'].max() - pd.Timedelta(weeks=4)))
    # Calculate number of visits (transactions) in recent and past periods
    recent_visits = transactions[recent_period].groupby('household_id').size()
    past_visits = transactions[past_period].groupby('household_id').size()
    # Calculate visit trend
    visit_trend = (recent_visits - past_visits) / past_visits.replace(0, np.nan)  # Avoid division by zero
    visit_trend = visit_trend.fillna(0)  # Replace NaN with 0 (no change if past visits is zero)
    visit_trend = visit_trend.to_frame()
    visit_trend.columns = ['visit_trend']
    
    # Find households with no transactions in the period (churned) and assign -1
    households_with_trends = set(visit_trend.index)
    missing_households = set(all_households) - households_with_trends
    if len(missing_households) > 0:
        missing_df = pd.DataFrame({'visit_trend': -1.0}, index=pd.Index(missing_households, name='household_id'))
        visit_trend = pd.concat([visit_trend, missing_df])
    
    return visit_trend

#visit trend is a measure of how much a household's visit frequency has changed recently compared to the past. 
#A positive trend indicates increased visit frequency, while a negative trend indicates decreased visit frequency.

def campaign_features(campaign_descriptions, campaigns, index):

    campaign_descriptions = campaign_descriptions.copy()


    campaign_hh = campaigns.merge(campaign_descriptions, on='campaign_id')

    agg = campaign_hh.groupby('household_id').agg(
        num_campaigns=('campaign_id', 'nunique'),
    )

    type_counts = (
        campaign_hh.groupby(['household_id', 'campaign_type'])
        .size()
        .unstack(fill_value=0)
    )
    type_counts.columns = [f'num_campaign_{c.lower().replace(" ", "_")}' for c in type_counts.columns]

    return agg.join(type_counts).reindex(index, fill_value=0)

def promotion_features(promotions, transactions):
    promo_transactions = transactions.merge(
        promotions, on=['product_id', 'store_id', 'week'], how='left'
    )
    promo_transactions['on_display'] = promo_transactions['display_location'] != '0'
    promo_transactions['on_mailer'] = promo_transactions['mailer_location'] != '0'
    promo_transactions['on_promotion'] = promo_transactions['on_display'] | promo_transactions['on_mailer']

    return promo_transactions.groupby('household_id').agg(
        promo_purchase_rate=('on_promotion', 'mean'),
        display_purchase_rate=('on_display', 'mean'),
        mailer_purchase_rate=('on_mailer', 'mean'),
    )



def build_features(transactions, campaigns, campaign_descriptions, promotions, spend_trend_data=None, visit_trend_data=None):
    """
    Concatenate all features to create X_train for decision tree.
    
    Parameters:
        transactions              : transactions dataframe
        campaigns                 : campaigns dataframe
        campaign_descriptions     : campaign_descriptions dataframe
        promotions               : promotions dataframe
        spend_trend_data         : pre-computed spend_trend (optional, computed if None)
        visit_trend_data         : pre-computed visit_trend (optional, computed if None)
    
    Returns:
        DataFrame with all features concatenated, indexed by household_id
    """
    # Get list of all households from transactions
    all_households = transactions['household_id'].unique()
    
    # Compute individual features
    total_spend_feat = total_spend(transactions).to_frame(name='total_spend')
    total_transactions_feat = total_transactions(transactions).to_frame(name='total_transactions')
    avg_basket_feat = average_basket_size(transactions).to_frame(name='average_basket_size')
    days_since_last_purchase_feat = days_since_last_purchase(transactions).to_frame(name='days_since_last_purchase')
    
    # Use pre-computed trends if provided, otherwise compute
    if spend_trend_data is None:
        spend_trend_feat = spend_trend(transactions)
    else:
        spend_trend_feat = spend_trend_data
    
    if visit_trend_data is None:
        visit_trend_feat = visit_trend(transactions)
    else:
        visit_trend_feat = visit_trend_data
    
    # Get campaign and promotion features
    campaign_feat = campaign_features(campaign_descriptions, campaigns, all_households)
    promotion_feat = promotion_features(promotions, transactions)
    
    # Reindex promotion features to include all households (fill missing with 0)
    promotion_feat = promotion_feat.reindex(all_households, fill_value=0)
    
    # Concatenate all features along columns (axis=1)
    X = pd.concat([
        total_spend_feat,
        total_transactions_feat,
        avg_basket_feat,
        days_since_last_purchase_feat,
        spend_trend_feat,
        visit_trend_feat,
        campaign_feat,
        promotion_feat
    ], axis=1)
    
    # Ensure all households are represented (fill missing rows with 0)
    X = X.reindex(all_households) 
    #instead of filling missing values, output error
    if X.isnull().any().any():
        raise ValueError("Missing values found in feature matrix X. Please check the data and feature engineering steps.")
    
    return X

