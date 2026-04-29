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

def transaction_frequency(transactions):
    transaction_freq = transactions.groupby('household_id').size()
    return transaction_freq

def average_basket_size(transactions):
    basket_size = (transactions.groupby(['household_id', 'basket_id'])['sales_value']
               .sum()
               .groupby('household_id')
               .mean())
    return basket_size


def spend_trend(transactions):
    transactions['transaction_datetime'] = pd.to_datetime(transactions['transaction_timestamp'])
    #we want spend trend for each houshold id to be spend trend = recent spend (last 3 weeks)- past spend (previous 4-8 weeks)/ past spend (previous 4-8 weeks)
    transactions['week'] = transactions['transaction_datetime'].dt.isocalendar().week
    transactions['year'] = transactions['transaction_datetime'].dt.year
    # Define recent and past periods
    recent_period = transactions['transaction_datetime'] >= (transactions['transaction_datetime'].max() - pd.Timedelta(weeks=3))
    past_period = (transactions['transaction_datetime'] < (transactions['transaction_datetime'].max() - pd.Timedelta(weeks=3))) 
    past_period &= (transactions['transaction_datetime'] >= (transactions['transaction_datetime'].max() - pd.Timedelta(weeks=8)))
    # Calculate spend in recent and past periods
    recent_spend = transactions[recent_period].groupby('household_id')['sales_value'].sum()
    past_spend = transactions[past_period].groupby('household_id')['sales_value'].sum()
    # Calculate spend trend
    spend_trend = (recent_spend - past_spend) / past_spend.replace(0, np.nan)  # Avoid division by zero
    spend_trend = spend_trend.fillna(0)  # Replace NaN with 0 (no change if past spend is zero)
    spend_trend = spend_trend.to_frame()
    spend_trend.columns = ['spend_trend']
    return spend_trend

#spend trend is a measure of how much a household's spending has changed recently compared to the past. 
#A positive trend indicates increased spending, while a negative trend indicates decreased spending.