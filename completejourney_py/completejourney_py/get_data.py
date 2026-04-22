from typing import Dict, List, Optional, Sequence, Union
import pandas as pd
import sys
import os

def get_data(which: Optional[Union[str, Sequence[str]]] = None) -> Dict[str, pd.DataFrame]:
    """Load datasets from the Complete Journey grocery transaction data.
    
    The Complete Journey dataset contains grocery store shopping transactions 
    from 2,469 households over one year, provided by 84.51°. Includes transaction 
    records, demographics, marketing campaigns, and coupon data.

    Args:
        which: Specify which dataset(s) to load. Options:
            - None (default): Load all 8 datasets
            - str: Load single dataset by name
            - Sequence[str]: Load multiple datasets by name
            
            Available datasets:
            - 'transactions': Purchase records (1.47M records)
            - 'demographics': Household demographic information
            - 'products': Product metadata and categories  
            - 'campaigns': Marketing campaigns per household
            - 'campaign_descriptions': Campaign metadata
            - 'promotions': Product placement in mailers/stores
            - 'coupons': Coupon metadata (UPC codes, campaigns)
            - 'coupon_redemptions': Detailed coupon usage records

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping dataset names to pandas 
            DataFrames. Each DataFrame contains the requested dataset with 
            appropriate column names and data types.

    Raises:
        FileNotFoundError: If specified dataset name doesn't exist.
        TypeError: If 'which' parameter is not None, str, or sequence of strings.

    Examples:
        Load all datasets:
        
        >>> data = get_data()
        >>> len(data)
        8
        
        Load single dataset:
        
        >>> transactions = get_data("transactions")["transactions"]
        >>> transactions.shape[0] > 1000000
        True
        
        Load multiple datasets:
        
        >>> sales_data = get_data(["transactions", "products", "demographics"])
        >>> list(sales_data.keys())
        ['transactions', 'products', 'demographics']
        
    Note:
        Data source: 84.51° "The Complete Journey" dataset
        Available at: http://www.8451.com/area51/
    """

    sources: List[str] = ["campaign_descriptions",
                          "coupons",
                          "promotions",
                          "campaigns",
                          "demographics",
                          "transactions",
                          "coupon_redemptions",
                          "products"]

    if which is None:
        which = sources
    elif isinstance(which, str):
        which = [which]
    else:
        which = list(which)

    def load_dataset(src: str) -> pd.DataFrame:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(script_dir, "data", f"{src}.parquet")
        return pd.read_parquet(data_file)
    
    return dict(map(lambda src: (src, load_dataset(src)), which))
