"""
Data preparation utilities for plex real estate analysis.

This module contains functions to prepare and transform raw centris data
for modeling and analysis, including feature engineering and standardization.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Optional, Tuple, List
import os
import json
import platform



def get_terminal_path_vars() -> dict:
    """
    Retrieve path-related variables from the VSCode terminal.integrated.env.* setting,
    automatically detecting the OS (osx, windows, linux).

    Returns
    -------
    dict
        Dictionary of path-related variables (e.g., PATH, PYTHONPATH) if found, else empty dict.
    """

    # Detect OS for VSCode setting key
    system = platform.system().lower()
    if system == "windows":
        env_var = "terminal.integrated.env.windows"
    else:
        env_var = "terminal.integrated.env.osx"

    # Try to get from environment variable first
    env_json = os.environ.get(env_var)
    if env_json:
        try:
            env_dict = json.loads(env_json)
            path_vars = {k: v for k, v in env_dict.items() if "PATH" in k}
            return path_vars
        except Exception:
            pass

    # Try to get from VSCode settings.json if available
    if system == "windows":
        settings_path = os.path.expandvars(r"%APPDATA%\Code\User\settings.json")
    elif system == "darwin":
        settings_path = os.path.expanduser("~/Library/Application Support/Code/User/settings.json")
    else:
        settings_path = os.path.expanduser("~/.config/Code/User/settings.json")

    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r") as f:
                settings = json.load(f)
            env_dict = settings.get(env_var, {})
            path_vars = {k: v for k, v in env_dict.items() if "PATH" in k}
            return path_vars
        except Exception:
            pass

    return {}


def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Parameters
    ----------
    lat1, lng1 : float
        Latitude and longitude of first point in decimal degrees
    lat2, lng2 : float
        Latitude and longitude of second point in decimal degrees
        
    Returns
    -------
    float
        Distance in kilometers
    """
    # Radius of Earth in kilometers
    R = 6371.0
    lat1, lng1, lat2, lng2 = np.radians([lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def prepare_plex_data(
    df: pd.DataFrame,
    target_year: int = 2025,
    assessment_growth_rate: float = 0.057,
    mtl_lat: float = 45.525098,
    mtl_lng: float = -73.647596,
    standardize: bool = True,
    drop_original_geo: bool = True
) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """
    Prepare plex real estate data for analysis and modeling.
    
    This function performs comprehensive data preparation including:
    - Assessment value adjustment for inflation
    - Geographic feature engineering (distance from Montreal)
    - Building style encoding
    - Financial ratio calculations
    - Property characteristic features
    - Optional standardization
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw centris data with required columns:
        ['mls', 'lat', 'lng', 'address', 'price', 'building_style', 'year_built', 
         'walkscore', 'lot_area_sqft', 'units_count', 'gross_potential_income', 
         'assessment_total', 'assessment_year']
    target_year : int, default=2025
        Year to adjust assessment values to
    assessment_growth_rate : float, default=0.057
        Annual growth rate for assessment value adjustment (5.7%)
    mtl_lat, mtl_lng : float, default=(45.525098, -73.647596)
        Montreal coordinates for distance calculation
    standardize : bool, default=True
        Whether to standardize numerical features
    drop_original_geo : bool, default=True
        Whether to drop original lat, lng, address columns
        
    Returns
    -------
    Tuple[pd.DataFrame, Optional[StandardScaler]]
        - Prepared dataframe with engineered features
        - StandardScaler object if standardize=True, else None
        
    Examples
    --------
    >>> df = pd.read_csv('centris_data.csv')
    >>> prepared_data, scaler = prepare_plex_data(df)
    >>> print(prepared_data.columns)
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Adjust assessment values to target year
    print(f"Adjusting assessment values to {target_year} using {assessment_growth_rate:.1%} annual growth...")
    data['assessment_total_adj'] = data.apply(
        lambda row: row['assessment_total'] * ((1 + assessment_growth_rate) ** (target_year - row['assessment_year']))
        if pd.notnull(row['assessment_total']) and pd.notnull(row['assessment_year']) else np.nan,
        axis=1
    )
    
    # 2. Select and prepare core columns
    required_cols = ['mls', 'lat', 'lng', 'address', 'price', 'building_style', 'year_built', 
                    'walkscore', 'lot_area_sqft', 'units_count', 'gross_potential_income', 'assessment_total_adj']
    
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    data = data[required_cols]
    
    # 3. Standardize building style labels
    print("Standardizing building style labels...")
    data['building_style'] = data['building_style'].replace({
        'Détaché': 'Detached',
        'En rangée': 'Row', 
        'Jumelé': 'Semi_detached'
    })
    
    # 4. Set MLS as index
    data.set_index('mls', inplace=True)
    
    # 5. Create geographic features
    print("Creating geographic features...")
    data['distance_from_mtl'] = data.apply(
        lambda row: haversine(row['lat'], row['lng'], mtl_lat, mtl_lng)
        if pd.notnull(row['lat']) and pd.notnull(row['lng']) else np.nan,
        axis=1
    )
    
    # 6. Drop original geographic columns if requested
    if drop_original_geo:
        data = data.drop(['lat', 'lng', 'address'], axis=1)
    
    # 7. One-hot encode building style
    print("Encoding building styles...")
    building_dummies = pd.get_dummies(data['building_style'], prefix='', prefix_sep='')
    data = pd.concat([data.drop('building_style', axis=1), building_dummies], axis=1)
    
    # 8. Create financial and property features
    print("Creating financial and property features...")
    
    # Revenue yield (annual income / purchase price as percentage)
    data['revenue_yield'] = (data['gross_potential_income'] / data['price']) * 100
    
    # Price per unit (assume 2 units if missing)
    data['price_per_unit'] = data['price'] / data['units_count'].fillna(2)
    
    # Price per square foot (lot area)
    data['price_per_sqft'] = data['price'] / data['lot_area_sqft']
    
    # Income per unit
    data['income_per_unit'] = data['gross_potential_income'] / data['units_count'].fillna(2)
    
    # Property age
    current_year = datetime.now().year
    data['property_age'] = current_year - data['year_built']
    
    # 9. List all created features
    new_features = [
        'revenue_yield', 'price_per_unit', 'price_per_sqft', 
        'income_per_unit', 'property_age', 'distance_from_mtl'
    ] + building_dummies.columns.tolist()
    
    print(f"New features created: {new_features}")
    
    # 10. Standardize numerical features if requested
    scaler = None
    if standardize:
        print("Standardizing numerical features...")
        num_cols = data.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        data[num_cols] = scaler.fit_transform(data[num_cols])
        print(f"Standardized columns: {num_cols.tolist()}")
    
    return data, scaler


def get_feature_groups(data: pd.DataFrame) -> dict:
    """
    Categorize features into logical groups for analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        Prepared dataset from prepare_plex_data()
        
    Returns
    -------
    dict
        Dictionary with feature categories
    """
    all_cols = data.columns.tolist()
    
    # Core property features
    core_features = [col for col in ['price', 'year_built', 'lot_area_sqft', 'units_count'] if col in all_cols]
    
    # Financial features
    financial_features = [col for col in ['gross_potential_income', 'assessment_total_adj', 'revenue_yield', 
                                         'price_per_unit', 'price_per_sqft', 'income_per_unit'] if col in all_cols]
    
    # Geographic features  
    geographic_features = [col for col in ['walkscore', 'distance_from_mtl'] if col in all_cols]
    
    # Building style dummies
    building_styles = [col for col in ['Detached', 'Row', 'Semi_detached'] if col in all_cols]
    
    # Derived features
    derived_features = [col for col in ['property_age'] if col in all_cols]
    
    return {
        'core': core_features,
        'financial': financial_features,
        'geographic': geographic_features,
        'building_styles': building_styles,
        'derived': derived_features
    }


def prepare_modeling_variables(
    data: pd.DataFrame, 
    target_col: str = 'price',
    exclude_cols: Optional[List[str]] = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    Prepare variable lists for stepwise modeling.
    
    Parameters
    ----------
    data : pd.DataFrame
        Prepared dataset
    target_col : str, default='price'
        Target variable column name
    exclude_cols : List[str], optional
        Additional columns to exclude from modeling
        
    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        - Numerical variables (excluding target)
        - Categorical variables  
        - Suggested interaction terms
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Numerical variables (excluding target and specified exclusions)
    num_vars = [col for col in data.select_dtypes(include=[np.number]).columns 
                if col != target_col and col not in exclude_cols]
    
    # Categorical variables (building styles)
    cat_vars = [col for col in ["Detached", "Row", "Semi_detached"] if col in data.columns]
    
    # Suggested interaction terms based on domain knowledge
    interaction_terms = []
    if 'units_count' in data.columns and 'revenue_yield' in data.columns:
        interaction_terms.append("units_count:revenue_yield")
    if 'units_count' in data.columns and 'gross_potential_income' in data.columns:
        interaction_terms.append("units_count:gross_potential_income")
    
    return num_vars, cat_vars, interaction_terms

def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae}
    
def create_formula_scope(
    target_col: str,
    num_vars: List[str], 
    cat_vars: List[str],
    interaction_terms: List[str]
) -> str:
    """
    Create formula scope for stepwise regression.
    
    Parameters
    ----------
    target_col : str
        Target variable name
    num_vars : List[str]
        Numerical variable names
    cat_vars : List[str] 
        Categorical variable names
    interaction_terms : List[str]
        Interaction term specifications
        
    Returns
    -------
    str
        Formula scope string for stepwise selection
    """
    all_terms = num_vars + cat_vars + interaction_terms
    return f"{target_col} ~ " + " + ".join(all_terms)


def find_discounted_properties(
    model,
    data: pd.DataFrame,
    price_col: str = 'price',
    feature_cols: Optional[List[str]] = None,
    discount_threshold: float = -50000,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Find properties that are priced below the model's prediction (potential bargains).
    
    Parameters
    ----------
    model : sklearn model
        Trained regression model with predict() method
    data : pd.DataFrame
        Dataset with features and actual prices (must include MLS index or column)
    price_col : str, default='price'
        Name of the price column
    feature_cols : List[str], optional
        List of feature columns to use for prediction. If None, uses all columns except price_col
    discount_threshold : float, default=-50000
        Minimum discount amount (negative value means price < prediction)
        e.g., -50000 means at least $50k below predicted price
    top_n : int, default=20
        Number of top discounted properties to return
        
    Returns
    -------
    pd.DataFrame
        DataFrame with discounted properties, sorted by discount amount, containing:
        - All original columns
        - predicted_price: Model prediction
        - discount_amount: Actual price - predicted price (negative = discount)
        - discount_percent: Percentage discount
        
    Examples
    --------
    >>> discounted = find_discounted_properties(xgb_model, prepared, discount_threshold=-25000, top_n=10)
    >>> print(f"Found {len(discounted)} properties with at least $25k discount")
    """
    # Prepare features
    if feature_cols is None:
        feature_cols = [col for col in data.columns if col != price_col]
    
    X = data[feature_cols].copy()
    y_actual = data[price_col].copy()
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate discount metrics
    results = data.copy()
    results['predicted_price'] = y_pred
    results['discount_amount'] = y_actual - y_pred
    results['discount_percent'] = (results['discount_amount'] / y_pred) * 100
    
    # Filter for discounted properties (negative discount_amount means actual < predicted)
    discounted = results[results['discount_amount'] <= discount_threshold].copy()
    
    # Sort by biggest discounts (most negative)
    discounted = discounted.sort_values('discount_amount', ascending=True)
    
    # Return top N
    return discounted.head(top_n)


def analyze_market_opportunities(
    discounted_properties: pd.DataFrame,
    top_n: int = 5,
    raw: Optional[pd.DataFrame] = None
) -> None:
    """
    Print a formatted analysis of discounted properties.
    
    Parameters
    ----------
    discounted_properties : pd.DataFrame
        DataFrame from find_discounted_properties()
    top_n : int, default=5
        Number of top opportunities to display
    """
    if discounted_properties.empty:
        print("No discounted properties found with the specified criteria.")
        return
    
    print(f"\n{'='*80}")
    print(f"TOP {top_n} MARKET OPPORTUNITIES - Properties Below Predicted Value")
    print(f"{'='*80}\n")
    # If raw is provided, merge to get listing_url and address
    merged = discounted_properties.head(top_n).copy()
    if raw is not None and 'mls' in raw.columns:
        merged = merged.reset_index() if 'mls' not in merged.columns else merged
        merged = merged.merge(raw[['mls', 'listing_url', 'address']], on='mls', how='left')
        merged = merged.set_index('mls')

    for idx, (mls_or_idx, row) in enumerate(merged.iterrows(), 1):
        print(f"{idx}. MLS: {mls_or_idx}")
        if 'listing_url' in row and pd.notnull(row['listing_url']):
            print(f"   Listing URL: {row['listing_url']}")
        if 'address' in row and pd.notnull(row['address']):
            print(f"   Address: {row['address']}")
        print(f"   Actual Price:    $ {row['price']:<12,.0f}")
        print(f"   Predicted Price: $ {row['predicted_price']:<12,.0f}")
        print(f"   Discount:        $ {abs(row['discount_amount']):<12,.0f} ({row['discount_percent']:.1f}%)")

        # Show key property details if available
        if 'units_count' in row:
            print(f"   Units: {row['units_count']:.0f}", end="")
        if 'year_built' in row:
            print(f" | Built: {row['year_built']:.0f}", end="")
        if 'lot_area_sqft' in row:
            print(f" | Lot: {row['lot_area_sqft']:,.0f} sqft", end="")
        print("\n")
