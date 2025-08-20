"""
Utilities for centris plex analysis.

This package contains utility functions for data preparation, 
stepwise regression, and model analysis.
"""

from .data_preparation import (
    prepare_plex_data,
    get_feature_groups,
    prepare_modeling_variables,
    create_formula_scope,
    haversine
)

from .stepAIC import step_aic, StepwiseResult

# Create step_criterion as an alias to step_aic for backwards compatibility
def step_criterion(data, initial, scope, criterion='aic', direction='both', **kwargs):
    """
    Wrapper function for step_aic that matches the step_criterion interface.
    
    Parameters
    ----------
    data : pd.DataFrame
        Model data
    initial : str
        Starting formula
    scope : str
        Upper scope formula
    criterion : str, default='aic'
        Selection criterion ('aic' or 'bic')
    direction : str, default='both'
        Step direction
    **kwargs
        Additional arguments passed to step_aic
        
    Returns
    -------
    StepwiseResult
        Result object with model and anova table
    """
    # Convert criterion to k parameter for step_aic
    if criterion.lower() == 'bic':
        # For BIC, use k = log(n) where n is number of observations
        import numpy as np
        n = len(data)
        k = np.log(n)
    else:
        k = 2.0  # Standard AIC
    
    return step_aic(
        data=data,
        initial=initial, 
        scope=scope,
        direction=direction,
        k=k,
        **kwargs
    )

__all__ = [
    'prepare_plex_data',
    'get_feature_groups', 
    'prepare_modeling_variables',
    'create_formula_scope',
    'haversine',
    'step_aic',
    'step_criterion',
    'StepwiseResult'
]
