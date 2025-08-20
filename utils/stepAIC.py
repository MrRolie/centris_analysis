import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


@dataclass
class StepwiseResult:
    model: object                      # statsmodels Results object
    anova: pd.DataFrame                # R-like step path
    keep: Optional[pd.DataFrame] = None


def _lhs_rhs(formula: str) -> Tuple[str, str]:
    lhs, rhs = formula.split("~", 1)
    return lhs.strip(), rhs.strip()


def _has_no_intercept(rhs: str) -> bool:
    # Accept both '-1' and '0' as "no intercept" signals
    # Keep simple textual detection; avoids Patsy internals
    tokens = [t.strip() for t in rhs.replace("\n", " ").split("+")]
    return any(tok in {"-1", "0"} for tok in tokens)


def _parse_terms(rhs: str) -> List[str]:
    """
    Split RHS into literal formula tokens separated by '+', stripping whitespace.
    Keeps interaction tokens (':', '*', 'I(...)', 'C(...)', etc.) intact.
    Removes empty pieces and intercept toggles ('1', '0', '-1').
    """
    parts = [t.strip() for t in rhs.replace("\n", " ").split("+")]
    cleaned = []
    for t in parts:
        if not t:
            continue
        if t in {"1", "0", "-1"}:
            continue
        cleaned.append(t)
    return cleaned


def _build_formula(lhs: str, terms: List[str], no_intercept: bool) -> str:
    if no_intercept:
        rhs = "-1"
        if terms:
            rhs = " -1 + " + " + ".join(terms)
    else:
        rhs = "1"
        if terms:
            rhs = "1 + " + " + ".join(terms)
    return f"{lhs} ~ {rhs}"


def _fit(formula: str,
         data: pd.DataFrame,
         family=None,
         **fit_kwargs):
    if family is None:
        model = smf.ols(formula, data=data)
    else:
        model = smf.glm(formula, data=data, family=family)
    res = model.fit(**fit_kwargs)
    return res


def _aic_k(result, k: float) -> float:
    # statsmodels' .aic uses k=2; compute generic k explicitly
    llf = getattr(result, "llf", None)
    if llf is None or not np.isfinite(llf):
        raise ValueError("Model log-likelihood not available; cannot compute AIC.")
    return -2.0 * llf + k * len(result.params)


def _my_deviance(result) -> float:
    # Mirror R’s mydeviance: use model deviance if present; else -2*llf
    dev = getattr(result, "deviance", None)
    if dev is not None and np.isfinite(dev):
        return float(dev)
    llf = getattr(result, "llf", None)
    if llf is not None and np.isfinite(llf):
        return float(-2.0 * llf)
    return np.nan


def step_aic(
    data: pd.DataFrame,
    initial: str,
    scope: Optional[Union[str, Dict[str, str]]] = None,
    *,
    direction: str = "both",            # "both", "backward", "forward"
    trace: int = 1,
    keep: Optional[Callable[[object, float], dict]] = None,
    steps: int = 1000,
    k: float = 2.0,
    family=None,                        # e.g., sm.families.Binomial()
    fit_kwargs: Optional[dict] = None,
) -> StepwiseResult:
    """
    Stepwise model selection by AIC, similar to R's MASS::stepAIC.

    Parameters
    ----------
    data : DataFrame
        Model data.
    initial : str
        Starting formula, e.g., 'y ~ x1 + x2'.
    scope : None | str | {'lower': str, 'upper': str}
        Search space. If a single formula string is given, it's treated as 'upper'.
        If None, only backward elimination from `initial`.
    direction : {'both','backward','forward'}
        Step direction.
    trace : int
        If >0, print progress and candidate tables.
    keep : callable | None
        A function f(result, aic) -> dict to record custom info along the path.
    steps : int
        Max number of steps.
    k : float
        AIC penalty (k=2 is standard AIC; set to log(n) for BIC-like behavior).
    family : statsmodels GLM family | None
        If provided, fits a GLM; otherwise OLS.
    fit_kwargs : dict | None
        Extra kwargs passed to .fit().

    Returns
    -------
    StepwiseResult(model=final_result, anova=path_table, keep=optional_df)
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    direction = direction.lower()
    if direction not in {"both", "backward", "forward"}:
        raise ValueError("direction must be 'both', 'backward', or 'forward'.")

    # Parse starting formula
    lhs, rhs0 = _lhs_rhs(initial)
    no_intercept0 = _has_no_intercept(rhs0)
    current_terms = _parse_terms(rhs0)

    # Parse scope
    if scope is None:
        lower_terms = []
        upper_terms = current_terms[:]  # backward only
        no_intercept_upper = no_intercept0
        no_intercept_lower = no_intercept0
        forward_allowed = direction in {"both", "forward"} and False  # matches R's behavior when scope missing
    else:
        if isinstance(scope, str):
            # treat as 'upper'
            _, rhs_u = _lhs_rhs(scope)
            upper_terms = _parse_terms(rhs_u)
            no_intercept_upper = _has_no_intercept(rhs_u)
            lower_terms = []
            no_intercept_lower = no_intercept0
        elif isinstance(scope, dict):
            lower_terms = []
            upper_terms = []
            no_intercept_lower = no_intercept0
            no_intercept_upper = no_intercept0
            if scope.get("lower") is not None:
                _, rhs_l = _lhs_rhs(scope["lower"])
                lower_terms = _parse_terms(rhs_l)
                no_intercept_lower = _has_no_intercept(rhs_l)
            if scope.get("upper") is not None:
                _, rhs_u = _lhs_rhs(scope["upper"])
                upper_terms = _parse_terms(rhs_u)
                no_intercept_upper = _has_no_intercept(rhs_u)
        else:
            raise TypeError("scope must be None, a formula string, or a dict with 'lower'/'upper'.")

        # If intercept settings differ between lower/upper/current, we respect the current model's intercept flag
        forward_allowed = direction in {"both", "forward"}

    backward_allowed = direction in {"both", "backward"}

    # Fit initial model
    current_formula = _build_formula(lhs, current_terms, no_intercept0)
    fit = _fit(current_formula, data, family=family, **fit_kwargs)
    base_aic = _aic_k(fit, k)
    nobs0 = int(fit.nobs)

    if trace:
        print(f"Start:  AIC={base_aic:.3f}")
        print(current_formula, "\n")

    # Path storage (mimics R's table)
    models_path = [{
        "change": "",
        "deviance": _my_deviance(fit),
        "df.resid": int(fit.df_resid),
        "AIC": base_aic
    }]
    keep_records = []
    if keep is not None:
        keep_records.append(keep(fit, base_aic))

    tol = 1e-7
    steps_remaining = steps

    # Helper to check droppable wrt lower
    def can_drop(terms: List[str], t: str) -> bool:
        if not lower_terms:
            return True
        after = set(terms) - {t}
        return set(lower_terms).issubset(after)

    # Helper to list candidates
    def candidates(terms: List[str]) -> Tuple[List[str], List[str]]:
        # drops: any current term whose removal keeps >= lower
        drops = [t for t in terms if can_drop(terms, t)]
        # adds: any term in upper not already included
        adds = [t for t in upper_terms if t not in terms]
        return drops, adds

    while steps_remaining > 0:
        steps_remaining -= 1

        best_change = None
        best_model = None
        best_aic = base_aic
        best_deviance = None
        best_df_resid = None

        drops, adds = candidates(current_terms)

        rows = []

        # Backward step(s)
        if backward_allowed and drops:
            for t in drops:
                new_terms = [x for x in current_terms if x != t]
                new_formula = _build_formula(lhs, new_terms, no_intercept0)
                try:
                    res = _fit(new_formula, data, family=family, **fit_kwargs)
                    aic_val = _aic_k(res, k)
                    rows.append(("-", t, aic_val))
                    if aic_val + tol < best_aic:
                        best_change = f"- {t}"
                        best_model = res
                        best_aic = aic_val
                        best_deviance = _my_deviance(res)
                        best_df_resid = int(res.df_resid)
                        new_best_terms = new_terms
                except Exception as e:
                    rows.append(("-", t, np.nan))
                    warnings.warn(f"Drop '{t}' failed: {e}")

        # Forward step(s)
        if forward_allowed and adds:
            for t in adds:
                new_terms = current_terms + [t]
                # Respect intercept setting of current model
                new_formula = _build_formula(lhs, new_terms, no_intercept0)
                try:
                    res = _fit(new_formula, data, family=family, **fit_kwargs)
                    aic_val = _aic_k(res, k)
                    rows.append(("+", t, aic_val))
                    if aic_val + tol < best_aic:
                        best_change = f"+ {t}"
                        best_model = res
                        best_aic = aic_val
                        best_deviance = _my_deviance(res)
                        best_df_resid = int(res.df_resid)
                        new_best_terms = new_terms
                except Exception as e:
                    rows.append(("+", t, np.nan))
                    warnings.warn(f"Add '{t}' failed: {e}")

        # If no moves considered, exit
        if not rows:
            break

        # Show candidates sorted by AIC
        cand_df = pd.DataFrame(rows, columns=["Op", "Term", "AIC"]).sort_values("AIC", na_position="last")
        if trace:
            print(cand_df.to_string(index=False))

        # No improvement? stop
        if best_change is None or not np.isfinite(best_aic) or best_aic >= base_aic - tol:
            break

        # Improve and continue
        fit = best_model
        current_terms = new_best_terms
        base_aic = best_aic

        # nobs consistency check (R stops; we warn)
        nobs_new = int(fit.nobs)
        if nobs_new != nobs0:
            warnings.warn(
                f"Number of observations changed from {nobs0} to {nobs_new}. "
                "This typically means missing values differ across models."
            )

        if trace:
            print(f"\nStep:  AIC={base_aic:.3f}")
            print(_build_formula(lhs, current_terms, no_intercept0), "\n")

        models_path.append({
            "change": best_change,
            "deviance": best_deviance,
            "df.resid": best_df_resid,
            "AIC": best_aic
        })
        if keep is not None:
            keep_records.append(keep(fit, best_aic))

    # Build R-like path table
    rd = [m["deviance"] for m in models_path]
    rdf = [m["df.resid"] for m in models_path]
    dd = [np.nan] + [abs(rd[i] - rd[i - 1]) if np.isfinite(rd[i]) and np.isfinite(rd[i - 1]) else np.nan
                     for i in range(1, len(rd))]
    ddf = [np.nan] + [abs(rdf[i] - rdf[i - 1]) if np.isfinite(rdf[i]) and np.isfinite(rdf[i - 1]) else np.nan
                      for i in range(1, len(rdf))]
    path = pd.DataFrame({
        "Step": [m["change"] for m in models_path],
        "Df": ddf,
        "Deviance": dd,
        "Resid. Df": rdf,
        "Resid. Dev": rd,
        "AIC": [m["AIC"] for m in models_path],
    })

    keep_df = None
    if keep_records:
        # Stack dicts to a tidy DataFrame (columns=what your keep() returns)
        keep_df = pd.DataFrame(keep_records)

    return StepwiseResult(model=fit, anova=path, keep=keep_df)
