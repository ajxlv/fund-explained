"""One row of behaviour features per fund (inputs to clustering and the label check)."""
import pandas as pd

from fundx import metrics as m

FEATURES = {
    "cagr_3y": "Compound annual growth over the last 3 years",
    "cagr_5y": "Compound annual growth over the last 5 years",
    "vol_3y": "Annualised volatility of daily returns, last 3 years",
    "max_dd_5y": "Worst peak-to-trough fall within the last 5 years",
    "beta_3y": "Beta vs the Nifty 50 proxy on daily returns, last 3 years",
    "corr_nifty_3y": "Correlation with the Nifty 50 proxy on daily returns, last 3 years",
    "downside_vol_3y": "Annualised downside deviation (falls only), last 3 years",
    "tracking_error_3y": "Annualised std of (fund - Nifty) daily returns, last 3 years",
}


def fund_features(nav: pd.Series, returns: pd.Series, bench_returns: pd.Series) -> dict:
    r3 = m.last_years(returns, 3)
    dd = m.max_drawdown(nav, 5)
    return {
        "cagr_3y": m.cagr(nav, 3),
        "cagr_5y": m.cagr(nav, 5),
        "vol_3y": m.volatility(r3),
        "max_dd_5y": dd.drawdown if dd else None,
        "beta_3y": m.beta_vs_benchmark(returns, bench_returns, 3),
        "corr_nifty_3y": m.correlation_vs_benchmark(returns, bench_returns, 3),
        "downside_vol_3y": m.downside_volatility(r3),
        "tracking_error_3y": m.tracking_error(returns, bench_returns, 3),
    }


def build_features(matrix: pd.DataFrame, returns: pd.DataFrame, bench_code: int) -> pd.DataFrame:
    """Features for every fund. Missing values stay NaN (e.g. too-short history)."""
    bench_r = returns[bench_code]
    rows = {code: fund_features(matrix[code], returns[code], bench_r) for code in matrix.columns}
    df = pd.DataFrame.from_dict(rows, orient="index", columns=list(FEATURES)).astype(float)
    df.index.name = "scheme_code"
    return df
