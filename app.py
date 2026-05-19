import argparse
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    date: pd.Timestamp
    action: str
    price: float
    units: float
    cash: float


def download_sp500(start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit("Missing dependency yfinance. Install with: pip install yfinance") from exc

    df = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit("No data downloaded for S&P 500")
    df = df.rename_axis("Date").reset_index()[["Date", "Close"]]
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def confirm_pivots(close: pd.Series, k: int = 3) -> pd.DataFrame:
    """Causal pivot confirmation.

    At day t, we can confirm if day t-k was a local min/max over [t-2k, t].
    This avoids lookahead in execution: signal is emitted at confirmation day t.
    """
    n = len(close)
    piv = pd.DataFrame(index=close.index, data={"pivot_type": 0, "pivot_price": np.nan, "pivot_idx": np.nan})

    for t in range(2 * k, n):
        c = t - k
        window = close.iloc[t - 2 * k : t + 1]
        center_price = close.iloc[c]
        if center_price == window.min():
            piv.iloc[t, piv.columns.get_loc("pivot_type")] = 1
            piv.iloc[t, piv.columns.get_loc("pivot_price")] = center_price
            piv.iloc[t, piv.columns.get_loc("pivot_idx")] = c
        elif center_price == window.max():
            piv.iloc[t, piv.columns.get_loc("pivot_type")] = -1
            piv.iloc[t, piv.columns.get_loc("pivot_price")] = center_price
            piv.iloc[t, piv.columns.get_loc("pivot_idx")] = c
    return piv


def backtest_elliott(df: pd.DataFrame, initial_cash: float = 10000.0, k: int = 3) -> tuple[pd.DataFrame, List[Trade]]:
    data = df.copy()
    piv = confirm_pivots(data["Close"], k=k)
    data = pd.concat([data, piv.reset_index(drop=True)], axis=1)

    cash = initial_cash
    units = 0.0
    position = 0
    trades: List[Trade] = []
    confirmed: List[tuple[int, int, float, pd.Timestamp]] = []  # idx, type, price, confirm_date

    values = []

    for i, row in data.iterrows():
        date = row["Date"]
        price = float(row["Close"])

        if row["pivot_type"] != 0:
            confirmed.append((int(row["pivot_idx"]), int(row["pivot_type"]), float(row["pivot_price"]), date))
            if len(confirmed) > 30:
                confirmed = confirmed[-30:]

        signal: Optional[str] = None

        if len(confirmed) >= 8:
            last8 = confirmed[-8:]
            pattern = [p[1] for p in last8]
            prices = [p[2] for p in last8]

            impulse_up = pattern == [1, -1, 1, -1, 1, -1, 1, -1]
            if impulse_up:
                h1, l2, h3, l4, h5, a, b, c = prices
                valid_rules = (
                    l2 > prices[0] * 0.85
                    and h3 > h1
                    and l4 > h1
                    and h5 >= h3 * 0.98
                    and c <= a
                )
                if valid_rules:
                    impulse_range = h5 - l2
                    retrace = (h5 - c) / impulse_range if impulse_range > 0 else 0
                    fib_zone = 0.35 <= retrace <= 0.68
                    if fib_zone:
                        signal = "BUY"

        # trend exit: break below last confirmed low pivot since entry
        if position == 1 and len(confirmed) >= 1:
            lows = [p[2] for p in confirmed if p[1] == 1]
            if lows and price < lows[-1] * 0.99:
                signal = "SELL"

        if signal == "BUY" and position == 0:
            units = cash / price
            cash = 0.0
            position = 1
            trades.append(Trade(date, "BUY", price, units, cash))
        elif signal == "SELL" and position == 1:
            cash = units * price
            trades.append(Trade(date, "SELL", price, units, cash))
            units = 0.0
            position = 0

        values.append(cash + units * price)

    data["equity_elliott"] = values
    return data, trades


def backtest_buy_hold(df: pd.DataFrame, initial_cash: float = 10000.0) -> pd.Series:
    first = float(df["Close"].iloc[0])
    units = initial_cash / first
    return df["Close"] * units


def backtest_dca(df: pd.DataFrame, initial_cash: float = 10000.0, freq: str = "M") -> pd.Series:
    data = df.copy()
    data = data.set_index("Date")
    schedule = data.resample(freq).last().index
    contrib = initial_cash / max(len(schedule), 1)
    cash = 0.0
    units = 0.0
    equity = []
    for date, row in data.iterrows():
        if date in schedule:
            cash += contrib
            units += cash / row["Close"]
            cash = 0.0
        equity.append(units * row["Close"] + cash)
    return pd.Series(equity, index=df.index)


def performance(equity: pd.Series, dates: pd.Series) -> dict:
    ret = equity.pct_change().fillna(0)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() / (ret.std() + 1e-12)) * np.sqrt(252)
    dd = equity / equity.cummax() - 1
    return {
        "final_value": float(equity.iloc[-1]),
        "total_return": float(total),
        "cagr": float(cagr),
        "volatility": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd.min()),
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest Elliott-like strategy vs DCA and Buy&Hold on S&P 500")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--pivot-k", type=int, default=3, help="Confirmation lag bars for pivot detection")
    parser.add_argument("--out", default="backtest_results.csv")
    args = parser.parse_args()

    df = download_sp500(args.start, args.end)
    elliott_df, trades = backtest_elliott(df, initial_cash=args.cash, k=args.pivot_k)
    elliott_df["equity_buy_hold"] = backtest_buy_hold(df, initial_cash=args.cash)
    elliott_df["equity_dca"] = backtest_dca(df, initial_cash=args.cash)

    perf = {
        "elliott": performance(elliott_df["equity_elliott"], elliott_df["Date"]),
        "buy_hold": performance(elliott_df["equity_buy_hold"], elliott_df["Date"]),
        "dca": performance(elliott_df["equity_dca"], elliott_df["Date"]),
    }

    print("=== PERFORMANCE ===")
    for name, metrics in perf.items():
        print(f"\n{name.upper()}")
        for k, v in metrics.items():
            if "value" in k:
                print(f"  {k:14s}: ${v:,.2f}")
            else:
                print(f"  {k:14s}: {v:.2%}")

    print(f"\nTrades Elliott: {len(trades)}")
    if trades:
        print("Last 5 trades:")
        for t in trades[-5:]:
            print(f"  {t.date.date()} {t.action:4s} @ {t.price:.2f}")

    elliott_df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
