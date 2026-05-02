import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from itertools import combinations
from datetime import datetime

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Momentum + Corr Strategy Lab", layout="wide")
st.title("Momentum + Correlation Strategy (Equal Slots vs Inverse-Vol)")
st.caption("Monthly momentum selection + daily correlation diversification. Includes official holdings, mid-month peek, stats, yearly returns, and downloads.")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Universe")

DEFAULT_TICKERS = "PDBC,VNQ,TLT,BWX,EWJ,IEMG,SCZ,SPY,QQQ,GLD,REM,IEF,VNQI,TIP,VGK,SHY"
tickers_str = st.sidebar.text_area("Tickers (comma-separated)", DEFAULT_TICKERS, height=90)
TICKERS = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

CASH_PROXY = st.sidebar.text_input("Cash proxy ticker", "SHY").strip().upper()

st.sidebar.header("Dates")
START_DATE = st.sidebar.text_input("Start date (YYYY-MM-DD)", "2007-01-01").strip()
END_DATE = st.sidebar.text_input("End date (YYYY-MM-DD, blank = today)", "").strip()

st.sidebar.header("Strategy: Selection")
PORTFOLIO_SLOTS = st.sidebar.slider("Slots to hold (max weight per asset = 1/slots)", 1, 10, 3, 1)
MOM_LOOKBACK = st.sidebar.slider("Momentum lookback (months)", 3, 18, 8, 1)
TOP_CANDIDATES = st.sidebar.slider("Top momentum candidates", 3, 30, 6, 1)
CORR_LB = st.sidebar.slider("Correlation lookback (trading days)", 10, 120, 20, 5)
REQUIRE_POSITIVE_MOMENTUM = st.sidebar.checkbox("Require positive momentum", True)

st.sidebar.header("Sizing")
SHOW_INVVOL = st.sidebar.checkbox("Compute inverse-vol sizing too", True)
VOL_LB = st.sidebar.slider("Inverse-vol lookback (days)", 10, 120, 20, 5)
VOL_CAP = st.sidebar.slider("Inverse-vol max weight cap", 0.10, 1.00, 0.45, 0.05)

st.sidebar.header("Performance")
RF_ANNUAL = st.sidebar.number_input("Risk-free rate (annual, for Sharpe)", min_value=0.0, max_value=0.20, value=0.0, step=0.005)

st.sidebar.header("Run")
RUN = st.sidebar.button("Run backtest")

# -----------------------------
# Helpers
# -----------------------------
def parse_date_or_today(s: str) -> pd.Timestamp:
    if not s:
        return pd.Timestamp.today().normalize()
    return pd.to_datetime(s).normalize()

def first_trading_day_on_or_after(index: pd.DatetimeIndex, d: pd.Timestamp):
    idx = index[index >= d]
    return idx.min() if len(idx) else None

def last_trading_day_on_or_before(index: pd.DatetimeIndex, d: pd.Timestamp):
    idx = index[index <= d]
    return idx.max() if len(idx) else None

def month_end_prices(daily_px: pd.DataFrame) -> pd.DataFrame:
    return daily_px.resample("ME").last()

def pad_to_slots(selection, slots, cash_proxy):
    sel = list(selection) if selection else []
    while len(sel) < slots:
        sel.append(cash_proxy)
    return sel[:slots]

def weights_equal_slots(slot_list):
    slots = len(slot_list)
    w = {}
    for t in slot_list:
        w[t] = w.get(t, 0.0) + 1.0 / slots
    return w

def weights_dict_to_series(wdict, cols):
    s = pd.Series(0.0, index=cols)
    for t, wt in wdict.items():
        if t in s.index:
            s.loc[t] = float(wt)
    tot = float(s.sum())
    return (s / tot) if tot > 0 else s

def fmt_holdings(wser):
    nz = wser[wser > 1e-10].sort_values(ascending=False)
    return ", ".join([f"{t} {wt*100:.1f}%" for t, wt in nz.items()])

def turnover_one_way(w_old: pd.Series, w_new: pd.Series) -> float:
    return 0.5 * float((w_new.fillna(0.0) - w_old.fillna(0.0)).abs().sum())

def sells_buys(old_slots: list[str], new_slots: list[str], cash_proxy: str):
    old_set = set([x for x in old_slots if x != cash_proxy])
    new_set = set([x for x in new_slots if x != cash_proxy])
    return sorted(old_set - new_set), sorted(new_set - old_set)

def perf_stats(monthly_ret: pd.Series, rf_annual: float = 0.0) -> dict:
    r = monthly_ret.dropna()
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    months = len(r)
    years = months / 12.0
    cagr = eq.iloc[-1] ** (1/years) - 1 if years > 0 else np.nan
    ann_vol = r.std(ddof=0) * np.sqrt(12)

    rf_m = (1 + rf_annual) ** (1/12) - 1
    ex = r - rf_m
    ex_vol = ex.std(ddof=0) * np.sqrt(12)
    sharpe = (ex.mean() * 12) / ex_vol if ex_vol and ex_vol > 0 else np.nan

    peak = eq.cummax()
    max_dd = (eq / peak - 1.0).min()
    win_pct = (r > 0).mean() * 100
    return {
        "CAGR": cagr,
        "Annualized Volatility": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Total Months": months,
        "Winning Months (%)": win_pct,
    }

def lowest_avg_corr_combo(candidates, daily_px, asof_date, lookback_days, k):
    candidates = list(candidates)
    if len(candidates) <= k:
        return candidates

    window = daily_px[candidates].loc[:asof_date].tail(lookback_days + 5)
    rets = window.pct_change().dropna()

    if rets.shape[0] < max(5, int(lookback_days * 0.6)):
        return candidates[:k]

    best, best_score = None, np.inf
    for combo in combinations(candidates, k):
        sub = rets[list(combo)].dropna()
        if sub.shape[0] < 5 or sub.shape[1] < k:
            continue
        c = sub.corr()
        vals = [c.iloc[i, j] for i in range(k) for j in range(i + 1, k)]
        if not vals:
            continue
        score = float(np.mean(vals))
        if score < best_score:
            best_score, best = score, list(combo)

    return best if best is not None else candidates[:k]

def select_assets_at_date(momentum_series_at_date: pd.Series,
                          decision_date: pd.Timestamp,
                          daily_px: pd.DataFrame,
                          cash_proxy: str,
                          top_n: int,
                          corr_lookback_days: int,
                          k: int,
                          require_positive: bool = True) -> list[str]:
    m = momentum_series_at_date.dropna().copy()
    if cash_proxy in m.index:
        m = m.drop(cash_proxy)
    if require_positive:
        m = m[m > 0.0]
    if m.empty:
        return []
    top = m.nlargest(min(top_n, len(m))).index.tolist()
    top = [t for t in top if t in daily_px.columns]
    if not top:
        return []
    return lowest_avg_corr_combo(top, daily_px, asof_date=decision_date, lookback_days=corr_lookback_days, k=k)

def inv_vol_weights_for_slots(daily_px, slot_list, asof_date, lookback_days=20, cap=0.45, ridge=1e-12):
    slots = list(slot_list)
    uniq = list(dict.fromkeys(slots))

    px = daily_px[uniq].loc[:asof_date].tail(lookback_days + 1)
    rets = px.pct_change().dropna()

    if rets.shape[0] < max(5, int(lookback_days * 0.5)):
        return weights_equal_slots(slots)

    vol = rets.std(ddof=0).replace(0.0, np.nan)
    if vol.isna().any():
        return weights_equal_slots(slots)

    raw = np.array([1.0 / (vol[t] + ridge) for t in slots], dtype=float)
    raw = raw / raw.sum()

    w = {}
    for t, ws in zip(slots, raw):
        w[t] = w.get(t, 0.0) + float(ws)

    if cap is not None:
        for _ in range(10):
            over = {t: wt for t, wt in w.items() if wt > cap}
            if not over:
                break
            excess = sum(w[t] - cap for t in over)
            for t in over:
                w[t] = cap
            rest = {t: wt for t, wt in w.items() if t not in over}
            rest_sum = sum(rest.values())
            if rest_sum <= 0:
                return weights_equal_slots(slots)
            for t in rest:
                w[t] = w[t] + excess * (rest[t] / rest_sum)

        s = sum(w.values())
        w = {t: wt / s for t, wt in w.items()} if s > 0 else weights_equal_slots(slots)

    return w

@st.cache_data(show_spinner=False)
def download_prices(tickers, start_date, end_date):
    px = pd.DataFrame()
    for t in tickers:
        df = yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            px[t] = df["Close"]
    px = px.dropna(axis=1, how="all").ffill().dropna()
    return px

def build_weight_schedule(selections_dict, daily_px, decision_dates, mode, universe_cols, cash_proxy,
                          slots, vol_lb, vol_cap):
    sched = {}
    for d in decision_dates:
        pick = selections_dict.get(d, [])
        slot_list = pad_to_slots(pick, slots, cash_proxy)
        if mode == "equal":
            w = weights_dict_to_series(weights_equal_slots(slot_list), universe_cols)
        else:
            w = weights_dict_to_series(inv_vol_weights_for_slots(
                daily_px, slot_list, asof_date=d,
                lookback_days=vol_lb, cap=vol_cap
            ), universe_cols)
        sched[d] = w
    return sched

def strategy_ytd_true(daily_px: pd.DataFrame,
                      selections_dict: dict,
                      year: int,
                      end_date: pd.Timestamp,
                      mode: str,
                      universe_cols,
                      cash_proxy,
                      slots,
                      vol_lb,
                      vol_cap) -> float:
    """
    TRUE daily YTD for the strategy, using OFFICIAL month-end holdings schedule.
    Rebalance at month-end close; weights apply starting next trading day.
    """
    end_td = last_trading_day_on_or_before(daily_px.index, end_date)
    if end_td is None:
        return np.nan

    year_start = pd.Timestamp(year=year, month=1, day=1)
    start_td = first_trading_day_on_or_after(daily_px.index, year_start)
    if start_td is None or start_td > end_td:
        return np.nan

    prior_month_end = (year_start - pd.offsets.MonthEnd(1)).normalize()
    eligible = [d for d in selections_dict.keys() if d <= prior_month_end]
    anchor_dec = max(eligible) if eligible else min(selections_dict.keys())

    last_month_end_before_end = month_end_prices(daily_px.loc[:end_td]).index.max()
    decision_dates = sorted([d for d in selections_dict.keys() if (d >= anchor_dec and d <= last_month_end_before_end)])
    if anchor_dec not in decision_dates:
        decision_dates = sorted([anchor_dec] + decision_dates)

    w_sched = build_weight_schedule(selections_dict, daily_px, decision_dates, mode, universe_cols,
                                    cash_proxy, slots, vol_lb, vol_cap)

    daily_ret = daily_px.pct_change().loc[start_td:end_td].dropna(how="all")
    weights_by_day = pd.DataFrame(0.0, index=daily_ret.index, columns=universe_cols)

    for i, d in enumerate(decision_dates):
        start_hold = first_trading_day_on_or_after(daily_px.index, (d + pd.Timedelta(days=1)).normalize())
        if start_hold is None:
            continue

        if i < len(decision_dates) - 1:
            next_d = decision_dates[i + 1]
            end_hold = last_trading_day_on_or_before(daily_px.index, next_d)
        else:
            end_hold = end_td

        if end_hold is None:
            continue

        start_hold = max(start_hold, start_td)
        end_hold = min(end_hold, end_td)
        if start_hold > end_hold:
            continue

        w = w_sched[d]
        mask = (weights_by_day.index >= start_hold) & (weights_by_day.index <= end_hold)
        weights_by_day.loc[mask, :] = w.values

    port_daily = (weights_by_day * daily_ret.reindex(weights_by_day.index)).sum(axis=1).dropna()
    if port_daily.empty:
        return np.nan
    return float((1 + port_daily).prod() - 1)

# -----------------------------
# Main run
# -----------------------------
if RUN:
    end_dt = parse_date_or_today(END_DATE)
    today_real = pd.Timestamp.today().normalize()
    completed_month_end_real = (today_real - pd.offsets.MonthEnd(1)).normalize()
    official_decision_date_real = completed_month_end_real

    with st.spinner("Downloading daily prices (cached)..."):
        daily_prices = download_prices(TICKERS, START_DATE, end_dt.strftime("%Y-%m-%d"))

    if daily_prices.empty:
        st.error("No price data downloaded. Check tickers and start date.")
        st.stop()

    # clip to today (live-safe)
    daily_prices = daily_prices.loc[:today_real].copy()

    if CASH_PROXY not in daily_prices.columns:
        st.error(f"Cash proxy {CASH_PROXY} not present in downloaded prices.")
        st.stop()
    if "SPY" not in daily_prices.columns:
        st.error("SPY not present in downloaded prices (needed for benchmarks). Add SPY.")
        st.stop()

    peek_date = today_real
    if peek_date not in daily_prices.index:
        peek_date = daily_prices.index[daily_prices.index <= peek_date].max()

    st.success(f"Data: {daily_prices.shape[0]:,} days × {daily_prices.shape[1]} assets "
               f"({daily_prices.index.min().date()} → {daily_prices.index.max().date()}).")

    mp = month_end_prices(daily_prices).dropna(how="all")
    mret = mp.pct_change()
    mom = mp.pct_change(MOM_LOOKBACK)

    universe_cols = list(daily_prices.columns)

    # Build selections at month-ends
    decision_dates = list(mom.index)[MOM_LOOKBACK:]
    selections = {}
    for d in decision_dates:
        selections[d] = select_assets_at_date(
            momentum_series_at_date=mom.loc[d],
            decision_date=d,
            daily_px=daily_prices,
            cash_proxy=CASH_PROXY,
            top_n=TOP_CANDIDATES,
            corr_lookback_days=CORR_LB,
            k=PORTFOLIO_SLOTS,
            require_positive=REQUIRE_POSITIVE_MOMENTUM
        )

    # Monthly realized table
    rows = []
    for d, pick in selections.items():
        held_month_end = (d + pd.offsets.MonthEnd(1)).normalize()
        if held_month_end > completed_month_end_real:
            continue
        if held_month_end not in mret.index:
            continue

        slots_list = pad_to_slots(pick, PORTFOLIO_SLOTS, CASH_PROXY)
        w_eq = weights_dict_to_series(weights_equal_slots(slots_list), universe_cols)

        w_iv = None
        if SHOW_INVVOL:
            w_iv = weights_dict_to_series(inv_vol_weights_for_slots(
                daily_prices, slots_list, asof_date=d, lookback_days=VOL_LB, cap=VOL_CAP
            ), universe_cols)

        r_eq = float((w_eq * mret.loc[held_month_end]).sum())
        r_iv = float((w_iv * mret.loc[held_month_end]).sum()) if SHOW_INVVOL else np.nan

        rows.append({
            "Held Month End": held_month_end,
            "Decision Date": d,
            "Held Month": held_month_end.strftime("%Y-%m"),
            "Slots": ", ".join(slots_list),
            "Equal Slots Weights": fmt_holdings(w_eq),
            "Inv-Vol Weights": fmt_holdings(w_iv) if SHOW_INVVOL else "",
            "Return Equal Slots": r_eq,
            "Return Inv-Vol": r_iv
        })

    monthly_compare = pd.DataFrame(rows).set_index("Held Month End").sort_index()

    # Current official holdings row + MTD
    eligible_official = [d for d in selections.keys() if d <= official_decision_date_real]
    if not eligible_official:
        st.error("Not enough history to determine official holdings (need at least momentum lookback months).")
        st.stop()

    official_decision = max(eligible_official)
    current_month_end = (official_decision + pd.offsets.MonthEnd(1)).normalize()
    current_month_label = current_month_end.strftime("%Y-%m")

    slots_official = pad_to_slots(selections[official_decision], PORTFOLIO_SLOTS, CASH_PROXY)
    w_eq_official = weights_dict_to_series(weights_equal_slots(slots_official), universe_cols)
    w_iv_official = None
    if SHOW_INVVOL:
        w_iv_official = weights_dict_to_series(inv_vol_weights_for_slots(
            daily_prices, slots_official, asof_date=official_decision, lookback_days=VOL_LB, cap=VOL_CAP
        ), universe_cols)

    month_start = (official_decision + pd.Timedelta(days=1)).normalize()
    start_td = first_trading_day_on_or_after(daily_prices.index, month_start)

    mtd_eq, mtd_iv = np.nan, np.nan
    if start_td is not None and start_td <= peek_date:
        px_start = daily_prices.loc[start_td, universe_cols]
        px_now   = daily_prices.loc[peek_date, universe_cols]
        asset_mtd = (px_now / px_start) - 1.0
        mtd_eq = float((w_eq_official * asset_mtd).sum())
        if SHOW_INVVOL:
            mtd_iv = float((w_iv_official * asset_mtd).sum())

    official_row = pd.DataFrame([{
        "Decision Date": official_decision,
        "Held Month": f"{current_month_label} (CURRENT - official)",
        "Slots": ", ".join(slots_official),
        "Equal Slots Weights": fmt_holdings(w_eq_official),
        "Inv-Vol Weights": fmt_holdings(w_iv_official) if SHOW_INVVOL else "",
        "Return Equal Slots": mtd_eq,
        "Return Inv-Vol": mtd_iv if SHOW_INVVOL else np.nan
    }], index=[current_month_end])

    monthly_compare = pd.concat([monthly_compare, official_row], axis=0)
    monthly_compare = monthly_compare[~monthly_compare.index.duplicated(keep="last")].sort_index()

    # Mid-month peek (indicative rebalance today) + diff
    mom_anchor = official_decision if official_decision in mom.index else max([d for d in mom.index if d <= official_decision])

    indicative_pick = select_assets_at_date(
        momentum_series_at_date=mom.loc[mom_anchor],
        decision_date=peek_date,
        daily_px=daily_prices.loc[:peek_date],
        cash_proxy=CASH_PROXY,
        top_n=TOP_CANDIDATES,
        corr_lookback_days=CORR_LB,
        k=PORTFOLIO_SLOTS,
        require_positive=REQUIRE_POSITIVE_MOMENTUM
    )

    slots_peek = pad_to_slots(indicative_pick, PORTFOLIO_SLOTS, CASH_PROXY)
    w_eq_peek = weights_dict_to_series(weights_equal_slots(slots_peek), universe_cols)
    w_iv_peek = None
    if SHOW_INVVOL:
        w_iv_peek = weights_dict_to_series(inv_vol_weights_for_slots(
            daily_prices.loc[:peek_date], slots_peek, asof_date=peek_date, lookback_days=VOL_LB, cap=VOL_CAP
        ), universe_cols)

    sells, buys = sells_buys(slots_official, slots_peek, CASH_PROXY)
    to_eq = turnover_one_way(w_eq_official, w_eq_peek)
    to_iv = turnover_one_way(w_iv_official, w_iv_peek) if SHOW_INVVOL else np.nan

    st.subheader("Mid-month peek (indicative rebalance today)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Official decision date (month-end)", official_decision.strftime("%Y-%m-%d"))
    c2.metric("Peek date (last trading day ≤ today)", peek_date.strftime("%Y-%m-%d"))
    c3.metric("Momentum anchor used", mom_anchor.strftime("%Y-%m-%d"))

    st.write(f"**Current official slots:** `{', '.join(slots_official)}`")
    st.write(f"**Indicative slots today:** `{', '.join(slots_peek)}`")
    st.write(f"**Indicative SELL:** `{', '.join(sells) if sells else 'None'}`   |   **Indicative BUY:** `{', '.join(buys) if buys else 'None'}`")
    st.write(f"**Indicative turnover:** Equal `{to_eq:.2%}`" + (f"  |  Inv-Vol `{to_iv:.2%}`" if SHOW_INVVOL else ""))

    peek_row = pd.DataFrame([{
        "Decision Date": f"{mom_anchor.date()} (Momentum) / {peek_date.date()} (Peek)",
        "Held Month": "INDICATIVE (rebalance today)",
        "Slots": ", ".join(slots_peek),
        "Equal Slots Weights": fmt_holdings(w_eq_peek),
        "Inv-Vol Weights": fmt_holdings(w_iv_peek) if SHOW_INVVOL else "",
        "Return Equal Slots": np.nan,
        "Return Inv-Vol": np.nan
    }], index=[peek_date + pd.Timedelta(hours=12)])

    monthly_compare = pd.concat([monthly_compare, peek_row], axis=0).sort_index()

    # -----------------------------
    # Performance series (realized months only)
    # -----------------------------
    realized_mask = monthly_compare.index <= completed_month_end_real
    r_equal = monthly_compare.loc[realized_mask, "Return Equal Slots"].astype(float).dropna()
    r_invvol = monthly_compare.loc[realized_mask, "Return Inv-Vol"].astype(float).dropna() if SHOW_INVVOL else pd.Series(dtype=float)

    spy_m = mp["SPY"].pct_change()
    spy_m_realized = spy_m.loc[spy_m.index <= completed_month_end_real].dropna()

    summary_rows = []
    summary_rows.append({"Model": f"Equal Slots ({PORTFOLIO_SLOTS} slots; fill={CASH_PROXY})", **perf_stats(r_equal, RF_ANNUAL)})
    if SHOW_INVVOL:
        summary_rows.append({"Model": f"Inverse-Vol (lb={VOL_LB}d cap={VOL_CAP:.2f})", **perf_stats(r_invvol, RF_ANNUAL)})
    summary_rows.append({"Model": "SPY (Buy & Hold)", **perf_stats(spy_m_realized, RF_ANNUAL)})

    stats_df = pd.DataFrame(summary_rows)

    # -----------------------------
    # Yearly returns with TRUE YTD
    # -----------------------------
    def yearly_from_monthly(monthly_ret: pd.Series) -> pd.Series:
        r = monthly_ret.dropna()
        yr = r.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        yr.index = yr.index.year
        return yr

    yr_equal_full = yearly_from_monthly(r_equal)
    yr_inv_full = yearly_from_monthly(r_invvol) if SHOW_INVVOL else pd.Series(dtype=float)
    yr_spy_full = yearly_from_monthly(spy_m_realized)

    cur_year = int(today_real.year)

    ytd_equal_true = strategy_ytd_true(
        daily_prices, selections, cur_year, peek_date,
        mode="equal",
        universe_cols=universe_cols,
        cash_proxy=CASH_PROXY,
        slots=PORTFOLIO_SLOTS,
        vol_lb=VOL_LB,
        vol_cap=VOL_CAP
    )
    ytd_inv_true = np.nan
    if SHOW_INVVOL:
        ytd_inv_true = strategy_ytd_true(
            daily_prices, selections, cur_year, peek_date,
            mode="invvol",
            universe_cols=universe_cols,
            cash_proxy=CASH_PROXY,
            slots=PORTFOLIO_SLOTS,
            vol_lb=VOL_LB,
            vol_cap=VOL_CAP
        )

    # SPY TRUE YTD (standard): last trading day of prior year close -> today close
    spy_series = daily_prices["SPY"].dropna()
    prev_year_end = pd.Timestamp(year=cur_year - 1, month=12, day=31)
    spy_prev_close_day = last_trading_day_on_or_before(spy_series.index, prev_year_end)
    spy_end_td = last_trading_day_on_or_before(spy_series.index, peek_date)
    spy_ytd_true = np.nan
    if spy_prev_close_day is not None and spy_end_td is not None and spy_prev_close_day < spy_end_td:
        spy_ytd_true = float(spy_series.loc[spy_end_td] / spy_series.loc[spy_prev_close_day] - 1.0)

    yearly_df = pd.DataFrame({
        "Equal Slots": yr_equal_full,
        "SPY": yr_spy_full
    }).sort_index()

    if SHOW_INVVOL:
        yearly_df["Inverse-Vol"] = yr_inv_full

    yearly_df.loc[cur_year, "Equal Slots"] = ytd_equal_true
    if SHOW_INVVOL:
        yearly_df.loc[cur_year, "Inverse-Vol"] = ytd_inv_true
    yearly_df.loc[cur_year, "SPY"] = spy_ytd_true

    # -----------------------------
    # Layout: Tables + Downloads
    # -----------------------------
    st.subheader("Monthly allocations & returns")
    show_df = monthly_compare.copy()
    st.dataframe(
        show_df[[
            "Decision Date", "Held Month", "Slots",
            "Equal Slots Weights",
            "Inv-Vol Weights" if SHOW_INVVOL else "Equal Slots Weights",
            "Return Equal Slots",
            "Return Inv-Vol" if SHOW_INVVOL else "Return Equal Slots"
        ]],
        use_container_width=True
    )

    colA, colB, colC = st.columns(3)
    colA.download_button(
        "Download monthly table (CSV)",
        data=show_df.to_csv(index=True).encode("utf-8"),
        file_name="monthly_table.csv",
        mime="text/csv"
    )
    colB.download_button(
        "Download yearly returns (CSV)",
        data=yearly_df.to_csv(index=True).encode("utf-8"),
        file_name="yearly_returns.csv",
        mime="text/csv"
    )
    colC.download_button(
        "Download summary stats (CSV)",
        data=stats_df.to_csv(index=False).encode("utf-8"),
        file_name="summary_stats.csv",
        mime="text/csv"
    )

    st.subheader("Summary stats (realized months only)")
    st.dataframe(stats_df, use_container_width=True)

    st.subheader("Yearly returns (current year is TRUE YTD through today)")
    st.dataframe(yearly_df, use_container_width=True)

    # -----------------------------
    # Chart: equity curves (realized months only) + SPY
    # -----------------------------
    st.subheader("Equity curves (monthly, realized only)")
    eq_equal = (1 + r_equal).cumprod()

    fig = plt.figure(figsize=(12, 5))
    plt.plot(eq_equal.index, eq_equal.values, label="Equal Slots")

    if SHOW_INVVOL and not r_invvol.empty:
        eq_inv = (1 + r_invvol).cumprod()
        plt.plot(eq_inv.index, eq_inv.values, label="Inverse-Vol")

    # align SPY to same dates as eq_equal for a fair overlay
    spy_aligned = spy_m_realized.reindex(eq_equal.index).dropna()
    if not spy_aligned.empty:
        eq_spy = (1 + spy_aligned).cumprod()
        plt.plot(eq_spy.index, eq_spy.values, label="SPY (Buy & Hold)")

    plt.grid(True)
    plt.xlabel("Month End")
    plt.ylabel("Growth of $1")
    plt.legend()
    st.pyplot(fig)

    # -----------------------------
    # Footer sanity prints
    # -----------------------------
    st.caption(
        f"Peek date used: {peek_date.date()} | "
        f"SPY YTD uses prior-year close day: {(spy_prev_close_day.date() if spy_prev_close_day is not None else 'N/A')} → {spy_end_td.date() if spy_end_td is not None else 'N/A'}"
    )
else:
    st.info("Set parameters in the sidebar and click **Run backtest**.")
