from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")


# .exe(PyInstaller) 실행 시에도 모델/로그가 exe 파일 옆 폴더에 영구 저장되도록 경로 분기
def get_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_model_dir() -> Path:
    path = get_app_dir() / "saved_models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_log_path() -> Path:
    return get_app_dir() / "signals_log.csv"


def sanitize_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "_").replace("^", "")


# 1. 단일 티커 다운로드
def download_ohlcv(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"데이터를 불러오지 못했습니다: {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()


# 2. 거시 지표 수집 및 한·미 시차 안전 정렬
def load_macro_features(
    target_index: pd.DatetimeIndex,
    is_kr_stock: bool,
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    macro_specs = [
        ("sp500", "^GSPC", 1 if is_kr_stock else 0),
        ("vix", "^VIX", 1 if is_kr_stock else 0),
        ("us10y", "^TNX", 1 if is_kr_stock else 0),
    ]
    if is_kr_stock:
        macro_specs.extend([
            ("kospi", "^KS11", 0),
            ("usdkrw", "KRW=X", 1),
        ])

    macro_df = pd.DataFrame(index=target_index)
    for prefix, ticker, shift_days in macro_specs:
        raw = download_ohlcv(ticker, start=start, end=end)["Close"]
        tmp = pd.DataFrame(index=raw.index)

        if prefix == "vix":
            tmp[f"{prefix}_level"] = raw
            tmp[f"{prefix}_chg_1d"] = raw.pct_change(1)
            tmp[f"{prefix}_ma20_ratio"] = raw / raw.rolling(20).mean() - 1.0
        elif prefix == "us10y":
            tmp[f"{prefix}_diff_1d"] = raw.diff(1)
            tmp[f"{prefix}_diff_5d"] = raw.diff(5)
        else:
            tmp[f"{prefix}_ret_1d"] = raw.pct_change(1)
            tmp[f"{prefix}_ret_5d"] = raw.pct_change(5)
            tmp[f"{prefix}_ma20_ratio"] = raw / raw.rolling(20).mean() - 1.0

        if shift_days > 0:
            tmp.index = tmp.index + pd.Timedelta(days=shift_days)

        macro_df = pd.concat([macro_df, tmp.reindex(target_index, method="ffill")], axis=1)

    return macro_df


# 3. 피처 생성 (오늘 마지막 행까지 유지하여 실시간 추론 지원)
def build_features_df(
    ticker: str,
    start: str = "2018-01-01",
    end: str | None = None,
    vol_window: int = 20,
    min_vol: float = 0.008,
) -> tuple[pd.DataFrame, list[str]]:
    is_kr = ticker.upper().endswith((".KS", ".KQ"))
    data = download_ohlcv(ticker, start=start, end=end)

    # 1차 추세 신호
    ema20 = data["Close"].ewm(span=20, adjust=False).mean()
    ema60 = data["Close"].ewm(span=60, adjust=False).mean()
    data["primary_signal"] = ((ema20 > ema60) & (data["Close"] > ema20)).astype(int)

    # 동적 장벽용 일별 변동성
    data["daily_vol"] = (
        data["Close"].pct_change().rolling(vol_window).std().clip(lower=min_vol)
    )

    # 기술적 피처
    data["ema_spread_20_60"] = ema20 / ema60 - 1.0
    for lag in [1, 3, 5, 10, 20]:
        data[f"ret_{lag}d"] = data["Close"].pct_change(lag)
    for window in [5, 20, 60]:
        ma = data["Close"].rolling(window).mean()
        data[f"ma_ratio_{window}"] = data["Close"] / ma - 1.0

    data["volatility_10"] = data["ret_1d"].rolling(10).std()
    data["volatility_20"] = data["ret_1d"].rolling(20).std()
    data["vol_regime"] = data["volatility_10"] / (data["volatility_20"] + 1e-9)
    data["hl_range"] = (data["High"] - data["Low"]) / data["Close"]
    data["oc_return"] = (data["Close"] - data["Open"]) / data["Open"]
    data["vol_ratio_20"] = data["Volume"] / (data["Volume"].rolling(20).mean() + 1e-9)

    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    data["rsi_14"] = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))

    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    data["macd_hist_norm"] = (macd - macd.ewm(span=9, adjust=False).mean()) / data["Close"]

    # 거시 지표 결합
    macro_df = load_macro_features(data.index, is_kr_stock=is_kr, start=start, end=end)
    data = pd.concat([data, macro_df], axis=1)

    bench_prefix = "kospi" if is_kr else "sp500"
    data["rel_strength_1d"] = data["ret_1d"] - data[f"{bench_prefix}_ret_1d"]
    data["rel_strength_5d"] = data["ret_5d"] - data[f"{bench_prefix}_ret_5d"]

    exclude_cols = {"Open", "High", "Low", "Close", "Volume", "primary_signal", "daily_vol"}
    feature_cols = [c for c in data.columns if c not in exclude_cols]

    data = data.dropna(subset=feature_cols + ["Close", "primary_signal", "daily_vol"]).copy()
    return data, feature_cols


# 4. Triple-Barrier 라벨 부착 (학습 및 백테스트용)
def attach_triple_barrier(
    feat_df: pd.DataFrame,
    pt_mult: float = 2.0,
    sl_mult: float = 1.5,
    max_holding: int = 10,
    fee_rate: float = 0.001,
) -> pd.DataFrame:
    df = feat_df.copy()
    n = len(df)
    opens = df["Open"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()
    vols = df["daily_vol"].to_numpy()
    dates = df.index.to_numpy()

    meta_labels = np.full(n, np.nan)
    exit_dates = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    pt_prices = closes * (1.0 + pt_mult * vols)
    sl_prices = closes * (1.0 - sl_mult * vols)

    min_ret = 2.0 * fee_rate
    for i in range(n - max_holding):
        entry = closes[i]
        upper = pt_prices[i]
        lower = sl_prices[i]

        label = None
        exit_idx = i + max_holding

        for j in range(i + 1, i + max_holding + 1):
            if opens[j] <= lower:
                label, exit_idx = 0, j
                break
            if opens[j] >= upper:
                label, exit_idx = 1, j
                break

            hit_sl = lows[j] <= lower
            hit_tp = highs[j] >= upper
            if hit_sl:
                label, exit_idx = 0, j
                break
            elif hit_tp:
                label, exit_idx = 1, j
                break

        if label is None:
            label = 1 if (closes[exit_idx] / entry - 1.0) > min_ret else 0

        meta_labels[i] = label
        exit_dates[i] = dates[exit_idx]

    df["meta_label"] = meta_labels
    df["tb_exit_date"] = pd.to_datetime(exit_dates)
    df["tb_pt_price"] = pt_prices
    df["tb_sl_price"] = sl_prices

    labeled_df = df.dropna(subset=["meta_label"]).copy()
    labeled_df["meta_label"] = labeled_df["meta_label"].astype(int)
    return labeled_df


# 5. Bet Sizing (확률 -> 투자 비중)
def calc_bet_size(
    proba: float,
    min_proba: float = 0.52,
    method: str = "afml",
    step: float = 0.1,
    max_position: float = 1.0,
    afml_scale: float = 1.5,
) -> float:
    if proba <= min_proba:
        return 0.0

    p = float(np.clip(proba, 1e-6, 1.0 - 1e-6))
    if method == "afml":
        z = (p - 0.5) / np.sqrt(p * (1.0 - p))
        raw_size = (2.0 * norm.cdf(z) - 1.0) * afml_scale
    else:
        raw_size = (p - min_proba) / max(0.75 - min_proba, 1e-6)

    raw_size = float(np.clip(raw_size, 0.0, max_position))
    return float(np.clip(round(raw_size / step) * step, 0.0, max_position))


# 6. 실전용 최신 모델 학습 및 디스크 저장 (매주 재학습용)
def train_and_save_latest_model(
    ticker: str,
    start: str = "2018-01-01",
    pt_mult: float = 2.0,
    sl_mult: float = 1.5,
    max_holding: int = 10,
    train_window: int = 504,
) -> dict:
    is_kr = ticker.upper().endswith((".KS", ".KQ"))
    fee_rate = 0.0015 if is_kr else 0.0010

    feat_df, feature_cols = build_features_df(ticker, start=start)
    labeled_df = attach_triple_barrier(
        feat_df, pt_mult=pt_mult, sl_mult=sl_mult, max_holding=max_holding, fee_rate=fee_rate
    )

    # 최근 구간에서 1차 추세 신호 발생일만 추출 후 시계열 85:15 분할 + Purging
    recent = labeled_df.iloc[-min(len(labeled_df), train_window + 120) :]
    active = recent[recent["primary_signal"] == 1]

    if len(active) < 40:
        active = labeled_df[labeled_df["primary_signal"] == 1]
    if len(active) < 30 or active["meta_label"].nunique() < 2:
        raise ValueError(f"[{ticker}] 학습 가능한 1차 추세 신호 샘플이 부족합니다.")

    split_idx = int(len(active) * 0.85)
    val = active.iloc[split_idx:]
    val_start_date = val.index[0]
    train = active.iloc[:split_idx]
    train = train[train["tb_exit_date"] < val_start_date]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.02,
        num_leaves=15,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        verbose=-1,
    )

    if len(val) >= 10 and val["meta_label"].nunique() == 2:
        model.fit(
            train[feature_cols],
            train["meta_label"],
            eval_set=[(val[feature_cols], val["meta_label"])],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )
    else:
        model.fit(train[feature_cols], train["meta_label"])

    artifact = {
        "ticker": ticker.upper(),
        "model": model,
        "feature_cols": feature_cols,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_label_date": str(labeled_df.index[-1].date()),
        "train_samples": len(train),
        "params": {
            "pt_mult": pt_mult,
            "sl_mult": sl_mult,
            "max_holding": max_holding,
            "fee_rate": fee_rate,
        },
    }

    model_path = get_model_dir() / f"{sanitize_ticker(ticker)}_meta_lgbm.joblib"
    joblib.dump(artifact, model_path)
    return artifact


# 7. 저장된 모델로 '오늘(최근 마감일)' 매수 신호 추론
def predict_today_signal(
    ticker: str,
    min_proba: float = 0.52,
    sizing_method: str = "afml",
    force_retrain: bool = False,
    retrain_days: int = 7,
    pt_mult: float = 2.0,
    sl_mult: float = 1.5,
    max_holding: int = 10,
) -> dict:
    model_path = get_model_dir() / f"{sanitize_ticker(ticker)}_meta_lgbm.joblib"

    need_retrain = force_retrain or (not model_path.exists())
    artifact = None

    if not need_retrain:
        artifact = joblib.load(model_path)
        trained_dt = datetime.strptime(artifact["trained_at"], "%Y-%m-%d %H:%M:%S")
        # 저장된 모델이 7일 이상 지났거나 장벽 파라미터가 다르면 자동 재학습
        p = artifact["params"]
        if (
            (datetime.now() - trained_dt).days >= retrain_days
            or p["pt_mult"] != pt_mult
            or p["sl_mult"] != sl_mult
            or p["max_holding"] != max_holding
        ):
            need_retrain = True

    if need_retrain:
        artifact = train_and_save_latest_model(
            ticker=ticker, pt_mult=pt_mult, sl_mult=sl_mult, max_holding=max_holding
        )

    # 최근 1년치 데이터만 빠르게 받아 오늘 피처 계산
    recent_start = (datetime.now() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    feat_df, _ = build_features_df(ticker, start=recent_start)
    latest = feat_df.iloc[-1]
    latest_date = str(feat_df.index[-1].date())

    close_price = float(latest["Close"])
    vol = float(latest["daily_vol"])
    primary = int(latest["primary_signal"])

    tp_price = close_price * (1.0 + pt_mult * vol)
    sl_price = close_price * (1.0 - sl_mult * vol)

    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    X_latest = feat_df.iloc[[-1]][feature_cols]

    if primary == 1:
        meta_proba = float(model.predict_proba(X_latest)[0, 1])
        bet_size = calc_bet_size(meta_proba, min_proba=min_proba, method=sizing_method)
    else:
        meta_proba = 0.0
        bet_size = 0.0

    if bet_size > 0.0:
        action = "BUY"
    elif primary == 1:
        action = "HOLD (2차 필터 미달)"
    else:
        action = "HOLD (하락/횡보 추세)"

    return {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker.upper(),
        "as_of_date": latest_date,
        "close": round(close_price, 2),
        "primary_trend": "UP" if primary == 1 else "OFF",
        "meta_proba": round(meta_proba, 4),
        "bet_size": round(bet_size, 2),
        "action": action,
        "tp_price": round(tp_price, 2),
        "sl_price": round(sl_price, 2),
        "max_holding": max_holding,
        "model_trained_at": artifact["trained_at"],
    }


# 8. Walk-Forward 백테스트 및 시뮬레이션 (대시보드 차트용)
def run_walk_forward_backtest(
    ticker: str,
    start: str = "2018-01-01",
    pt_mult: float = 2.0,
    sl_mult: float = 1.5,
    max_holding: int = 10,
    min_proba: float = 0.52,
    sizing_method: str = "afml",
    step_size: int = 21,
) -> dict:
    is_kr = ticker.upper().endswith((".KS", ".KQ"))
    fee_rate = 0.0015 if is_kr else 0.0010

    feat_df, feature_cols = build_features_df(ticker, start=start)
    data = attach_triple_barrier(
        feat_df, pt_mult=pt_mult, sl_mult=sl_mult, max_holding=max_holding, fee_rate=fee_rate
    )

    train_window, val_window = 504, 84
    cursor = train_window + val_window
    n = len(data)
    if cursor >= n:
        raise ValueError("백테스트할 기간이 너무 짧습니다. 시작일을 더 과거로 설정해 주세요.")

    oos_chunks, importances_list = [], []
    while cursor < n:
        test_end = min(cursor + step_size, n)
        test_chunk = data.iloc[cursor:test_end].copy()
        test_chunk["meta_proba"] = 0.0
        test_start_date = test_chunk.index[0]

        val_raw = data.iloc[cursor - val_window : cursor]
        val = val_raw[(val_raw["tb_exit_date"] < test_start_date) & (val_raw["primary_signal"] == 1)]

        val_start_date = val.index[0] if not val.empty else data.index[cursor - val_window]
        train_end_pos = cursor - val_window
        train_start_pos = max(0, train_end_pos - train_window)
        train_raw = data.iloc[train_start_pos:train_end_pos]
        train = train_raw[(train_raw["tb_exit_date"] < val_start_date) & (train_raw["primary_signal"] == 1)]

        if (
            len(train) < 30
            or len(val) < 10
            or train["meta_label"].nunique() < 2
            or val["meta_label"].nunique() < 2
        ):
            oos_chunks.append(test_chunk)
            cursor += step_size
            continue

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.02,
            num_leaves=15,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            train[feature_cols],
            train["meta_label"],
            eval_set=[(val[feature_cols], val["meta_label"])],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )

        active_mask = test_chunk["primary_signal"] == 1
        if active_mask.any():
            test_chunk.loc[active_mask, "meta_proba"] = model.predict_proba(
                test_chunk.loc[active_mask, feature_cols]
            )[:, 1]

        oos_chunks.append(test_chunk)
        importances_list.append(model.feature_importances_)
        cursor += step_size

    oos_df = pd.concat(oos_chunks)
    mean_imp = (
        pd.Series(np.mean(importances_list, axis=0), index=feature_cols).sort_values(ascending=False)
        if importances_list
        else pd.Series(0.0, index=feature_cols)
    )

    bh_cum = oos_df["Close"] / oos_df["Close"].iloc[0]
    prim_cum, prim_exp, prim_trades = _simulate_positions(
        oos_df, max_holding, fee_rate, use_meta=False
    )
    meta_cum, meta_exp, meta_trades = _simulate_positions(
        oos_df, max_holding, fee_rate, use_meta=True, min_proba=min_proba, sizing_method=sizing_method
    )

    metrics_df = pd.DataFrame([
        {"전략": "Buy & Hold", **_calc_metrics(bh_cum)},
        {"전략": "1차 추세 단독 (100%)", **_calc_metrics(prim_cum, prim_exp, prim_trades)},
        {"전략": "Meta-Labeling + BetSize", **_calc_metrics(meta_cum, meta_exp, meta_trades)},
    ])

    return {
        "bh_cum": bh_cum,
        "prim_cum": prim_cum,
        "meta_cum": meta_cum,
        "meta_exp": meta_exp,
        "trades_df": meta_trades,
        "metrics_df": metrics_df,
        "feature_importance": mean_imp,
    }


def _simulate_positions(
    oos_df: pd.DataFrame,
    max_holding: int,
    fee_rate: float,
    use_meta: bool,
    min_proba: float = 0.52,
    sizing_method: str = "afml",
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    cash, shares, in_pos = 1.0, 0.0, False
    entry_price = tp_price = sl_price = current_bet = 0.0
    hold_days, entry_date = 0, None
    equity_curve, exposure_curve, trades = [], [], []

    dates = oos_df.index
    opens = oos_df["Open"].to_numpy()
    highs = oos_df["High"].to_numpy()
    lows = oos_df["Low"].to_numpy()
    closes = oos_df["Close"].to_numpy()
    pt_arr = oos_df["tb_pt_price"].to_numpy()
    sl_arr = oos_df["tb_sl_price"].to_numpy()
    primary = oos_df["primary_signal"].to_numpy()
    meta_proba = oos_df["meta_proba"].to_numpy()

    for i in range(len(oos_df)):
        exited_today = False
        if in_pos:
            hold_days += 1
            exit_price, reason = None, None
            if opens[i] <= sl_price:
                exit_price, reason = opens[i], "SL_GAP"
            elif opens[i] >= tp_price:
                exit_price, reason = opens[i], "TP_GAP"
            elif lows[i] <= sl_price:
                exit_price, reason = sl_price, "SL"
            elif highs[i] >= tp_price:
                exit_price, reason = tp_price, "TP"
            elif hold_days >= max_holding:
                exit_price, reason = closes[i], "TIME"

            if exit_price is not None:
                cash += shares * exit_price * (1.0 - fee_rate)
                net_ret = (exit_price / entry_price) * ((1.0 - fee_rate) ** 2) - 1.0
                trades.append({
                    "진입일": str(entry_date.date()),
                    "청산일": str(dates[i].date()),
                    "청산사유": reason,
                    "보유일": hold_days,
                    "투입비중": f"{current_bet:.0%}",
                    "수익률": f"{net_ret:.2%}",
                    "_ret_num": net_ret,
                    "_bet_num": current_bet,
                })
                shares, in_pos, current_bet, exited_today = 0.0, False, 0.0, True

        if (not in_pos) and (not exited_today) and (i < len(oos_df) - 1) and primary[i] == 1:
            bet_size = (
                calc_bet_size(meta_proba[i], min_proba=min_proba, method=sizing_method)
                if use_meta
                else 1.0
            )
            if bet_size > 0.0:
                allocated = cash * bet_size
                cash -= allocated
                entry_price = closes[i]
                shares = (allocated * (1.0 - fee_rate)) / entry_price
                tp_price, sl_price = pt_arr[i], sl_arr[i]
                hold_days, entry_date, current_bet, in_pos = 0, dates[i], bet_size, True

        total_equity = cash + shares * closes[i]
        equity_curve.append(total_equity)
        exposure_curve.append((shares * closes[i]) / max(total_equity, 1e-9))

    return (
        pd.Series(equity_curve, index=dates),
        pd.Series(exposure_curve, index=dates),
        pd.DataFrame(trades),
    )


def _calc_metrics(
    cum_curve: pd.Series,
    exposure: pd.Series | None = None,
    trades_df: pd.DataFrame | None = None,
) -> dict:
    daily_ret = cum_curve.pct_change().fillna(cum_curve.iloc[0] - 1.0)
    total_return = cum_curve.iloc[-1] - 1.0
    years = max(len(cum_curve) / 252, 1e-9)
    cagr = cum_curve.iloc[-1] ** (1.0 / years) - 1.0
    sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-9)) * np.sqrt(252)
    mdd = ((cum_curve / cum_curve.cummax()) - 1.0).min()

    res = {
        "누적수익률": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "MDD": f"{mdd:.2%}",
        "평균비중": f"{exposure.mean():.1%}" if exposure is not None else "100.0%",
    }
    if trades_df is not None and not trades_df.empty:
        res["매매횟수"] = len(trades_df)
        res["승률"] = f"{(trades_df['_ret_num'] > 0).mean():.1%}"
    else:
        res["매매횟수"] = 0
        res["승률"] = "-"
    return res

# 9. 확장된 관심 종목 유니버스 (총 20개)
STOCK_UNIVERSE: dict[str, dict[str, str]] = {
    # 미국 개별주
    "AAPL": {"name": "애플", "group": "미국 개별주"},
    "NVDA": {"name": "엔비디아", "group": "미국 개별주"},
    "MSFT": {"name": "마이크로소프트", "group": "미국 개별주"},
    "TSLA": {"name": "테슬라", "group": "미국 개별주"},
    "AMZN": {"name": "아마존", "group": "미국 개별주"},
    "GOOGL": {"name": "알파벳(구글)", "group": "미국 개별주"},
    "META": {"name": "메타", "group": "미국 개별주"},
    # 미국 ETF
    "SPY": {"name": "S&P500 ETF", "group": "미국 ETF"},
    "QQQ": {"name": "나스닥100 ETF", "group": "미국 ETF"},
    "SOXX": {"name": "미국 반도체 ETF", "group": "미국 ETF"},
    "TLT": {"name": "미국 장기국채 ETF", "group": "미국 ETF"},
    # 한국 코스피
    "005930.KS": {"name": "삼성전자", "group": "한국 코스피"},
    "000660.KS": {"name": "SK하이닉스", "group": "한국 코스피"},
    "035420.KS": {"name": "NAVER", "group": "한국 코스피"},
    "005380.KS": {"name": "현대차", "group": "한국 코스피"},
    "373220.KS": {"name": "LG에너지솔루션", "group": "한국 코스피"},
    "069500.KS": {"name": "KODEX 200", "group": "한국 코스피"},
    # 한국 코스닥
    "247540.KQ": {"name": "에코프로비엠", "group": "한국 코스닥"},
    "196170.KQ": {"name": "알테오젠", "group": "한국 코스닥"},
    "086520.KQ": {"name": "에코프로", "group": "한국 코스닥"},
}


# 10. 다중 종목 일괄 스캔 및 추천 순위 산출
def scan_and_recommend_stocks(
    tickers: list[str],
    min_proba: float = 0.52,
    sizing_method: str = "afml",
    force_retrain: bool = False,
    pt_mult: float = 2.0,
    sl_mult: float = 1.5,
    max_holding: int = 10,
    progress_callback=None,
) -> pd.DataFrame:
    rows = []
    total = len(tickers)

    for idx, ticker in enumerate(tickers):
        info = STOCK_UNIVERSE.get(ticker.upper(), {"name": ticker.upper(), "group": "기타"})
        if progress_callback is not None:
            progress_callback(idx, total, f"{info['name']} ({ticker})")

        try:
            sig = predict_today_signal(
                ticker=ticker,
                min_proba=min_proba,
                sizing_method=sizing_method,
                force_retrain=force_retrain,
                pt_mult=pt_mult,
                sl_mult=sl_mult,
                max_holding=max_holding,
            )
            tp_pct = (sig["tp_price"] / sig["close"] - 1.0) * 100.0
            sl_pct = (sig["sl_price"] / sig["close"] - 1.0) * 100.0

            if sig["bet_size"] >= 0.5:
                grade = "강력 추천"
            elif sig["bet_size"] > 0.0:
                grade = "매수 추천"
            elif sig["primary_trend"] == "UP":
                grade = "관망 (확률 미달)"
            else:
                grade = "관망 (추세 이탈)"

            rows.append({
                "티커": sig["ticker"],
                "종목명": info["name"],
                "구분": info["group"],
                "기준일": sig["as_of_date"],
                "추천등급": grade,
                "최근종가": sig["close"],
                "1차추세": sig["primary_trend"],
                "성공확률(%)": round(sig["meta_proba"] * 100.0, 1),
                "추천비중(%)": int(round(sig["bet_size"] * 100.0)),
                "목표익절가": sig["tp_price"],
                "기대수익(%)": round(tp_pct, 2),
                "손절기준가": sig["sl_price"],
                "손절폭(%)": round(sl_pct, 2),
                "판정": sig["action"],
                "학습시각": sig["model_trained_at"],
            })
        except Exception as e:
            rows.append({
                "티커": ticker.upper(),
                "종목명": info["name"],
                "구분": info["group"],
                "기준일": "-",
                "추천등급": "오류",
                "최근종가": 0.0,
                "1차추세": "-",
                "성공확률(%)": 0.0,
                "추천비중(%)": 0,
                "목표익절가": 0.0,
                "기대수익(%)": 0.0,
                "손절기준가": 0.0,
                "손절폭(%)": 0.0,
                "판정": f"오류: {e}",
                "학습시각": "-",
            })

    if progress_callback is not None:
        progress_callback(total, total, "스캔 완료")

    df = pd.DataFrame(rows)
    if not df.empty:
        # 추천 비중 -> 성공 확률 순으로 내림차순 정렬
        df = df.sort_values(
            by=["추천비중(%)", "성공확률(%)"],
            ascending=[False, False],
        ).reset_index(drop=True)
    return df