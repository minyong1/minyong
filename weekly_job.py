from __future__ import annotations

import argparse
import pandas as pd
from core_engine import (
    STOCK_UNIVERSE,
    get_log_path,
    scan_and_recommend_stocks,
)

DEFAULT_WATCHLIST = list(STOCK_UNIVERSE.keys())


def run_job(tickers: list[str], force_retrain: bool = False, min_proba: float = 0.52) -> pd.DataFrame:
    def _cli_progress(idx: int, total: int, label: str):
        if idx < total:
            print(f"[{idx + 1}/{total}] {label} 분석 중... (강제 재학습={force_retrain})")

    out_df = scan_and_recommend_stocks(
        tickers=tickers,
        min_proba=min_proba,
        force_retrain=force_retrain,
        progress_callback=_cli_progress,
    )

    if out_df.empty:
        return out_df

    out_df.insert(0, "실행시각", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    log_path = get_log_path()

    if log_path.exists():
        prev_df = pd.read_csv(log_path)
        combined = pd.concat([prev_df, out_df], ignore_index=True)
    else:
        combined = out_df
    combined.to_csv(log_path, index=False, encoding="utf-8-sig")

    show_cols = ["티커", "종목명", "구분", "추천등급", "최근종가", "성공확률(%)", "추천비중(%)", "목표익절가", "손절기준가"]
    print("\n================ [전체 종목 스캔 결과] ================")
    print(out_df[show_cols].to_string(index=False))

    buy_df = out_df[out_df["추천비중(%)"] > 0]
    print("\n================ [오늘의 추천 매수 종목] ================")
    if buy_df.empty:
        print("현재 진입 조건을 만족하는 추천 종목이 없습니다. (현금 관망)")
    else:
        print(buy_df[show_cols].head(5).to_string(index=False))

    print(f"\n로그 저장 완료: {log_path}")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="저장된 모델을 무시하고 강제 재학습")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_WATCHLIST, help="분석할 티커 목록")
    args = parser.parse_args()

    run_job(tickers=args.tickers, force_retrain=args.retrain)