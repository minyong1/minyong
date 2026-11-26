from __future__ import annotations

import platform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core_engine import (
    STOCK_UNIVERSE,
    calc_forecast_range,
    find_similar_chart_patterns,
    get_log_path,
    predict_today_signal,
    run_walk_forward_backtest,
    scan_and_recommend_stocks,
)

# OS별 한글 폰트 자동 설정
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="Meta-Labeling 주가 예측 & 추천 대시보드", layout="wide")
st.title("Triple-Barrier + Meta-Labeling 퀀트 대시보드")

# 사이드바 파라미터 제어
with st.sidebar:
    st.header("개별 종목 및 전략 설정")

    ticker_options = [f"{t} | {info['name']} ({info['group']})" for t, info in STOCK_UNIVERSE.items()]
    ticker_options.append("직접 입력")
    selected_label = st.selectbox("빠른 종목 선택 (총 20개)", ticker_options)

    if selected_label == "직접 입력":
        ticker = st.text_input("티커 입력 (예: PLTR, 005380.KS)", value="AAPL").strip().upper()
    else:
        ticker = selected_label.split(" | ")[0]

    start_date = st.text_input("백테스트 시작일", value="2018-01-01")

    st.divider()
    pt_mult = st.slider("익절 장벽 배수 (변동성 × N)", 1.0, 4.0, 2.0, 0.1)
    sl_mult = st.slider("손절 장벽 배수 (변동성 × N)", 0.5, 3.0, 1.5, 0.1)
    max_holding = st.slider("최대 보유 거래일", 3, 20, 10, 1)
    min_proba = st.slider("2차 진입 기준 확률", 0.45, 0.65, 0.52, 0.01)
    sizing_method = st.radio("비중 조절 방식 (Bet Sizing)", ["afml", "linear"], horizontal=True)

tab_scan, tab_single, tab_bt, tab_log, tab_scenario = st.tabs([
    "오늘의 추천 종목 (스캐너)",
    "개별 종목 신호",
    "Walk-Forward 백테스트",
    "자동 실행 로그",
    "차트 시나리오 도우미",
])

# 탭 1: 전체 종목 스캔 및 추천
with tab_scan:
    st.subheader("관심 종목 일괄 스캔 & 매수 추천 순위")
    st.caption("1차 상승 추세 필터와 2차 LightGBM Meta-Model 성공 확률을 기준으로 진입 후보를 자동 정렬합니다.")

    c_filter, c_retrain = st.columns([2, 1])
    groups = ["전체 (20개)", "미국 개별주", "미국 ETF", "한국 코스피", "한국 코스닥"]
    selected_group = c_filter.selectbox("스캔 대상 시장", groups)
    scan_force_retrain = c_retrain.checkbox("스캔 시 전체 모델 강제 재학습", value=False)

    if selected_group == "전체 (20개)":
        scan_tickers = list(STOCK_UNIVERSE.keys())
    else:
        scan_tickers = [t for t, info in STOCK_UNIVERSE.items() if info["group"] == selected_group]

    if st.button(f"선택 시장({len(scan_tickers)}개 종목) 스캔 및 추천 보기", type="primary"):
        progress_bar = st.progress(0.0, text="스캔 준비 중...")

        def _update_ui(idx: int, total: int, label: str):
            ratio = min(idx / max(total, 1), 1.0)
            progress_bar.progress(ratio, text=f"[{idx}/{total}] {label}")

        scan_df = scan_and_recommend_stocks(
            tickers=scan_tickers,
            min_proba=min_proba,
            sizing_method=sizing_method,
            force_retrain=scan_force_retrain,
            pt_mult=pt_mult,
            sl_mult=sl_mult,
            max_holding=max_holding,
            progress_callback=_update_ui,
        )
        progress_bar.empty()

        buy_df = scan_df[scan_df["추천비중(%)"] > 0]
        st.divider()
        if buy_df.empty:
            st.warning("현재 설정된 기준 확률과 추세 조건을 통과한 매수 추천 종목이 없습니다. (현금 관망 구간)")
        else:
            st.success(f"총 {len(buy_df)}개 종목에서 매수 신호가 포착되었습니다. (상위 3개 요약)")
            top_n = buy_df.head(3)
            cols = st.columns(len(top_n))
            for col, (_, row) in zip(cols, top_n.iterrows()):
                with col:
                    st.metric(
                        label=f"{row['종목명']} ({row['티커']}) - {row['추천등급']}",
                        value=f"비중 {row['추천비중(%)']}%",
                        delta=f"성공확률 {row['성공확률(%)']}% | 목표 +{row['기대수익(%)']}%",
                    )
                    st.caption(
                        f"종가: {row['최근종가']:,.2f} | 익절: {row['목표익절가']:,.2f} | 손절: {row['손절기준가']:,.2f}"
                    )

        st.subheader("전체 스캔 결과 (추천 비중 · 확률 순)")
        display_cols = [
            "티커", "종목명", "구분", "기준일", "추천등급",
            "최근종가", "성공확률(%)", "추천비중(%)",
            "목표익절가", "기대수익(%)", "손절기준가", "손절폭(%)",
        ]
        st.dataframe(scan_df[display_cols], use_container_width=True, hide_index=True)

# 탭 2: 개별 종목 상세 신호
with tab_single:
    col_b1, col_b2 = st.columns([1, 1])
    run_infer = col_b1.button("오늘 매수 신호 확인 (저장 모델 사용)", type="primary", use_container_width=True)
    run_retrain = col_b2.button("최신 데이터로 지금 재학습 후 확인", use_container_width=True)

    if run_infer or run_retrain:
        with st.spinner(f"{ticker} 분석 및 추론 중..."):
            try:
                sig = predict_today_signal(
                    ticker=ticker,
                    min_proba=min_proba,
                    sizing_method=sizing_method,
                    force_retrain=run_retrain,
                    pt_mult=pt_mult,
                    sl_mult=sl_mult,
                    max_holding=max_holding,
                )
                name = STOCK_UNIVERSE.get(sig["ticker"], {}).get("name", sig["ticker"])
                st.subheader(f"[{name} / {sig['ticker']}] 기준일: {sig['as_of_date']} | 판정: {sig['action']}")

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("최근 종가", f"{sig['close']:,.2f}")
                m2.metric("1차 추세", sig["primary_trend"])
                m3.metric("2차 성공확률", f"{sig['meta_proba']*100:.1f}%")
                m4.metric("추천 투자비중", f"{sig['bet_size']*100:.0f}%")
                m5.metric("최대 보유일", f"{sig['max_holding']}일")

                p1, p2, p3 = st.columns(3)
                p1.info(f"목표 익절가: **{sig['tp_price']:,.2f}**")
                p2.error(f"손절 기준가: **{sig['sl_price']:,.2f}**")
                p3.caption(f"모델 최종 학습 시각: {sig['model_trained_at']}")
            except Exception as e:
                st.error(f"오류 발생: {e}")

# 탭 3: Walk-Forward 백테스트
with tab_bt:
    step_days = st.selectbox(
        "백테스트 재학습 주기",
        [21, 5],
        format_func=lambda x: "약 1개월 (21거래일)" if x == 21 else "매주 (5거래일, 느림)",
    )
    if st.button("Walk-Forward 백테스트 실행", type="primary"):
        with st.spinner(f"{ticker} 전진 분석 백테스트 수행 중... (10~30초 소요)"):
            try:
                bt = run_walk_forward_backtest(
                    ticker=ticker,
                    start=start_date,
                    pt_mult=pt_mult,
                    sl_mult=sl_mult,
                    max_holding=max_holding,
                    min_proba=min_proba,
                    sizing_method=sizing_method,
                    step_size=step_days,
                )

                st.subheader("전략별 성과 비교")
                st.dataframe(bt["metrics_df"], use_container_width=True, hide_index=True)

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]})
                ax1.plot(bt["bh_cum"].index, bt["bh_cum"], label="Buy & Hold", linestyle="--", alpha=0.7)
                ax1.plot(bt["prim_cum"].index, bt["prim_cum"], label="1차 추세 단독", alpha=0.8)
                ax1.plot(bt["meta_cum"].index, bt["meta_cum"], label="Meta-Labeling + BetSize", linewidth=2)
                ax1.set_title(f"{ticker} Out-of-Sample 누적 수익률")
                ax1.legend()
                ax1.grid(alpha=0.3)

                ax2.fill_between(bt["meta_exp"].index, 0, bt["meta_exp"], alpha=0.4, color="tab:green")
                ax2.set_ylim(0, 1.05)
                ax2.set_ylabel("투입 비중")
                ax2.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                c1, c2 = st.columns([1, 2])
                with c1:
                    st.subheader("상위 10개 피처 중요도")
                    st.bar_chart(bt["feature_importance"].head(10))
                with c2:
                    st.subheader("최근 매매 기록")
                    if not bt["trades_df"].empty:
                        show_cols = ["진입일", "청산일", "청산사유", "보유일", "투입비중", "수익률"]
                        st.dataframe(bt["trades_df"][show_cols].tail(20), use_container_width=True, hide_index=True)
                    else:
                        st.info("조건을 만족한 거래 내역이 없습니다.")
            except Exception as e:
                st.error(f"백테스트 오류: {e}")

# 탭 4: 스케줄러 자동 실행 로그
with tab_log:
    log_path = get_log_path()
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        st.dataframe(log_df.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("아직 자동 실행 로그(signals_log.csv)가 없습니다. weekly_job.py를 먼저 실행해 보세요.")

# 탭 5: 차트 시나리오 도우미 (2가지 모드 버튼 전환)
with tab_scenario:
    st.subheader(f"차트 시나리오 도우미 ({ticker})")
    st.caption("증권사 차트 흐름을 참고할 때 쓸 수 있는 보조 도구입니다. 투자 권유가 아닌 참고용 통계 계산입니다.")

    if "scenario_mode" not in st.session_state:
        st.session_state.scenario_mode = "pattern"

    b_col1, b_col2, _ = st.columns([1.4, 1.4, 2.2])
    if b_col1.button(
        "① 과거 유사 구간 찾기",
        type="primary" if st.session_state.scenario_mode == "pattern" else "secondary",
        use_container_width=True,
    ):
        st.session_state.scenario_mode = "pattern"

    if b_col2.button(
        "② 예상 레인지 계산기",
        type="primary" if st.session_state.scenario_mode == "range" else "secondary",
        use_container_width=True,
    ):
        st.session_state.scenario_mode = "range"

    st.divider()

    # 모드 1: 과거 유사 구간 찾기
    if st.session_state.scenario_mode == "pattern":
        st.markdown("#### 과거 유사 차트 구간 찾기")
        st.caption("최근 N일 가격 모양과 비슷한 과거 구간을 찾아, 이후 M일 동안 어떻게 움직였는지 확인합니다.")

        p_c1, p_c2, p_c3 = st.columns(3)
        lookback_d = p_c1.slider("비교할 최근 기간 (거래일)", 5, 30, 15, 1)
        horizon_d = p_c2.slider("이후 확인할 기간 (거래일)", 3, 20, 10, 1)
        top_k_n = p_c3.slider("상위 유사 구간 수", 3, 10, 5, 1)

        if st.button("유사 구간 검색 실행", type="primary"):
            with st.spinner(f"{ticker} 과거 차트 패턴 비교 중..."):
                try:
                    res = find_similar_chart_patterns(
                        ticker=ticker,
                        start=start_date,
                        lookback_days=lookback_d,
                        horizon_days=horizon_d,
                        top_k=top_k_n,
                    )
                    st.session_state["last_pattern_res"] = res
                    st.session_state["last_vol_pct"] = res["daily_vol_pct"]
                    st.session_state["last_close_price"] = res["latest_close"]
                except Exception as e:
                    st.error(f"패턴 분석 오류: {e}")

        if "last_pattern_res" in st.session_state:
            res = st.session_state["last_pattern_res"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"유사 구간 평균 ({res['horizon_days']}일 후)", f"{res['avg_ret_pct']:+.2f}%")
            m2.metric("상승 비율 (양수)", f"{res['win_rate_pct']:.1f}%")
            m3.metric("최고 / 최저 수익률", f"{res['best_ret_pct']:+.2f}% / {res['worst_ret_pct']:+.2f}%")
            m4.metric("평균 시나리오 환산가", f"{res['expected_price']:,.2f}")

            st.caption(
                f"기준일: {res['as_of_date']} | 현재 종가: {res['latest_close']:,.2f} | "
                f"참고 범위: {res['low_band_price']:,.2f} ~ {res['high_band_price']:,.2f}"
            )

            fig, ax = plt.subplots(figsize=(10, 4.5))
            x_hist = np.arange(-res["lookback_days"] + 1, 1)
            x_fut = np.arange(0, res["horizon_days"] + 1)

            ax.plot(x_hist, res["cur_norm_pct"], color="black", linewidth=2.5, label=f"현재 ({res['ticker']})")
            for m in res["top_matches"]:
                ax.plot(
                    x_fut,
                    m["future_path_pct"],
                    linestyle="--",
                    alpha=0.55,
                    label=f"{m['end_date'].strftime('%Y-%m-%d')} 이후 ({m['future_ret_pct']:+.1f}%)",
                )
            ax.plot(
                x_fut,
                res["avg_future_path_pct"],
                color="tab:red",
                linewidth=2.2,
                label=f"상위 {len(res['top_matches'])}개 평균 경로",
            )
            ax.axvline(0, color="gray", linestyle=":", linewidth=1)
            ax.axhline(0, color="gray", linestyle=":", linewidth=1)
            ax.set_xlabel("상대 거래일 (0 = 기준일)")
            ax.set_ylabel("시작점 대비 누적 변화율 (%)")
            ax.set_title(f"{res['ticker']} 유사 과거 구간 및 이후 {res['horizon_days']}일 흐름")
            ax.legend(loc="best", fontsize=8)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(res["summary_df"], use_container_width=True, hide_index=True)

    # 모드 2: 예상 레인지 계산기
    else:
        st.markdown("#### 예상 레인지(가격 범위) 계산기")
        st.caption("증권사 차트에서 확인한 현재가와 변동성을 입력해 예상 상단·하단 가격 범위를 계산합니다.")

        default_price = float(st.session_state.get("last_close_price", 100.0))
        default_vol = float(st.session_state.get("last_vol_pct", 2.0))

        r_c1, r_c2, r_c3 = st.columns(3)
        manual_price = r_c1.number_input("현재가 (또는 증권사 차트 가격)", min_value=0.01, value=default_price, step=1.0)
        manual_vol_pct = r_c2.number_input("일별 변동성 (%)", min_value=0.0, max_value=30.0, value=default_vol, step=0.1)
        range_days = r_c3.slider("예상 기간 (거래일)", 1, 30, max_holding, 1)

        r_c4, r_c5 = st.columns(2)
        drift_day_pct = r_c4.number_input("일 평균 기대 방향성 (%)", min_value=-5.0, max_value=5.0, value=0.0, step=0.05)
        sigma_mult = r_c5.slider("레인지 폭 배수 (1σ ≈ 68%, 1.5σ ≈ 87%)", 0.5, 2.5, 1.0, 0.1)

        if st.button("예상 레인지 계산", type="primary"):
            try:
                f_res = calc_forecast_range(
                    current_price=manual_price,
                    daily_vol_pct=manual_vol_pct,
                    horizon_days=range_days,
                    drift_pct_per_day=drift_day_pct,
                    sigma_mult=sigma_mult,
                )

                k1, k2, k3 = st.columns(3)
                k1.success(f"예상 상단: **{f_res['final_upper']:,.2f}** ({f_res['final_up_pct']:+.2f}%)")
                k2.info(f"중간 기준선: **{f_res['final_mid']:,.2f}**")
                k3.error(f"예상 하단: **{f_res['final_lower']:,.2f}** ({f_res['final_down_pct']:+.2f}%)")

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(f_res["days"], f_res["mid_curve"], color="tab:blue", linewidth=2, label="중간 기준선")
                ax.fill_between(
                    f_res["days"],
                    f_res["lower_curve"],
                    f_res["upper_curve"],
                    color="tab:blue",
                    alpha=0.2,
                    label=f"예상 레인지 ({sigma_mult:.1f}σ)",
                )
                ax.set_xlabel("향후 거래일")
                ax.set_ylabel("예상 가격")
                ax.set_title(f"{range_days}거래일 예상 가격 레인지")
                ax.legend()
                ax.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.dataframe(f_res["range_df"], use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"레인지 계산 오류: {e}")