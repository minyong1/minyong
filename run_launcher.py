from __future__ import annotations

import os
import sys
from pathlib import Path


def resolve_bundled_file(filename: str) -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / filename
    return Path(__file__).resolve().parent / filename


def main():
    # 1) 작업 스케줄러 모드: StockQuantApp.exe --weekly [--retrain]
    if "--weekly" in sys.argv:
        from weekly_job import DEFAULT_WATCHLIST, run_job

        force_retrain = "--retrain" in sys.argv
        run_job(tickers=DEFAULT_WATCHLIST, force_retrain=force_retrain)
        return

    # 2) 일반 더블클릭 모드: 내장 Streamlit 웹 대시보드 자동 실행
    from streamlit.web import cli as stcli

    dashboard_path = resolve_bundled_file("dashboard_app.py")
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

    sys.argv = [
        "streamlit",
        "run",
        str(dashboard_path),
        "--global.developmentMode=false",
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()