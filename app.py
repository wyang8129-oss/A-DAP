
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.naive_bayes import GaussianNB
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib
import platform

# --- 한글 폰트 설정 ---
st.markdown("""
<style>
/* 구글 Noto Sans KR 폰트 로드 */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Noto Sans KR', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.title("스마트팜 수확량 + 생육 예측 XAI 통합 대시보드")

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

# -------------------------------------------------------------
# 작물선택
# -------------------------------------------------------------
crop_name = st.selectbox("작물 선택", ["토마토", "오이"])

# -------------------------------------------------------------
# 파일 업로드
# -------------------------------------------------------------
sensor_file = st.file_uploader("환경센서 데이터 업로드 (CSV)", type=["csv"])
yield_file = st.file_uploader("수확/생육 데이터 업로드 (CSV)", type=["csv"])

if sensor_file and yield_file:
    sensor_df = pd.read_csv(sensor_file)
    yield_df = pd.read_csv(yield_file)

    st.subheader("환경센서 데이터")
    st.dataframe(sensor_df.head())

    st.subheader("수확/생육 데이터")
    st.dataframe(yield_df.head())

    # -------------------------------------------------------------
    # 환경 센서 컬럼 선택 (가로 5개)
    # -------------------------------------------------------------
    st.subheader("컬럼 선택")
    st.markdown("**환경 센서 데이터 컬럼 선택**")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        date_col_sensor = st.selectbox("날짜시간", sensor_df.columns)
    with col2:
        temp_col = st.selectbox("온도", sensor_df.columns)
    with col3:
        hum_col = st.selectbox("습도", sensor_df.columns)
    with col4:
        co2_col = st.selectbox("CO₂", sensor_df.columns)
    with col5:
        solar_col = st.selectbox("일사량", sensor_df.columns)

    st.markdown("---")

    # -------------------------------------------------------------
    # 수확량 컬럼 선택 (가로 3개)
    # -------------------------------------------------------------
    st.markdown("**수확량 데이터 컬럼 선택**")

    col6, col7, col8 = st.columns(3)

    with col6:
        date_col_yield = st.selectbox("조사일자", yield_df.columns)
    with col7:
        harvest_count_col = st.selectbox("수확수", yield_df.columns)
    with col8:
        harvest_weight_col = st.selectbox("평균과중", yield_df.columns)

    st.markdown("---")

    # -------------------------------------------------------------
    # 작물별 생육 컬럼 선택
    # -------------------------------------------------------------
    st.markdown("**추가 생육 컬럼 선택**")

    if crop_name == "토마토":
        growth_features = ["초장", "생장길이", "엽수", "엽장", "엽폭", "줄기굵기", "화방높이"]
    else:   # 오이
        growth_features = ["초장", "엽수", "엽장", "엽폭", "줄기굵기", "화방높이"]

    growth_cols = {}

    # 3개씩 끊어서 UI 가로 배치
    for i in range(0, len(growth_features), 3):
        cols = st.columns(3)
        for j, gf in enumerate(growth_features[i:i + 3]):
            with cols[j]:
                # 사용자가 yield_df의 컬럼을 매핑하도록 함(없으면 None)
                options = [None] + yield_df.columns.tolist()
                if gf in yield_df.columns:
                    default_idx = yield_df.columns.get_loc(gf) + 1
                else:
                    default_idx = 0
                growth_cols[gf] = st.selectbox(f"{gf}", options, index=default_idx)

        # -------------------------------------------------------------
        # 날짜 변환 (안전하게)
        # -------------------------------------------------------------
        sensor_df[date_col_sensor] = pd.to_datetime(sensor_df[date_col_sensor], errors='coerce')
        yield_df[date_col_yield] = pd.to_datetime(yield_df[date_col_yield], errors='coerce')

        # 변환 실패(NaT) 제거
        sensor_df = sensor_df.dropna(subset=[date_col_sensor])
        yield_df = yield_df.dropna(subset=[date_col_yield])

        # date, hour, time 컬럼 생성
        sensor_df["date"] = sensor_df[date_col_sensor].dt.date
        sensor_df["hour"] = sensor_df[date_col_sensor].dt.hour
        sensor_df["time"] = sensor_df[date_col_sensor].dt.time

    # --- 주 선택 슬라이더 동기화 ---
    if "weeks" not in st.session_state:
        st.session_state.weeks = 7  # 초기값


    def update_weeks_1():
        st.session_state.weeks = st.session_state.weeks_slider_1


    def update_weeks_2():
        st.session_state.weeks = st.session_state.weeks_slider_2


    weeks1 = st.slider("평균 계산 기간 (주 단위) - 센서 평균용",
                       1, 7, st.session_state.weeks, key="weeks_slider_1", on_change=update_weeks_1)
    days = st.session_state.weeks * 7

    # 표준화된 동적 컬럼명 (days 기반)
    temp_col_name = f"{days}일평균온도(24시간)"
    hum_col_name = f"{days}일평균습도(08~18시)"
    co2_col_name = f"{days}일평균CO₂(06~18시)"
    solar_col_name = f"{days}일평균누적일사량(0:00기준)"

    # -------------------------------------------------------------
    # 매핑 계산 (yield_df 행마다 sensor 데이터로 파생변수 생성)
    # -------------------------------------------------------------
    results = []

    for idx, row in yield_df.iterrows():
        date = row[date_col_yield]
        start_date = date - timedelta(days=days)

        mask = (sensor_df[date_col_sensor] >= start_date) & (sensor_df[date_col_sensor] <= date)
        subset = sensor_df.loc[mask]

        # 초기값 None (존재하지 않으면 None 유지)
        avg_solar = None
        avg_co2 = None
        avg_temp = None
        avg_hum = None

        if not subset.empty:
            # --- 일사량(0시 기준): 각 date의 00:00 레코드에서 일사량 추출 후 그 날짜들 평균
            midnight_values = subset[subset["time"].astype(str) == "00:00:00"]
            if not midnight_values.empty:
                midnight_daily = midnight_values.groupby("date")[solar_col].first().reset_index()
                if not midnight_daily.empty:
                    avg_solar = midnight_daily[solar_col].mean()

            # --- CO2 (06~18시)
            co2_daytime = subset[(subset["hour"] >= 6) & (subset["hour"] <= 18)]
            if not co2_daytime.empty:
                co2_daily_mean = co2_daytime.groupby("date")[co2_col].mean().reset_index()
                if not co2_daily_mean.empty:
                    avg_co2 = co2_daily_mean[co2_col].mean()

            # --- 온도 (24시간 평균)
            if temp_col in subset.columns:
                avg_temp = subset[temp_col].mean()

            # --- 습도 (08~18시)
            hum_daytime = subset[(subset["hour"] >= 8) & (subset["hour"] <= 18)]
            if not hum_daytime.empty and hum_col in hum_daytime.columns:
                avg_hum = hum_daytime[hum_col].mean()

        # 결과 행에 반드시 동적 컬럼명으로 저장 (일관성 확보)
        result_row = {
            "조사일자": date,
            "수확수": row[harvest_count_col] if harvest_count_col in row else None,
            "평균과중": row[harvest_weight_col] if harvest_weight_col in row else None,
            temp_col_name: avg_temp,
            hum_col_name: avg_hum,
            co2_col_name: avg_co2,
            solar_col_name: avg_solar
        }

        # 생육 컬럼 추가 (사용자가 매핑한 컬럼명에서 값 추출)
        for gf, col in growth_cols.items():
            if col and col in row.index:
                result_row[gf] = row[col]
            else:
                result_row[gf] = None

        results.append(result_row)

    df = pd.DataFrame(results)

    st.subheader("매핑 데이터")
    st.dataframe(df)

    # -------------------------------------------------------------
    # 환경 컬럼 매핑 (사용자에게 보여줄 라벨 → 실제 컬럼명)
    # -------------------------------------------------------------
    env_mapping = {
        f"{days}일평균온도(24시간)": temp_col_name,
        f"{days}일평균습도(08~18시)": hum_col_name,
        f"{days}일평균CO₂(06~18시)": co2_col_name,
        f"{days}일평균누적일사량(0:00기준)": solar_col_name
    }

    env_cols = st.multiselect(
        "환경 그래프로 표시할 항목 선택",
        list(env_mapping.keys()),
        default=list(env_mapping.keys())
    )

    # -------------------------------------------------------------
    # 환경 그래프 출력 (2열 배치)
    # -------------------------------------------------------------
    if env_cols:
        for i in range(0, len(env_cols), 2):
            cols = st.columns(2)
            for j, label in enumerate(env_cols[i:i + 2]):
                with cols[j]:
                    true_col = env_mapping[label]
                    if true_col in df.columns:
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.plot(df["조사일자"], df[true_col], marker="o", linestyle="-")
                        ax.set_title(f"{label} 시계열")
                        ax.set_xlabel("조사일자")
                        ax.set_ylabel(label)
                        ax.tick_params(axis='x', rotation=45)
                        ax.grid(True, linestyle="--", alpha=0.5)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info(f"컬럼 '{true_col}'(실제 데이터)가 없어 '{label}' 그래프를 그릴 수 없습니다.")

    # --- 날짜 정렬
    df = df.sort_values("조사일자")

    # -------------------------------------------------------------
    # 그래프로 표시할 항목 선택(수확/생육)
    # -------------------------------------------------------------
    growth_options = ["수확수", "평균과중"] + growth_features

    plot_cols = st.multiselect(
        "그래프로 표시할 항목 선택",
        growth_options,
        default=["수확수", "평균과중"]
    )

    if plot_cols:
        for i in range(0, len(plot_cols), 3):
            cols = st.columns(3)
            for j, col_name in enumerate(plot_cols[i:i + 3]):
                with cols[j]:
                    if col_name in df.columns:
                        fig, ax = plt.subplots(figsize=(4.5, 3))
                        ax.plot(df["조사일자"], df[col_name], marker="o", linestyle="-")
                        ax.set_title(f"{col_name} 시계열")
                        ax.set_xlabel("조사일자")
                        ax.set_ylabel(col_name)
                        ax.tick_params(axis='x', rotation=45)
                        ax.grid(True, linestyle="--", alpha=0.5)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info(f"컬럼 '{col_name}' 가 없어 그래프를 그릴 수 없습니다.")

    # --- 🌿 환경 vs 생육 2축 시계열 그래프 (4개 비교) ---
    st.subheader("🌿 환경 vs 생육 2축 시계열 그래프 (4개 비교)")

    # env_list: 좌측 레이블과 실제 컬럼명(동적)
    env_list = [
        ("평균온도", temp_col_name),
        ("평균습도", hum_col_name),
        ("평균CO₂", co2_col_name),
        ("평균누적일사량", solar_col_name)
    ]

    growth_choice = st.selectbox(
        "생육 컬럼 선택 (2축 그래프에서 표시할 항목)",
        growth_options,
        index=0
    )

    for i in range(0, len(env_list), 2):
        cols = st.columns(2)
        for j, (title, col_name) in enumerate(env_list[i:i + 2]):
            with cols[j]:
                if col_name not in df.columns:
                    st.info(f"환경 컬럼 '{col_name}' 가 없어 '{title}' 플롯을 건너뜁니다.")
                    continue
                if growth_choice not in df.columns:
                    st.info(f"생육 컬럼 '{growth_choice}' 가 없어 오른쪽 축을 표시할 수 없습니다.")
                fig, ax1 = plt.subplots(figsize=(5.5, 3.5))
                # 환경 (왼쪽)
                color1 = "tab:blue"
                ax1.set_xlabel("조사일자")
                ax1.set_ylabel(title, color=color1)
                ax1.plot(df["조사일자"], df[col_name], color=color1, marker="o", label=title)
                ax1.tick_params(axis='y', labelcolor=color1)
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, linestyle="--", alpha=0.4)
                # 생육 (오른쪽) — only if exists
                ax2 = ax1.twinx()
                color2 = "tab:red"
                if growth_choice in df.columns:
                    ax2.plot(df["조사일자"], df[growth_choice], color=color2, marker="s", linestyle="--", label=growth_choice)
                    ax2.set_ylabel(growth_choice, color=color2)
                    ax2.tick_params(axis='y', labelcolor=color2)
                ax1.legend(loc="best", fontsize=8)
                ax1.set_title(f"{title} vs {growth_choice}", fontsize=11)
                st.pyplot(fig)
                plt.close(fig)

    # --- Plotly interactive 2x2 ---
    st.subheader("🌿 환경요소 vs 생육컬럼 2축 시계열 그래프 (Plotly 인터랙티브 2×2)")

    growth_choice_plotly = st.selectbox(
        "생육 컬럼 선택 (Plotly 그래프용)",
        growth_options,
        index=0,
        key="plotly_growth_choice"
    )

    # prepare subplot
    fig_plotly = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{title} vs {growth_choice_plotly}" for title, _ in env_list],
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}]]
    )

    for idx, (title, env_col) in enumerate(env_list):
        row = idx // 2 + 1
        col = idx % 2 + 1

        if env_col in df.columns:
            fig_plotly.add_trace(
                go.Scatter(
                    x=df["조사일자"],
                    y=df[env_col],
                    mode='lines+markers',
                    name=title,
                    line=dict(color='blue'),
                    hovertemplate=f"{title}: %{{y}}<br>조사일자: %{{x}}"
                ), row=row, col=col, secondary_y=False
            )
        else:
            # add empty trace so subplot remains visible
            fig_plotly.add_trace(
                go.Scatter(x=[], y=[], name=f"{title} (no data)"),
                row=row, col=col, secondary_y=False
            )

        if growth_choice_plotly in df.columns:
            fig_plotly.add_trace(
                go.Scatter(
                    x=df["조사일자"],
                    y=df[growth_choice_plotly],
                    mode='lines+markers',
                    name=growth_choice_plotly,
                    line=dict(color='red', dash='dash'),
                    hovertemplate=f"{growth_choice_plotly}: %{{y}}<br>조사일자: %{{x}}"
                ), row=row, col=col, secondary_y=True
            )
        else:
            fig_plotly.add_trace(
                go.Scatter(x=[], y=[], name=f"{growth_choice_plotly} (no data)"),
                row=row, col=col, secondary_y=True
            )

        fig_plotly.update_yaxes(title_text=title, row=row, col=col, secondary_y=False)
        fig_plotly.update_yaxes(title_text=growth_choice_plotly, row=row, col=col, secondary_y=True)

    fig_plotly.update_layout(
        height=800,
        width=950,
        title_text="환경요소 vs 생육컬럼 2축 시계열 (인터랙티브)",
        showlegend=True,
        hovermode="x unified",
        margin=dict(l=30, r=30, t=60, b=30)
    )

    st.plotly_chart(fig_plotly, use_container_width=True)

    # --- 모델 선택 ---
    st.subheader("모델 선택")
    model_options = ["RandomForest", "GradientBoosting", "XGBoost", "LGBM", "GaussianNB"]
    model_choice = st.selectbox("모델 선택", model_options)

    target_col = st.selectbox("예측 대상 컬럼 선택", ["수확수", "평균과중"] + growth_features)
    features = [col for col in df.columns if col not in ["조사일자", "수확수", "평균과중"] + growth_features]

    X = df[features]
    y = df[target_col]
    X = X.fillna(X.mean())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_choice == "RandomForest":
        model = RandomForestRegressor(random_state=42)
    elif model_choice == "GradientBoosting":
        model = GradientBoostingRegressor(random_state=42)
    elif model_choice == "XGBoost":
        model = XGBRegressor(random_state=42)
    elif model_choice == "LGBM":
        model = LGBMRegressor(random_state=42)
    elif model_choice == "GaussianNB":
        model = GaussianNB()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    weeks2 = st.slider("평균 계산 기간 (주 단위) - 모델용",
                       1, 7, st.session_state.weeks, key="weeks_slider_2", on_change=update_weeks_2)
    days = st.session_state.weeks * 7

    # --- 평가지표 ---
    st.subheader("모델 평가 지표")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    st.write(f"R²: {r2_score(y_test, y_pred):.3f}")

    # ---------------------------
    # SHAP, Feature Importance 레이아웃 재배치 및 ICE/PDP/ALE 추가
    # ---------------------------

    import math
    from sklearn.utils import check_array


    # 간단한 ALE 계산 함수 (수치형 feature 전용, 모델의 predict 사용)
    def compute_ale(model, X, feature, bins=10):
        """
        간단한 1차원 ALE 근사
        model: 학습된 모델 (predict 메서드 사용)
        X: DataFrame (원본 특성 행렬)
        feature: feature 이름(string)
        bins: bin 수
        returns: bin_centers, ale_values
        """
        x = X[feature].values
        # remove nan rows for feature
        mask = ~np.isnan(x)
        x = x[mask]
        X_valid = X.loc[mask].reset_index(drop=True)
        percentiles = np.linspace(0, 100, bins + 1)
        cutpoints = np.percentile(x, percentiles)
        # 중복 컷포인트 처리: 유니크로
        cutpoints = np.unique(cutpoints)
        if len(cutpoints) < 2:
            # 변동이 거의 없을 때
            return np.array([np.mean(x)]), np.array([0.0])

        # 각 구간별 평균 기여 계산
        local_effects = []
        bin_centers = []
        for i in range(len(cutpoints) - 1):
            lo, hi = cutpoints[i], cutpoints[i + 1]
            # 해당 구간에 속하는 인덱스
            in_bin = (X_valid[feature] >= lo) & (X_valid[feature] <= hi)
            if in_bin.sum() == 0:
                # 해당 구간에 점이 없으면 0 넣기
                local_effects.append(0.0)
                bin_centers.append((lo + hi) / 2.0)
                continue
            X_lo = X_valid.copy()
            X_hi = X_valid.copy()
            # 왼쪽 경계값으로, 오른쪽 경계값으로 바꿔서 예측 차이를 봄
            X_lo.loc[in_bin, feature] = lo
            X_hi.loc[in_bin, feature] = hi
            try:
                preds_hi = model.predict(X_hi)
                preds_lo = model.predict(X_lo)
            except Exception:
                # some models require numpy array
                preds_hi = model.predict(X_hi.values)
                preds_lo = model.predict(X_lo.values)
            diff = preds_hi - preds_lo
            # 지역 평균 기여
            local_effect = diff[in_bin.values].mean() if in_bin.sum() > 0 else 0.0
            local_effects.append(local_effect)
            bin_centers.append((lo + hi) / 2.0)

        # 누적합으로 ALE 계산 (baseline을 0으로 맞춤)
        ale = np.cumsum(local_effects)
        # 평균을 0 기준으로 조정
        ale = ale - ale.mean()
        return np.array(bin_centers), ale


    # --------------------- XAI: SHAP / FI / ICE / PDP / ALE + 자동 리포트 ---------------------
    import shap
    from sklearn.inspection import PartialDependenceDisplay
    from sklearn.linear_model import LinearRegression

    st.subheader("SHAP / Feature Importance / ICE / PDP / ALE — 자동 리포트 포함")

    # 안전하게 필요한 objects 준비
    try:
        # model, X_train, X_test, features 등이 존재한다고 가정
        _ = model
        _ = X_train
        _ = X_test
    except Exception as e:
        st.error("모델·데이터(예: model, X_train, X_test 등)가 준비되어야 XAI를 실행할 수 있습니다.")
        st.stop()


    # ---------- 유틸 함수들 ----------
    def safe_inverse_transform(scaler, arr):
        try:
            return scaler.inverse_transform(arr)
        except Exception:
            return arr


    def find_top_contiguous_interval(x, y, top_frac=0.9, min_width=1):
        """
        x: 1d array (sorted)
        y: 1d array same length
        목표: y가 최대인 연속 구간을 찾아 (start, end, mean_y, max_y)
        방안: y의 top percentile 영역에서 가장 긴 연속 구간 선택
        """
        import numpy as np
        thresh = np.quantile(y, top_frac)
        mask = y >= thresh
        # find contiguous true segments
        segments = []
        i = 0
        while i < len(mask):
            if mask[i]:
                j = i
                while j < len(mask) and mask[j]:
                    j += 1
                if (j - i) >= min_width:
                    segments.append((i, j - 1))
                i = j
            else:
                i += 1
        if not segments:
            # fallback: return argmax single point expanded by 1 on each side
            idx = int(np.argmax(y))
            left = max(0, idx - 1)
            right = min(len(x) - 1, idx + 1)
            return x[left], x[right], float(np.mean(y[left:right + 1])), float(np.max(y[left:right + 1]))
        # choose segment with largest mean y
        best = None
        best_score = -1e9
        for s, t in segments:
            score = float(np.mean(y[s:t + 1]))
            if score > best_score:
                best_score = score
                best = (s, t)
        s, t = best
        return x[s], x[t], float(np.mean(y[s:t + 1])), float(np.max(y[s:t + 1]))


    def summarize_pdp(model, X, feature, grid_resolution=50):
        """
        Compute PDP (average) and return summary info: best interval where PDP is high.
        """
        import numpy as np
        try:
            display = PartialDependenceDisplay.from_estimator(model, X, [feature], kind="average",
                                                              grid_resolution=grid_resolution)
            # sklearn returns axes data in different versions; try to extract x and y
            # Attempt 1: from display
            try:
                pdp_ax = display.axes_[0, 0]
                lines = pdp_ax.get_lines()
                if len(lines) > 0:
                    x = lines[0].get_xdata()
                    y = lines[0].get_ydata()
                else:
                    # fallback: use display.pd_results if present
                    x = display.pd_results[0][0]
                    y = display.pd_results[0][1]
            except Exception:
                # fallback: try attribute
                res = display.pd_results[0]
                x = res["values"]
                y = res["average"]
        except Exception:
            # fallback: manual grid eval
            col = X[feature]
            x = np.linspace(col.min(), col.max(), grid_resolution)
            y = []
            Xbase = X.copy()
            for val in x:
                Xtmp = Xbase.copy()
                Xtmp[feature] = val
                preds = model.predict(Xtmp)
                if preds.ndim == 2:
                    preds = preds.mean(axis=1)
                y.append(np.mean(preds))
            x = np.array(x)
            y = np.array(y)
        # find top contiguous interval (upper 10% area)
        start, end, mean_y, max_y = find_top_contiguous_interval(x, y, top_frac=0.9)
        return x, y, {"best_interval": (start, end), "mean_val": float(mean_y), "max_val": float(max_y)}


    def summarize_ice_linear_slope(model, X, feature, n_samples=50):
        """
        For ICE: sample up to n_samples rows, compute for each row the model prediction
        across a grid of feature values, and estimate slope via simple linear regression.
        Returns average slope (pred change per unit feature), and sign summary.
        """
        import numpy as np
        Xs = X.sample(n=min(n_samples, len(X)), random_state=42)
        xs = np.linspace(X[feature].min(), X[feature].max(), 30)
        slopes = []
        for _, row in Xs.iterrows():
            Xtmp = pd.DataFrame(np.tile(row.values, (len(xs), 1)), columns=X.columns)
            Xtmp[feature] = xs
            preds = model.predict(Xtmp)
            if preds.ndim == 2:
                # if multioutput, average
                preds = preds.mean(axis=1)
            # fit linear regression slope
            lr = LinearRegression()
            lr.fit(xs.reshape(-1, 1), preds)
            slopes.append(lr.coef_[0])
        slopes = np.array(slopes)
        return float(np.mean(slopes)), float(np.std(slopes)), len(slopes)


    def summarize_ale_intervals(bin_centers, ale_vals):
        """
        Analyze ALE array to find positive/negative intervals and steep changes (threshold by derivative)
        """
        import numpy as np
        bc = np.array(bin_centers)
        av = np.array(ale_vals)
        # derivative
        deriv = np.gradient(av, bc)
        # find peak positive regions
        pos_mask = av > 0
        neg_mask = av < 0
        # find steep points where absolute derivative > 1.5 * std(deriv)
        thr = 1.5 * (np.std(deriv) + 1e-9)
        steep_idx = np.where(np.abs(deriv) > thr)[0]

        # contiguous positive/negative ranges
        def contiguous_ranges(mask):
            ranges = []
            i = 0
            while i < len(mask):
                if mask[i]:
                    j = i
                    while j < len(mask) and mask[j]:
                        j += 1
                    ranges.append((i, j - 1))
                    i = j
                else:
                    i += 1
            return ranges

        pos_ranges = contiguous_ranges(pos_mask)
        neg_ranges = contiguous_ranges(neg_mask)
        # convert to value intervals
        pos_intervals = [(float(bc[s]), float(bc[t]), float(np.mean(av[s:t + 1]))) for s, t in pos_ranges]
        neg_intervals = [(float(bc[s]), float(bc[t]), float(np.mean(av[s:t + 1]))) for s, t in neg_ranges]
        steep_points = [(int(i), float(bc[i]), float(deriv[i])) for i in steep_idx]
        return {"pos_intervals": pos_intervals, "neg_intervals": neg_intervals, "steep_points": steep_points}


    # ---------- SHAP & FI (top row) ----------
    top_col1, top_col2 = st.columns([1, 1])

    with top_col1:
        st.markdown("### 🔍 SHAP Summary")
        if model_choice == "GaussianNB":
            st.info("GaussianNB 모델은 SHAP 사용이 제한적입니다.")
        else:
            try:
                # Compute or reuse SHAP
                try:
                    shap_values  # if existing
                except NameError:
                    explainer = shap.Explainer(model, X_train)
                    shap_values = explainer(X_test)
                # summary plot
                fig_shap, ax_shap = plt.subplots(figsize=(6, 4))
                shap.summary_plot(shap_values, X_test, show=False)
                st.pyplot(fig_shap)
                plt.close(fig_shap)
                # Summary table: mean(|shap|)
                shap_mean = np.abs(shap_values.values).mean(axis=0)
                shap_df = pd.DataFrame({"Feature": features, "Mean(|SHAP|)": shap_mean}).sort_values(by="Mean(|SHAP|)",
                                                                                                     ascending=False)
                st.dataframe(shap_df.head(12).round(6))
                # Text summary: top contributors
                top_feats = shap_df.head(5)
                text_lines = []
                total = shap_df["Mean(|SHAP|)"].sum()
                for i, row in top_feats.iterrows():
                    pct = 100.0 * row["Mean(|SHAP|)"] / total if total > 0 else 0.0
                    text_lines.append(f"{row['Feature']}: 영향도 {pct:.1f}%")
                st.markdown("**상위 특징(Mean |SHAP| 기준)**")
                for ln in text_lines:
                    st.write("•", ln)
            except Exception as e:
                st.error(f"SHAP 계산/시각화 오류: {e}")

    with top_col2:
        st.markdown("### 📊 Feature Importance (Model-based)")
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                fi_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance",
                                                                                                   ascending=False)
            else:
                # fallback: permutation importance would be better; here zero-fill
                fi_df = pd.DataFrame({"Feature": features, "Importance": np.zeros(len(features))}).sort_values(
                    by="Importance", ascending=False)
                st.warning("선택 모델에 feature_importances_ 속성이 없습니다. Permutation importance 권장을 권장합니다.")
            # plot
            fig_fi, ax_fi = plt.subplots(figsize=(6, 4))
            ax_fi.barh(fi_df["Feature"], fi_df["Importance"])
            ax_fi.invert_yaxis()
            ax_fi.set_title("Feature Importance")
            st.pyplot(fig_fi)
            plt.close(fig_fi)
            # text summary
            st.markdown("**Feature Importance 요약**")
            top = fi_df.head(5)
            tot = fi_df["Importance"].sum()
            for _, r in top.iterrows():
                pct = 100.0 * r["Importance"] / tot if tot > 0 else 0.0
                st.write(f"• {r['Feature']}: {pct:.1f}%")
        except Exception as e:
            st.error(f"Feature Importance 처리 오류: {e}")

    # ---------- SHAP 상세: 특정 샘플 영향 (예: CO2가 +2kg 기여 같은 문장) ----------
    st.markdown("### 🔎 SHAP 샘플별 상세 해석")
    sample_idx = st.number_input("샘플 인덱스 (X_test 기준)", min_value=0, max_value=max(0, len(X_test) - 1), value=0, step=1)
    if model_choice != "GaussianNB":
        try:
            # get one sample
            xi = X_test.reset_index(drop=True).iloc[sample_idx:sample_idx + 1]
            # shap values for that sample
            shp_s = shap_values[sample_idx].values  # shape (n_features,)
            shp_df = pd.DataFrame({"Feature": features, "SHAP": shp_s}).sort_values(by="SHAP", key=lambda s: np.abs(s),
                                                                                    ascending=False)
            st.dataframe(shp_df.head(20).round(4))
            # textual interpretation: top positive / negative contributors
            pos = shp_df[shp_df["SHAP"] > 0].head(5)
            neg = shp_df[shp_df["SHAP"] < 0].head(5)
            if not pos.empty:
                st.write("상위 양(+) 기여 변수:")
                for _, r in pos.iterrows():
                    st.write(f"• {r['Feature']}: 예측 증가에 기여 +{r['SHAP']:.3f}")
            if not neg.empty:
                st.write("상위 음(-) 기여 변수:")
                for _, r in neg.iterrows():
                    st.write(f"• {r['Feature']}: 예측 감소에 기여 {r['SHAP']:.3f}")
            # Example style sentence (자동생성)
            if "CO2" in shp_df["Feature"].values or "CO₂" in shp_df["Feature"].values:
                # find CO2 row if exists
                co2_rows = shp_df[shp_df["Feature"].str.contains("CO2|CO₂", regex=True)]
                if not co2_rows.empty:
                    r = co2_rows.iloc[0]
                    sign = "+" if r["SHAP"] > 0 else "-"
                    st.info(f"예시 해석: 특정 샘플에서 CO₂ 농도는 수확량 예측에 {sign}{abs(r['SHAP']):.3f}만큼 기여했습니다.")
        except Exception as e:
            st.error(f"SHAP 샘플 분석 오류: {e}")
    else:
        st.info("GaussianNB 모델은 SHAP 상세 분석을 지원하지 않습니다.")

    # ---------- 하단: ICE / PDP / ALE (그래프 + 자동 리포트) ----------
    st.subheader("ICE / PDP / ALE — 그래프 + 최적 구간 리포트")

    ice_feature = st.selectbox("분석할 Feature 선택 (ICE/PDP/ALE)", features, key="xai_feature")
    n_samples = st.slider("ICE 샘플 수", 1, max(1, len(X_test)), value=min(50, len(X_test)), key="ice_samples")
    ale_bins = st.slider("ALE bins 수", 4, 30, 10)

    col_ice, col_pdp, col_ale = st.columns(3)

    # ICE
    with col_ice:
        st.markdown("**ICE Plot & 민감도(단위당 변화량)**")
        try:
            fig_ice, ax_ice = plt.subplots(figsize=(5, 3))
            try:
                PartialDependenceDisplay.from_estimator(model,
                                                        X_test.sample(n=min(n_samples, len(X_test)), random_state=42),
                                                        features=[ice_feature], kind="individual", ax=ax_ice,
                                                        line_kw={"alpha": 0.3})
            except Exception:
                # fallback simple: draw for sampled rows
                Xs = X_test.sample(n=min(n_samples, len(X_test)), random_state=42)
                xs = np.linspace(X_test[ice_feature].min(), X_test[ice_feature].max(), 50)
                for _, row in Xs.iterrows():
                    Xtmp = pd.DataFrame(np.tile(row.values, (len(xs), 1)), columns=X_test.columns)
                    Xtmp[ice_feature] = xs
                    preds = model.predict(Xtmp)
                    if preds.ndim == 2:
                        preds = preds.mean(axis=1)
                    ax_ice.plot(xs, preds, alpha=0.2)
            ax_ice.set_title(f"ICE: {ice_feature}")
            ax_ice.set_xlabel(ice_feature)
            ax_ice.set_ylabel("Predicted")
            st.pyplot(fig_ice)
            plt.close(fig_ice)
            # sensitivity summary (slope)
            mean_slope, std_slope, cnt = summarize_ice_linear_slope(model, X_test, ice_feature,
                                                                    n_samples=min(n_samples, len(X_test)))
            st.write(f"샘플 {cnt}개 평균 기울기(단위 {ice_feature} 당 예측 변화): {mean_slope:.4f} ± {std_slope:.4f}")
            if mean_slope > 0:
                st.info(f"해석: {ice_feature}가 증가할 때 평균적으로 예측(수확량 등)도 증가하는 경향이 있습니다.")
            elif mean_slope < 0:
                st.info(f"해석: {ice_feature}가 증가할 때 평균적으로 예측이 감소하는 경향이 있습니다.")
            else:
                st.info("해석: 평균적인 민감도가 거의 0입니다.")
        except Exception as e:
            st.error(f"ICE 처리 오류: {e}")

    # PDP
    with col_pdp:
        st.markdown("**PDP (Average) & 최적 구간**")
        try:
            xvals, yvals, pdp_summary = summarize_pdp(model, X_test, ice_feature, grid_resolution=50)
            fig_pdp, ax_pdp = plt.subplots(figsize=(5, 3))
            ax_pdp.plot(xvals, yvals, color="red", lw=2)
            ax_pdp.set_title(f"PDP: {ice_feature}")
            ax_pdp.set_xlabel(ice_feature)
            ax_pdp.set_ylabel("Predicted")
            st.pyplot(fig_pdp)
            plt.close(fig_pdp)
            s, e, meanv, maxv = pdp_summary["best_interval"][0], pdp_summary["best_interval"][1], pdp_summary[
                "mean_val"], pdp_summary["max_val"]
            # NOTE: summarize_pdp returns dict with best_interval tuple as value, but above we returned differently; handle both
            try:
                start, end = pdp_summary["best_interval"]
            except Exception:
                start, end = s, e
            st.write(f"최적(예측이 큰) 구간: {start:.3f} ~ {end:.3f}")
            st.write(f"구간 평균 예측값: {pdp_summary['mean_val']:.3f}, 구간 최대값: {pdp_summary['max_val']:.3f}")
            # agricultural interpretation template
            st.markdown("**농업적 해석(예시)**")
            st.write(f"• 만약 {ice_feature}가 {start:.1f}–{end:.1f} 구간에 자주 머문다면 모델은 이 구간을 비교적 우호적으로 평가합니다.")
        except Exception as e:
            st.error(f"PDP 처리 오류: {e}")

    # ALE
    with col_ale:
        st.markdown("**ALE (근사) & 임계구간 탐지**")
        try:
            bin_centers, ale_vals = compute_ale(model, X_test.reset_index(drop=True), ice_feature, bins=ale_bins)
            fig_ale, ax_ale = plt.subplots(figsize=(5, 3))
            if len(bin_centers) > 1:
                ax_ale.plot(bin_centers, ale_vals, marker="o", linestyle="-")
            else:
                ax_ale.hlines(ale_vals[0], bin_centers[0] - 0.5, bin_centers[0] + 0.5)
            ax_ale.set_title(f"ALE (approx): {ice_feature}")
            ax_ale.set_xlabel(ice_feature)
            ax_ale.set_ylabel("ALE")
            st.pyplot(fig_ale)
            plt.close(fig_ale)
            # summarize ALE
            ale_summary = summarize_ale_intervals(bin_centers, ale_vals)
            if ale_summary["pos_intervals"]:
                st.write("모델이 우호적으로 보는 구간(양의 ALE):")
                for a, b, mv in ale_summary["pos_intervals"]:
                    st.write(f"• {a:.2f} ~ {b:.2f} (평균 ALE: {mv:.3f})")
            if ale_summary["neg_intervals"]:
                st.write("모델이 불리하게 보는 구간(음의 ALE):")
                for a, b, mv in ale_summary["neg_intervals"]:
                    st.write(f"• {a:.2f} ~ {b:.2f} (평균 ALE: {mv:.3f})")
            if ale_summary["steep_points"]:
                st.write("ALE에서 급격히 변화하는 점(임계점) 예시:")
                for idx, val, deriv in ale_summary["steep_points"][:5]:
                    st.write(f"• idx {idx}, {ice_feature}≈{val:.2f}, 기울기≈{deriv:.3f}")
            # example agricultural interpretation template
            st.markdown("**농업적(생리학적) 해석 예시 템플릿**")
            st.write("• 모델이 특정 온도구간을 우호적으로 평가한다면(예: 20~21℃), 해당 구간에서 광합성·개화·착과가 유리할 가능성이 있습니다.")
            st.write("• 반대로 특정 구간에서 ALE가 급격히 감소하면(임계온도 존재), 그 지점을 알람으로 설정하여 관리/환기/차광 등의 제어전략을 고려하세요.")
        except Exception as e:
            st.error(f"ALE 처리 오류: {e}")

    # ---------------------------
    # 2D ALE (interaction) + ALE with bootstrap CI
    # ---------------------------
    import numpy as np
    import matplotlib.pyplot as plt


    def _safe_predict(model, X_df):
        """model.predict가 DataFrame을 바로 받지 못할 경우 대비"""
        try:
            return model.predict(X_df)
        except Exception:
            return model.predict(X_df.values)


    def compute_2d_ale(model, X, feat_x, feat_y, grid=10, min_count_in_cell=1):
        """
        2D ALE (second-order interaction) approximation.
        Returns:
            x_centers, y_centers, ale2d (shape (len(x_centers), len(y_centers)))
        Method (simplified Apley & Zhu):
          - Partition feat_x and feat_y into bins (quantile-based)
          - For each cell (i,j) compute average second-order finite diff:
              delta = f(x_hi,y_hi) - f(x_lo,y_hi) - f(x_hi,y_lo) + f(x_lo,y_lo)
          - raw_interaction[i,j] = mean(delta over samples in cell)
          - cumulative (double sum) over x then y to get ALE surface
          - center ALE to have mean 0
        """
        X = X.copy().reset_index(drop=True)
        xv = X[feat_x].values
        yv = X[feat_y].values

        # edges by quantile to respect data density
        x_edges = np.unique(np.percentile(xv, np.linspace(0, 100, grid + 1)))
        y_edges = np.unique(np.percentile(yv, np.linspace(0, 100, grid + 1)))

        # if too few unique edges -> fallback to unique centers
        if len(x_edges) < 2:
            return np.array([np.mean(xv)]), np.array([np.mean(yv)]), np.array([[0.0]])
        if len(y_edges) < 2:
            return np.array([np.mean(xv)]), np.array([np.mean(yv)]), np.array([[0.0]])

        nx = len(x_edges) - 1
        ny = len(y_edges) - 1

        raw = np.zeros((nx, ny))
        counts = np.zeros((nx, ny), dtype=int)

        # Precompute bin membership
        x_bin_idx = np.digitize(xv, x_edges, right=False) - 1  # 0..nx-1
        y_bin_idx = np.digitize(yv, y_edges, right=False) - 1  # 0..ny-1

        # clamp indices (edge cases)
        x_bin_idx = np.clip(x_bin_idx, 0, nx - 1)
        y_bin_idx = np.clip(y_bin_idx, 0, ny - 1)

        # For each cell compute the average second-order diff
        for i in range(nx):
            x_lo, x_hi = x_edges[i], x_edges[i + 1]
            for j in range(ny):
                y_lo, y_hi = y_edges[j], y_edges[j + 1]
                # mask of samples that fall in this cell
                mask = (x_bin_idx == i) & (y_bin_idx == j)
                idxs = np.where(mask)[0]
                counts[i, j] = len(idxs)
                if len(idxs) < min_count_in_cell:
                    raw[i, j] = 0.0
                    continue

                # build 4 matrices for modified rows (only for rows in cell)
                X_ll = X.loc[idxs].copy()  # x_lo, y_lo
                X_lh = X.loc[idxs].copy()  # x_lo, y_hi
                X_hl = X.loc[idxs].copy()  # x_hi, y_lo
                X_hh = X.loc[idxs].copy()  # x_hi, y_hi

                X_ll[feat_x] = x_lo;
                X_ll[feat_y] = y_lo
                X_lh[feat_x] = x_lo;
                X_lh[feat_y] = y_hi
                X_hl[feat_x] = x_hi;
                X_hl[feat_y] = y_lo
                X_hh[feat_x] = x_hi;
                X_hh[feat_y] = y_hi

                p_hh = _safe_predict(model, X_hh)
                p_hl = _safe_predict(model, X_hl)
                p_lh = _safe_predict(model, X_lh)
                p_ll = _safe_predict(model, X_ll)

                # second-order finite difference per sample
                delta = p_hh - p_hl - p_lh + p_ll
                raw[i, j] = np.mean(delta)

        # double cumulative sum to obtain ALE surface (integration)
        cum_x = np.cumsum(raw, axis=0)  # cumulative over x for each y
        cum_xy = np.cumsum(cum_x, axis=1)  # then cumulative over y

        # The above orientation gives shape (nx, ny). Depending on plotting, may transpose later.
        ale2d = cum_xy

        # Centering: subtract mean (only over cells with counts>0)
        valid = counts > 0
        if valid.any():
            mean_val = np.mean(ale2d[valid])
        else:
            mean_val = 0.0
        ale2d = ale2d - mean_val

        # bin centers for plotting
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

        return x_centers, y_centers, ale2d


    def compute_ale_with_bootstrap_ci(model, X, feature, bins=10, B=50, ci=(2.5, 97.5), random_state=42):
        """
        Compute 1D ALE and bootstrap confidence interval.
        Returns:
          bin_centers, ale_mean, ale_lower, ale_upper, ale_all (B x n_bins array)
        """
        rng = np.random.RandomState(random_state)
        ale_list = []
        # If dataset small, reduce B
        for b in range(B):
            # bootstrap sample of rows (with replacement)
            idxs = rng.randint(0, len(X), size=len(X))
            Xb = X.iloc[idxs].reset_index(drop=True)
            centers, ale_vals = compute_ale(model, Xb, feature, bins=bins)
            # ensure consistent length: if compute_ale returns single-point for degenerate, pad/reshape
            ale_list.append(ale_vals)

        # pad to same length if necessary (rare if bins same)
        lengths = [len(a) for a in ale_list]
        maxlen = max(lengths)
        arr = np.zeros((B, maxlen))
        arr[:] = np.nan
        for i, a in enumerate(ale_list):
            arr[i, :len(a)] = a

        # compute mean and percentile CI ignoring nan
        ale_mean = np.nanmean(arr, axis=0)
        ale_lower = np.nanpercentile(arr, ci[0], axis=0)
        ale_upper = np.nanpercentile(arr, ci[1], axis=0)

        # use centers from last run or recompute on original X for a reliable grid
        centers, _ = compute_ale(model, X, feature, bins=bins)
        return centers, ale_mean, ale_lower, ale_upper, arr


    # ---------------------------
    # Streamlit UI additions to call 2D ALE and bootstrap CI
    # ---------------------------
    st.markdown("---")
    st.subheader("추가: 2차원 ALE (교호작용) 및 ALE 부트스트랩 신뢰구간")

    # ==========================================================
    # (1) 2차원 ALE (교호작용) + 자동 해석 리포트
    # ==========================================================
    with st.expander("2차원 ALE (feat A × feat B) 계산 / 시각화"):
        col_left, col_right = st.columns([1, 1])  # 왼쪽 UI, 오른쪽 그래프

        with col_left:
            feat_x_2d = st.selectbox("X 축 (Feature A)", features, index=0, key="feat_x_2d")
            feat_y_2d = st.selectbox(
                "Y 축 (Feature B)", [f for f in features if f != feat_x_2d], index=0, key="feat_y_2d"
            )
            grid_size = st.slider("그리드 크기 (각 축 bin 수)", 4, 20, 8, key="ale2d_grid")
            min_count = st.number_input(
                "셀 당 최소 샘플 수 (노이즈 방지)", min_value=1, max_value=100, value=1, step=1
            )
            run_ale2d = st.button("2D ALE 계산/시각화", key="run_ale2d")

        with col_right:
            if run_ale2d:
                try:
                    with st.spinner("2D ALE 계산 중..."):
                        x_centers, y_centers, ale2d = compute_2d_ale(
                            model, X_test.reset_index(drop=True),
                            feat_x_2d, feat_y_2d,
                            grid=grid_size, min_count_in_cell=min_count
                        )

                    # ----------- 시각화 -----------
                    fig2d, ax2d = plt.subplots(figsize=(6, 4))
                    XX, YY = np.meshgrid(x_centers, y_centers, indexing='xy')

                    cs = ax2d.contourf(XX, YY, ale2d.T, cmap='RdBu_r', levels=20)
                    ax2d.set_xlabel(feat_x_2d)
                    ax2d.set_ylabel(feat_y_2d)
                    ax2d.set_title(f"2D ALE interaction: {feat_x_2d} × {feat_y_2d}")
                    fig2d.colorbar(cs, ax=ax2d, label="ALE (interaction)")
                    st.pyplot(fig2d)
                    plt.close(fig2d)

                    # ========================================================
                    # (추가) 자동 해석 리포트 생성
                    # ========================================================
                    max_ale = np.max(ale2d)
                    min_ale = np.min(ale2d)

                    max_loc = np.unravel_index(np.argmax(ale2d), ale2d.shape)
                    min_loc = np.unravel_index(np.argmin(ale2d), ale2d.shape)

                    x_max = x_centers[max_loc[0]]
                    y_max = y_centers[max_loc[1]]
                    x_min = x_centers[min_loc[0]]
                    y_min = y_centers[min_loc[1]]

                    # 리포트 생성
                    st.subheader("📘 2D ALE 자동 해석 리포트")

                    report = f"""
    ### 🔍 **교호작용 분석 개요**
    특성 **{feat_x_2d}** 와 **{feat_y_2d}** 의 조합이 모델 예측(수확량)에 미치는  
    **순수한 상호작용 효과(Interaction Effect)**를 분석한 결과입니다.  
    즉, 두 변수가 함께 변화할 때 단독 효과로 설명되지 않는 추가적인 영향만을 보여줍니다.

    ---

    ### 🔺 **1) 수확량 증가에 가장 크게 기여하는 조합 (양의 상호작용)**
    - ALE 최대값: **{max_ale:.3f}**
    - 발생 구간:  
      - **{feat_x_2d}: {x_max:.2f}**  
      - **{feat_y_2d}: {y_max:.2f}**

    ➡️ 이 구간에서는 두 변수가 결합할 때 **수확량을 강하게 증가시키는 생육 조건**을 의미합니다.

    ---

    ### 🔻 **2) 수확량 감소에 가장 크게 기여하는 조합 (음의 상호작용)**
    - ALE 최소값: **{min_ale:.3f}**
    - 발생 구간:  
      - **{feat_x_2d}: {x_min:.2f}**  
      - **{feat_y_2d}: {y_min:.2f}**

    ➡️ 이 조합은 두 변수가 동시에 존재할 때  
    단독 효과보다 **더 큰 수확량 감소 효과**를 일으킨다는 뜻입니다.

    ---

    ### 🧭 **3) 상호작용 패턴 요약**
    - 🔴 **양의 ALE(빨간색)** → 수확량 증가 조합  
    - 🔵 **음의 ALE(파란색)** → 수확량 감소 조합  
    - ⚪ **0 근처(흰색)** → 상호작용이 거의 없음  

    두 변수가 서로 영향을 증폭하거나 상쇄하는 **비선형 구조**가 존재함을 의미합니다.

    ---

    ### 🌱 **4) 농업적 해석 (토마토 기준)**
    - 양의 조합 구간은 **생육에 유리하거나 광·온도 환경이 적절하게 맞아 떨어지는 조건**  
    - 음의 조합 구간은 **열·광 스트레스, 과습, 비효율 광합성 발생 가능**  
    - 특정 온도에서 일사량이 증가할 때 또는 특정 일사량에서 온도가 상승할 때  
      **수확량이 급격히 증가/감소하는 구간이 존재함을 모델이 학습했다는 의미**  

    ---

    ### 📌 **5) 의사결정 활용**
    본 분석은 다음 의사결정에 유용합니다:
    - 최적 생육기 조성(온도·일사량 조합 설정)
    - 스트레스 환경 조기 예측
    - 환경 제어 장비(스크린·환기·난방 등) 설정 기준 수립
    - 작기별(초기/중기/후기) 최적 온·광 조건 도출

    ---

    ### 📄 **요약**
    2D ALE 분석을 통해  
    **“{feat_x_2d}와(과) {feat_y_2d}의 조합이 단독 변화보다 어떤 방식으로 수확량에 비선형적인 영향을 주는지”**  
    직접 확인할 수 있습니다.

    이 리포트는 모델이 식물 생육의 상호작용 패턴을 어떻게 학습했는지  
    정량적·직관적으로 설명해주는 자동 분석 결과입니다.
    """

                    st.markdown(report)

                except Exception as e:
                    st.error(f"2D ALE 계산/시각화 오류: {e}")

    # ==========================================================
    # (2) 1D ALE 부트스트랩 신뢰구간 (CI)  - 안전한 길이/NaN 처리 추가
    # ==========================================================
    with st.expander("1D ALE 부트스트랩 신뢰구간 (CI)"):
        col_left, col_right = st.columns([1, 1])

        with col_left:
            ci_feature = st.selectbox("ALE CI 대상 Feature", features, index=0, key="ci_feature")
            B = st.slider("부트스트랩 반복 수 B", 10, 300, 50, step=10, key="ale_boot_B")
            alpha = st.slider("신뢰구간(%) - 상한/하한 퍼센타일 중심값", 80, 99, 95, key="ale_boot_alpha")
            lower_pct = (100 - alpha) / 2.0
            upper_pct = 100 - lower_pct
            run_ale_ci = st.button("ALE + 부트스트랩 CI 계산", key="run_ale_ci")

        with col_right:
            if run_ale_ci:
                try:
                    with st.spinner("ALE 부트스트랩 계산 중..."):
                        centers, mean_ale, lower_ale, upper_ale, all_vals = compute_ale_with_bootstrap_ci(
                            model, X_test.reset_index(drop=True), ci_feature,
                            bins=ale_bins, B=B, ci=(lower_pct, upper_pct)
                        )

                    # -----------------------------------------------
                    # 1) 기본 유효성 검사
                    # -----------------------------------------------
                    if centers is None or len(centers) == 0:
                        st.warning("ALE 계산 결과가 비어있습니다. 입력 데이터 또는 bins 값을 확인하세요.")
                    else:
                        # 길이 일관성 확보: 최소 길이를 기준으로 자른다
                        len_list = [len(arr) for arr in [centers, mean_ale, lower_ale, upper_ale] if arr is not None]
                        if len(len_list) == 0:
                            st.warning("ALE 결과 배열이 비어있습니다.")
                        else:
                            min_len = min(len_list)

                            # 잘라내기(혹은 변환)
                            centers_s = np.array(centers)[:min_len]
                            mean_s = np.array(mean_ale)[:min_len]
                            lower_s = np.array(lower_ale)[:min_len]
                            upper_s = np.array(upper_ale)[:min_len]

                            # -----------------------------------------------
                            # 2) NaN 있는 인덱스 제거
                            # -----------------------------------------------
                            valid_mask = (~np.isnan(centers_s)) & (~np.isnan(mean_s)) & (~np.isnan(lower_s)) & (
                                ~np.isnan(upper_s))
                            valid_idx = np.where(valid_mask)[0]

                            if valid_idx.size < 2:
                                st.warning("유효한 ALE 포인트가 충분하지 않습니다 (2 미만). 더 많은 데이터 또는 bin 수를 변경하세요.")
                            else:
                                # 안전한 plot 데이터
                                centers_plot = centers_s[valid_idx]
                                mean_plot = mean_s[valid_idx]
                                lower_plot = lower_s[valid_idx]
                                upper_plot = upper_s[valid_idx]

                                # -----------------------------------------------
                                # 3) all_vals 정리: (B x k) -> (B x valid_points)
                                # -----------------------------------------------
                                all_vals_arr = np.array(all_vals)  # 다차원일 수 있음
                                # 보장: all_vals_arr.shape == (B_eff, k_eff) 혹은 (B_eff,) 등
                                if all_vals_arr.ndim == 1:
                                    # 만약 (B,)인 경우 B=1 혹은 잘못된 shape; reshape 시도
                                    all_vals_arr = all_vals_arr.reshape((all_vals_arr.shape[0], 1))
                                B_eff = all_vals_arr.shape[0]
                                k_eff = all_vals_arr.shape[1]

                                # pad or truncate to min_len
                                if k_eff < min_len:
                                    pad_width = min_len - k_eff
                                    all_vals_arr = np.concatenate([all_vals_arr, np.full((B_eff, pad_width), np.nan)],
                                                                  axis=1)
                                # truncate then select valid indices
                                all_vals_trunc = all_vals_arr[:, :min_len][:, valid_idx]  # shape (B_eff, valid_points)
                                std_boot = np.nanstd(all_vals_trunc, axis=0)

                                # -----------------------------------------------
                                # 4) 그리기 (길이/NaN 문제 해결 후)
                                # -----------------------------------------------
                                fig_ci, ax_ci = plt.subplots(figsize=(6, 3))
                                ax_ci.plot(centers_plot, mean_plot, marker='o', label='ALE mean')
                                ax_ci.fill_between(centers_plot, lower_plot, upper_plot, alpha=0.3,
                                                   label=f"{alpha}% CI")
                                ax_ci.set_title(f"ALE with {alpha}% bootstrap CI: {ci_feature}")
                                ax_ci.set_xlabel(ci_feature)
                                ax_ci.set_ylabel("ALE")
                                ax_ci.legend()
                                st.pyplot(fig_ci)
                                plt.close(fig_ci)

                                # -----------------------------------------------
                                # 5) 요약표 출력 (centers_plot 길이에 맞춤)
                                # -----------------------------------------------
                                summary_df = pd.DataFrame({
                                    "center": centers_plot,
                                    "ale_mean": mean_plot,
                                    "ale_lower": lower_plot,
                                    "ale_upper": upper_plot,
                                    "ale_std": std_boot
                                })
                                st.markdown("부트스트랩 결과 요약 (유효 포인트):")
                                st.dataframe(summary_df)


                                # -----------------------------------------------
                                # 6) 자동 해석 리포트 생성 (안전하게 centers_plot, mean_plot 사용)
                                # -----------------------------------------------
                                def interpret_ale_result(feature_name, centers, mean_vals):
                                    """
                                    feature_name: 문자열
                                    centers: X축 값 (1D array)
                                    mean_vals: ALE 평균값 (1D array, same length)
                                    반환: (trend, domain_text)
                                    """
                                    # 기본 방어 코드
                                    centers = np.asarray(centers)
                                    mean_vals = np.asarray(mean_vals)
                                    if centers.size < 2 or mean_vals.size < 2:
                                        return "데이터 부족으로 패턴을 해석할 수 없습니다.", ""

                                    # 변화율(기울기) 계산 (인접 구간 차분)
                                    diffs = np.diff(mean_vals)
                                    slope_mean = np.nanmean(diffs)

                                    # 기울기 해석 (문구 숫자 임계값은 필요시 튜닝)
                                    if slope_mean > 0.5:
                                        trend = "강한 양(+)의 영향 — 값이 증가할수록 수확량이 뚜렷하게 증가하는 패턴"
                                    elif slope_mean > 0.1:
                                        trend = "약한 양(+)의 영향 — 값이 증가하면 수확량이 완만하게 증가"
                                    elif slope_mean < -0.5:
                                        trend = "강한 음(-)의 영향 — 값이 증가할수록 수확량이 뚜렷하게 감소하는 패턴"
                                    elif slope_mean < -0.1:
                                        trend = "약한 음(-)의 영향 — 값이 증가하면 수확량이 완만하게 감소"
                                    else:
                                        trend = "유의미한 변화 없음 — 증가/감소 경향이 약함"

                                    # 스마트팜 도메인 관점 보조 해석 (문구)
                                    fname = feature_name.lower()
                                    domain_text = ""
                                    if "co" in fname or "co2" in fname:
                                        domain_text = ("CO₂ 관련: CO₂ 농도 증가는 일반적으로 광합성·생장 촉진을 통해 수확량을 "
                                                       "증가시킬 수 있지만, 과도한 농도에서는 오히려 부정적 영향을 줄 수 있습니다.")
                                    elif "온도" in feature_name or "temp" in fname:
                                        domain_text = ("온도 관련: 적정 범위 내 온도 증가는 생장 촉진을 유도하지만, 과온 시 생리적 "
                                                       "스트레스가 발생하여 수확량을 감소시킬 수 있습니다.")
                                    elif "습" in feature_name or "hum" in fname:
                                        domain_text = ("습도 관련: 적절한 습도는 유리하나 과습은 병해를 유발할 수 있고, 너무 낮으면 "
                                                       "기공 닫힘으로 광합성이 억제될 수 있습니다.")
                                    elif "일사" in feature_name or "solar" in fname or "irradi" in fname:
                                        domain_text = ("일사량 관련: 일사량 증가는 광합성 증가로 일반적으로 수확량을 올리지만, "
                                                       "품종과 상황에 따라 과다 시 스트레스가 생길 수 있습니다.")
                                    else:
                                        domain_text = "일반적: 이 Feature는 모델 예측에 영향이 있습니다."

                                    return trend, domain_text


                                # 실제 해석 호출
                                trend_text, domain_text = interpret_ale_result(ci_feature, centers_plot, mean_plot)

                                st.markdown("### 🔎 ALE 패턴 자동 해석")
                                st.write(f"**Feature:** {ci_feature}")
                                st.write(f"**전반적 경향:** {trend_text}")
                                if domain_text:
                                    st.write(f"**스마트팜 관점 해석:** {domain_text}")

                                # ------------------------
                                # 정량적 패턴 출력
                                # ------------------------
                                delta = mean_plot[-1] - mean_plot[0]
                                st.markdown("### 📈 정량 요약")
                                st.write(f"- 분석 구간 전체 ALE 변화량: **{delta:.3f}**")
                                st.write(f"- 평균 구간 기울기: **{np.nanmean(np.diff(mean_plot)):.4f}**")
                                st.write(
                                    f"- 최대 양(+) 영향 구간: center={centers_plot[np.argmax(mean_plot)]:.2f}, ALE={np.max(mean_plot):.3f}")
                                st.write(
                                    f"- 최대 음(-) 영향 구간: center={centers_plot[np.argmin(mean_plot)]:.2f}, ALE={np.min(mean_plot):.3f}")

                except Exception as e:
                    st.error(f"ALE 부트스트랩 오류: {e}")

