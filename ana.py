import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression

# 페이지 기본 설정
st.set_page_config(
    page_title="비트코인 가격 대시보드",
    page_icon="🪙",
    layout="wide"
)

# 데이터 로드 함수 (캐싱하여 속도 향상)
@st.cache_data
def load_data():
    # 구분자를 ';'로 지정하여 CSV 파일 읽기
    df = pd.read_csv('coin.csv', sep=';')
    
    # 시간 데이터를 datetime 형식으로 변환
    df['timeOpen'] = pd.to_datetime(df['timeOpen'])
    
    # 날짜 기준으로 오름차순 정렬
    df = df.sort_values('timeOpen').reset_index(drop=True)
    
    return df

# 메인 타이틀
st.title("🪙 비트코인(Bitcoin) 가격 분석 대시보드")
st.markdown("업로드된 데이터를 바탕으로 비트코인의 가격 추이와 거래량을 분석하고 내일의 가격을 예측합니다.")

try:
    df = load_data()
    
    # 사이드바: 날짜 필터링
    st.sidebar.header("검색 설정")
    
    min_date = df['timeOpen'].min().date()
    max_date = df['timeOpen'].max().date()
    
    start_date, end_date = st.sidebar.date_input(
        "조회할 기간을 선택하세요",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # 선택된 기간으로 데이터 필터링
    mask = (df['timeOpen'].dt.date >= start_date) & (df['timeOpen'].dt.date <= end_date)
    filtered_df = df.loc[mask]
    
    if filtered_df.empty:
        st.warning("선택하신 기간에 데이터가 없습니다.")
    else:
        # --- 1. 핵심 지표 (KPI) 섹션 ---
        st.subheader("📊 요약 지표")
        
        # 최신 데이터와 이전 데이터 가져오기
        latest_data = filtered_df.iloc[-1]
        
        if len(filtered_df) > 1:
            prev_data = filtered_df.iloc[-2]
            price_change = latest_data['close'] - prev_data['close']
            price_change_pct = (price_change / prev_data['close']) * 100
        else:
            price_change = 0
            price_change_pct = 0
            
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="최신 종가", 
                value=f"₩{latest_data['close']:,.0f}", 
                delta=f"{price_change:,.0f} ({price_change_pct:.2f}%)"
            )
        with col2:
            st.metric(label="최고가 (선택 기간)", value=f"₩{filtered_df['high'].max():,.0f}")
        with col3:
            st.metric(label="최저가 (선택 기간)", value=f"₩{filtered_df['low'].min():,.0f}")
        with col4:
            st.metric(label="최근 거래량", value=f"₩{latest_data['volume']:,.0f}")

        st.markdown("---")
        
        # --- 2. 차트 섹션 ---
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("📈 캔들스틱 차트")
            fig_candle = go.Figure(data=[go.Candlestick(
                x=filtered_df['timeOpen'],
                open=filtered_df['open'],
                high=filtered_df['high'],
                low=filtered_df['low'],
                close=filtered_df['close'],
                name='Bitcoin'
            )])
            fig_candle.update_layout(
                xaxis_rangeslider_visible=False,
                margin=dict(l=0, r=0, t=30, b=0),
                height=400
            )
            st.plotly_chart(fig_candle, use_container_width=True)
            
        with col_chart2:
            st.subheader("📉 종가 추이")
            fig_line = px.line(
                filtered_df, 
                x='timeOpen', 
                y='close',
                labels={'timeOpen': '날짜', 'close': '종가 (KRW)'}
            )
            fig_line.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
            st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("---")

        # 거래량 차트
        st.subheader("📊 거래량 추이")
        fig_volume = px.bar(
            filtered_df, 
            x='timeOpen', 
            y='volume',
            labels={'timeOpen': '날짜', 'volume': '거래량'}
        )
        fig_volume.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300)
        fig_volume.update_traces(marker_color='rgba(50, 171, 96, 0.6)')
        st.plotly_chart(fig_volume, use_container_width=True)

        st.markdown("---")

        # --- 3. AI 예측 섹션 (선형회귀) ---
        st.subheader("🤖 AI 내일 가격 예측 (선형회귀 모델)")
        st.markdown("과거의 모든 시가, 고가, 저가, 종가, 거래량 데이터를 학습하여, 선택하신 기간의 **마지막 날 기준 다음 날(내일)**의 가격이 오를지 내릴지 예측합니다.")

        # 전체 데이터를 복사하여 예측 타겟(다음 날 종가) 생성
        ml_df = df.copy()
        ml_df['next_close'] = ml_df['close'].shift(-1) # 한 칸 위로 당겨서 내일 종가를 매핑
        
        # 학습에 사용할 특성(Features)
        features = ['open', 'high', 'low', 'close', 'volume']
        
        # 마지막 행은 next_close가 NaN이므로 학습에서 제외
        train_df = ml_df.dropna(subset=['next_close'])
        
        X = train_df[features]
        y = train_df['next_close']
        
        # 선형회귀 모델 훈련
        model = LinearRegression()
        model.fit(X, y)
        
        # 예측 진행: 현재 화면에 필터링된 데이터 중 가장 최신(마지막) 데이터 기준
        latest_features_df = filtered_df.iloc[-1][features].to_frame().T
        predicted_price = model.predict(latest_features_df)[0]
        
        current_price = latest_data['close']
        pred_diff = predicted_price - current_price
        pred_diff_pct = (pred_diff / current_price) * 100
        
        if pred_diff > 0:
            pred_text = "상승 📈"
            st.success(f"분석 결과, 다음 날 비트코인 가격은 현재 대비 **{pred_text}**할 것으로 예측됩니다.")
        else:
            pred_text = "하락 📉"
            st.error(f"분석 결과, 다음 날 비트코인 가격은 현재 대비 **{pred_text}**할 것으로 예측됩니다.")
            
        col_pred1, col_pred2, col_pred3 = st.columns(3)
        with col_pred1:
            st.metric(label="현재(기준) 종가", value=f"₩{current_price:,.0f}")
        with col_pred2:
            st.metric(label="내일 예측 종가", value=f"₩{predicted_price:,.0f}", delta=f"{pred_diff:,.0f} ({pred_diff_pct:.2f}%)")
        
        st.markdown("---")

        # --- 4. 원본 데이터 섹션 ---
        with st.expander("데이터 테이블 보기"):
            display_df = filtered_df[['timeOpen', 'open', 'high', 'low', 'close', 'volume']].copy()
            display_df['timeOpen'] = display_df['timeOpen'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(display_df, use_container_width=True)

except FileNotFoundError:
    st.error("⚠️ 'coin.csv' 파일을 찾을 수 없습니다. 파이썬 스크립트와 같은 폴더에 데이터 파일이 있는지 확인해주세요.")
except Exception as e:
    st.error(f"⚠️ 오류가 발생했습니다: {e}")
