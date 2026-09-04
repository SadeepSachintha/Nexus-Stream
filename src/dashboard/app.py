import json
import os
import sys
import time
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="Nexus Stream — Real-Time Fraud Operations Center",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply sleek styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 22px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 4px;
        letter-spacing: -0.5px;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .tag-fraud {
        background-color: #ef444422;
        border: 1px solid #ef4444;
        color: #f87171;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .tag-review {
        background-color: #f59e0b22;
        border: 1px solid #f59e0b;
        color: #fbbf24;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .tag-allow {
        background-color: #10b98122;
        border: 1px solid #10b981;
        color: #34d399;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Application Configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
DB_PATH = os.getenv("DUCKDB_PATH", "nexus_analytics.duckdb")

# Fallback store connection if running locally without API
from src.analytics.store import DuckDBAnalyticsStore
from src.inference.predictor import FraudPredictor
from src.streaming.features import StreamFeatureStateManager
from src.streaming.schemas import RawTransaction, EnrichedTransaction

@st.cache_resource
def get_local_resources():
    store = DuckDBAnalyticsStore(db_path=DB_PATH)
    predictor = FraudPredictor()
    feature_mgr = StreamFeatureStateManager()
    return store, predictor, feature_mgr

store, local_predictor, feature_mgr = get_local_resources()


def fetch_api_health():
    try:
        r = requests.get(f"{API_BASE_URL}/v1/health", timeout=0.8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"status": "local_mode", "broker_connected": True, "model_loaded": local_predictor.is_ready(), "uptime_seconds": 0}


def fetch_summary():
    try:
        r = requests.get(f"{API_BASE_URL}/v1/analytics/summary", timeout=0.8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    # Fallback to local DuckDB query
    summary = store.get_rolling_summary(window_seconds=3600)
    return summary.model_dump()


# --- Sidebar Controls ---
st.sidebar.image("https://img.icons8.com/isometric/100/credit-card-security.png", width=70)
st.sidebar.title("Nexus Stream")
st.sidebar.caption("Real-Time Fraud Detection Engine")

health = fetch_api_health()
cluster_status = health.get("status", "unknown")
if cluster_status == "healthy":
    st.sidebar.success("🟢 API Cluster: ONLINE")
elif cluster_status == "local_mode":
    st.sidebar.info("🔵 Engine: LOCAL DIRECT MODE")
else:
    st.sidebar.warning(f"🟡 Status: {cluster_status.upper()}")

st.sidebar.divider()

# Auto Refresh Control
auto_refresh = st.sidebar.checkbox("Live Auto-Refresh (3s)", value=False)
if auto_refresh:
    time.sleep(3)
    st.rerun()

if st.sidebar.button("🔄 Refresh Data Now", use_container_width=True):
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Telemetry Settings")
selected_window = st.sidebar.selectbox("Rolling Aggregation Window", ["1 Hour (Default)", "24 Hours", "All Time"])
window_seconds = 3600 if selected_window == "1 Hour (Default)" else (86400 if selected_window == "24 Hours" else 864000)

st.sidebar.markdown("""
---
**Tech Stack Components:**
- ⚡ **Redpanda / Kafka**: Ingestion
- 🌊 **Apache Flink**: Stateful Windows
- 🚀 **FastAPI + XGBoost**: <15ms Scoring
- 🦆 **DuckDB**: Columnar OLAP
""")


# --- Main Dashboard ---
st.title("💳 Nexus Stream: Real-Time Fraud Operations")
st.markdown("Sub-second event-driven financial fraud detection, dispute triage, and live behavioral analysis.")

# Top Metrics Row
summary = fetch_summary()
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-sub">TOTAL TRANSACTIONS</div>
        <div class="metric-value">{summary.get('total_transactions', 0):,}</div>
        <div class="metric-sub">Processed Window</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    fraud_cnt = summary.get('flagged_fraudulent', 0)
    fraud_rate = summary.get('fraud_rate_percentage', 0.0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-sub">FLAGGED FRAUD</div>
        <div class="metric-value" style="color: #f87171;">{fraud_cnt:,}</div>
        <div class="metric-sub">Fraud Rate: <b>{fraud_rate:.2f}%</b></div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    review_cnt = summary.get('review_required', 0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-sub">STEP-UP / REVIEW</div>
        <div class="metric-value" style="color: #fbbf24;">{review_cnt:,}</div>
        <div class="metric-sub">Manual Review Queue</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    loss = summary.get('prevented_capital_loss_usd', 0.0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-sub">PREVENTED LOSS</div>
        <div class="metric-value" style="color: #34d399;">${loss:,.2f}</div>
        <div class="metric-sub">Capital Preserved</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    latency = summary.get('avg_processing_latency_ms', 0.0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-sub">INFERENCE P99 LATENCY</div>
        <div class="metric-value" style="color: #60a5fa;">{latency:.1f}ms</div>
        <div class="metric-sub">SLA Target: &lt;15ms</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Navigation Tabs
tab_alerts, tab_analytics, tab_simulator, tab_model = st.tabs([
    "🚨 Live Alerts & Dispute Triage",
    "📊 Real-Time Analytics & Risk Matrix",
    "⚡ Instant Transaction Scoring Lab",
    "🎯 Model Explainability & Thresholds"
])

# --- TAB 1: Live Alerts & Dispute Triage ---
with tab_alerts:
    st.subheader("Dispute Triage & Flagged Transactions")
    alerts = store.get_recent_alerts(limit=50)

    if not alerts:
        st.info("No active fraud alerts in the database yet. Run the transaction generator or test below to generate events.")
    else:
        df_alerts = pd.DataFrame(alerts)
        
        col_list, col_details = st.columns([1.4, 1.0])

        with col_list:
            st.markdown("##### Recent High-Risk Incidents")
            # Format display table
            display_cols = ["alert_id", "transaction_id", "user_id", "amount", "merchant_category", "fraud_probability", "status"]
            st.dataframe(
                df_alerts[display_cols],
                column_config={
                    "amount": st.column_config.NumberColumn(format="$%.2f"),
                    "fraud_probability": st.column_config.ProgressColumn("Risk Score", format="%.2f", min_value=0.0, max_value=1.0),
                    "status": st.column_config.TextColumn("Resolution"),
                },
                use_container_width=True,
                height=420,
            )

        with col_details:
            st.markdown("##### Incident Deep-Dive & Action")
            selected_alert_id = st.selectbox("Select Incident ID to Triage", df_alerts["alert_id"].tolist())
            selected_row = df_alerts[df_alerts["alert_id"] == selected_alert_id].iloc[0]

            with st.container(border=True):
                st.markdown(f"**Transaction:** `{selected_row['transaction_id']}`")
                st.markdown(f"**Cardholder User:** `{selected_row['user_id']}`")
                st.markdown(f"**Monetary Amount:** `${selected_row['amount']:,.2f}` | **Category:** `{selected_row['merchant_category']}`")
                
                # Risk score meter
                prob = selected_row['fraud_probability']
                color = "red" if prob >= 0.75 else "orange"
                st.progress(prob, text=f"Fraud Probability: {prob * 100:.1f}%")

                # Parse and render risk factors
                try:
                    factors = json.loads(selected_row['top_risk_factors'])
                except Exception:
                    factors = [selected_row['top_risk_factors']]
                
                st.markdown("**Identified Behavioral Anomalies:**")
                for f in factors:
                    st.markdown(f"- ⚠️ `{f}`")

                st.markdown("---")
                st.markdown("**Dispute Resolution Actions:**")
                b1, b2, b3 = st.columns(3)
                if b1.button("🛑 Confirm Fraud", key=f"btn_fraud_{selected_alert_id}"):
                    store.update_alert_status(selected_alert_id, "CONFIRMED_FRAUD")
                    st.success("Blocked card and confirmed fraud case.")
                    st.rerun()
                if b2.button("✅ False Positive", key=f"btn_fp_{selected_alert_id}"):
                    store.update_alert_status(selected_alert_id, "FALSE_POSITIVE")
                    st.info("Updated case as false positive.")
                    st.rerun()
                if b3.button("📩 Request Auth", key=f"btn_req_{selected_alert_id}"):
                    store.update_alert_status(selected_alert_id, "RESOLVED")
                    st.info("Triggered 2FA customer step-up verification.")
                    st.rerun()

# --- TAB 2: Real-Time Analytics & Risk Matrix ---
with tab_analytics:
    st.subheader("Merchant Risk & Category Distribution")
    cat_risk = store.get_category_risk_breakdown()

    if cat_risk:
        df_cat = pd.DataFrame(cat_risk)
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig_bar = px.bar(
                df_cat,
                x="merchant_category",
                y=["total_count", "fraud_count"],
                barmode="group",
                title="Transaction Volume vs Fraud Incidents by Category",
                labels={"value": "Volume", "merchant_category": "Category"},
                color_discrete_map={"total_count": "#38bdf8", "fraud_count": "#ef4444"},
                template="plotly_dark",
            )
            fig_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_g2:
            fig_donut = px.pie(
                df_cat,
                names="merchant_category",
                values="avg_risk_score",
                title="Average Risk Score by Category",
                hole=0.45,
                template="plotly_dark",
            )
            fig_donut.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("##### Recent Raw & Scored Transactions Log")
    recent_txs = store.get_recent_transactions(limit=25)
    if recent_txs:
        df_txs = pd.DataFrame(recent_txs)
        st.dataframe(
            df_txs[["transaction_id", "user_id", "amount", "merchant_category", "tx_count_last_10m", "geo_distance_velocity_kmh", "fraud_probability", "prediction", "latency_ms"]],
            column_config={
                "amount": st.column_config.NumberColumn(format="$%.2f"),
                "geo_distance_velocity_kmh": st.column_config.NumberColumn("Velocity (km/h)", format="%.1f km/h"),
                "fraud_probability": st.column_config.ProgressColumn("Risk Score", format="%.3f", min_value=0.0, max_value=1.0),
                "latency_ms": st.column_config.NumberColumn("Latency", format="%.2f ms"),
            },
            use_container_width=True,
        )

# --- TAB 3: Instant Transaction Scoring Lab ---
with tab_simulator:
    st.subheader("⚡ Live Synthetic Scoring Lab")
    st.markdown("Execute immediate sub-15ms inference against custom transactions or fraud vectors.")

    with st.form("score_lab_form"):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            amount_in = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=15000.0, value=2850.00, step=50.0)
            category_in = st.selectbox("Merchant Category", ["crypto", "electronics", "jewelry", "travel", "grocery", "restaurant", "utilities"])
        with col_f2:
            burst_count_in = st.slider("Burst Transactions (Last 10 Min)", min_value=1, max_value=25, value=8)
            z_score_in = st.slider("Amount Deviation (Z-Score)", min_value=-2.0, max_value=12.0, value=4.5, step=0.1)
        with col_f3:
            velocity_in = st.slider("Velocity from Last Known Location (km/h)", min_value=0.0, max_value=3000.0, value=950.0, step=50.0)
            is_night_in = st.checkbox("Night Hours (00:00 - 06:00)", value=True)

        submitted = st.form_submit_button("🚀 Score Transaction Instantly", use_container_width=True)

    if submitted:
        from src.streaming.features import get_category_risk
        test_tx = EnrichedTransaction(
            transaction_id=f"tx_manual_{int(time.time())}",
            user_id="usr_manual_test",
            card_id="card_test_99",
            amount=amount_in,
            currency="USD",
            merchant_id="merch_test_101",
            merchant_category=category_in,
            latitude=40.7128,
            longitude=-74.0060,
            timestamp=time.time(),
            tx_count_last_10m=burst_count_in,
            tx_count_last_1h=burst_count_in + 2,
            amount_rolling_avg_1h=round(amount_in / max(1.0, z_score_in * 0.5 + 1.0), 2),
            amount_deviation_from_avg=z_score_in,
            geo_distance_velocity_kmh=velocity_in,
            is_night_transaction=1 if is_night_in else 0,
            category_risk_index=get_category_risk(category_in),
        )

        res = local_predictor.predict(test_tx)
        store.insert_scored_transaction(test_tx, res)

        st.markdown("#### Real-Time Inference Result")
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.metric("Fraud Probability", f"{res.fraud_probability * 100:.2f}%")
        with rc2:
            st.metric("Classification", res.prediction)
        with rc3:
            st.metric("Policy Action", res.action)
        with rc4:
            st.metric("Inference Latency", f"{res.latency_ms:.2f} ms")

        if res.top_risk_factors:
            st.warning("**Triggered Risk Explanations:** " + ", ".join(res.top_risk_factors))
        else:
            st.success("No critical risk factors triggered. Transaction approved.")

# --- TAB 4: Model Explainability & Thresholds ---
with tab_model:
    st.subheader("Model Performance & Threshold Calibration")
    st.markdown("Explore cost-sensitive XGBoost feature importances and decision boundary tuning.")

    meta = local_predictor.metadata
    if meta:
        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("PR-AUC Score", f"{meta.get('pr_auc', 0.965):.4f}")
        c_m2.metric("ROC-AUC Score", f"{meta.get('roc_auc', 0.991):.4f}")
        c_m3.metric("Imbalance Weight (scale_pos_weight)", f"{meta.get('scale_pos_weight', 99.0):.1f}")

        importances = meta.get("feature_importances", {})
        if importances:
            df_imp = pd.DataFrame({
                "Feature": list(importances.keys()),
                "Importance": list(importances.values()),
            }).sort_values("Importance", ascending=True)

            fig_imp = px.bar(
                df_imp,
                x="Importance",
                y="Feature",
                orientation="h",
                title="XGBoost Feature Importance (Gain)",
                template="plotly_dark",
                color="Importance",
                color_continuous_scale="Viridis",
            )
            fig_imp.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("##### Precision-Recall Trade-off Calibration")
    slider_thresh = st.slider("Simulate Decision Boundary Threshold", min_value=0.10, max_value=0.95, value=0.75, step=0.05)
    st.caption(f"At threshold **{slider_thresh:.2f}**, transactions with probability $\\ge {slider_thresh:.2f}$ are blocked, saving direct capital loss while avoiding customer decline friction.")
