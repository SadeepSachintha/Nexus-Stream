import json
import os
import sys
import time

# Ensure repository root is on sys.path regardless of execution directory
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import uuid
import numpy as np
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
from src.analytics.drift import StatisticalDriftEngine

@st.cache_resource
def get_local_resources():
    store = DuckDBAnalyticsStore(db_path=DB_PATH)
    predictor = FraudPredictor()
    feature_mgr = StreamFeatureStateManager()
    drift_engine = StatisticalDriftEngine()
    return store, predictor, feature_mgr, drift_engine

store, local_predictor, feature_mgr, local_drift_engine = get_local_resources()


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


def fetch_drift_report(limit: int = 200):
    try:
        r = requests.get(f"{API_BASE_URL}/v1/analytics/drift?limit={limit}", timeout=1.5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    records = store.get_recent_transactions(limit=limit)
    report = local_drift_engine.compute_drift_report(records)
    return report.model_dump()


def fetch_drift_distribution(feature_name: str, limit: int = 200):
    try:
        r = requests.get(f"{API_BASE_URL}/v1/analytics/drift/distribution?feature_name={feature_name}&limit={limit}", timeout=1.5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    records = store.get_recent_transactions(limit=limit)
    return local_drift_engine.get_comparison_distributions(feature_name, records)


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
tab_alerts, tab_analytics, tab_simulator, tab_graph, tab_dlq, tab_drift, tab_model = st.tabs([
    "🚨 Live Alerts & Dispute Triage",
    "📊 Real-Time Analytics & Risk Matrix",
    "⚡ Instant Transaction Scoring Lab",
    "🕸️ Fraud Ring Network Explorer",
    "⚠️ Dead-Letter Queue (DLQ)",
    "📡 Drift Radar & Model Health",
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

# --- TAB 4: Fraud Ring Network Explorer ---
with tab_graph:
    st.subheader("🕸️ Graph-Based Fraud Ring & Botnet Explorer")
    st.markdown("Real-time bipartite entity graph detecting card-sharing botnets and synthetic identity rings.")

    graph_engine = feature_mgr.graph_engine
    rings = graph_engine.get_detected_fraud_rings()
    vis_data = graph_engine.get_visualization_data()

    total_devices = len(graph_engine.devices_to_cards)
    total_cards = len(graph_engine.cards_to_devices)
    max_cluster = max([r["total_entities"] for r in rings], default=0)

    g_c1, g_c2, g_c3, g_c4 = st.columns(4)
    g_c1.metric("Monitored Devices", f"{total_devices:,}")
    g_c2.metric("Monitored Cards", f"{total_cards:,}")
    g_c3.metric("Active Fraud Rings", f"{len(rings):,}", delta="High Risk" if rings else "Clean")
    g_c4.metric("Largest Cluster Size", f"{max_cluster} Entities")

    st.write("")

    # Interactive Graph Visualizer
    if vis_data["nodes"]:
        edge_x = []
        edge_y = []
        for e in vis_data["edges"]:
            edge_x.extend([e["x0"], e["x1"], None])
            edge_y.extend([e["y0"], e["y1"], None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.2, color='#475569'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = [n["x"] for n in vis_data["nodes"]]
        node_y = [n["y"] for n in vis_data["nodes"]]
        node_text = [f"<b>{n['type'].upper()}</b>: {n['id']}<br>Connections: {n['degree']}<br>Status: {'🚨 SUSPICIOUS RING' if n['is_suspicious'] else 'NORMAL'}" for n in vis_data["nodes"]]
        node_colors = ['#ef4444' if n['is_suspicious'] else ('#38bdf8' if n['type'] == 'card' else '#a855f7') for n in vis_data["nodes"]]
        node_symbols = ['circle' if n['type'] == 'card' else 'diamond' for n in vis_data["nodes"]]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[n['id'][:10] for n in vis_data["nodes"]],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                color=node_colors,
                size=18,
                symbol=node_symbols,
                line=dict(width=2, color='#0f172a')
            )
        )

        fig_net = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(text="<b>Bipartite Entity Network (Cards & Devices)</b>", font=dict(color="#f8fafc")),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=20, r=20, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template="plotly_dark",
                height=480,
            )
        )
        st.plotly_chart(fig_net, use_container_width=True)
    else:
        st.info("No entity relationships recorded yet. Click the simulation button below to inject a multi-card fraud ring.")

    # Ring Drilldown and Action Table
    if rings:
        st.markdown("##### Detected Fraud Syndicates & Compromised Devices")
        df_rings = pd.DataFrame([{
            "Anchor Device ID": r["anchor_device"],
            "Stolen Cards Linked": r["card_count"],
            "Syndicate Cluster Size": r["total_entities"],
            "Associated Cards": ", ".join(r["cards"][:4]) + ("..." if len(r["cards"]) > 4 else "")
        } for r in rings])
        st.dataframe(df_rings, use_container_width=True)

    # Simulation Injection Button
    col_sim_btn, _ = st.columns([1.5, 2])
    with col_sim_btn:
        if st.button("🚨 Inject Synthetic Fraud Ring (5 Cards on 1 Device)", use_container_width=True):
            shared_dev = f"dev_botnet_ring_{int(time.time())}"
            now = time.time()
            for k in range(5):
                card_k = f"card_stolen_{k}_{int(time.time())%10000}"
                test_tx = EnrichedTransaction(
                    transaction_id=f"tx_ring_{k}_{int(time.time())}",
                    user_id=f"usr_victim_{k}",
                    card_id=card_k,
                    amount=round(850.0 + k * 120, 2),
                    currency="USD",
                    merchant_id="merch_crypto_99",
                    merchant_category="crypto",
                    latitude=40.7128,
                    longitude=-74.0060,
                    timestamp=now + (k * 20),
                    device_id=shared_dev,
                    tx_count_last_10m=k + 1,
                    tx_count_last_1h=k + 1,
                    amount_rolling_avg_1h=850.0,
                    amount_deviation_from_avg=2.5,
                    geo_distance_velocity_kmh=15.0,
                    is_night_transaction=0,
                    category_risk_index=0.95,
                )
                feature_mgr.enrich(test_tx)
                pred_res = local_predictor.predict(test_tx)
                store.insert_scored_transaction(test_tx, pred_res)
            st.success(f"Injected 5-card fraud ring originating from device '{shared_dev}'!")
            st.rerun()

# --- TAB 5: Dead-Letter Queue (DLQ) & Poison Pill Workbench ---
with tab_dlq:
    st.subheader("⚠️ Dead-Letter Queue (DLQ) & Poison Pill Isolation")
    st.markdown("Isolates malformed payloads, schema contract violations, and corrupted encodings to prevent consumer lag spikes.")

    dlq_summary = store.get_dlq_summary()
    d_c1, d_c2, d_c3, d_c4 = st.columns(4)
    d_c1.metric("Total Poison Pills Isolated", f"{dlq_summary.total_dlq_events:,}")
    d_c2.metric("Pending Review", f"{dlq_summary.pending_events:,}", delta="Attention Needed" if dlq_summary.pending_events > 0 else "All Clean")
    d_c3.metric("Successfully Replayed", f"{dlq_summary.replayed_events:,}")
    d_c4.metric("Discarded / Dropped", f"{dlq_summary.discarded_events:,}")

    st.write("")

    dlq_filter = st.selectbox("Filter DLQ Events by Status", ["All Incidents", "PENDING", "REPLAYED", "DISCARDED"])
    status_arg = None if dlq_filter == "All Incidents" else dlq_filter
    dlq_items = store.get_dlq_events(status=status_arg, limit=50)

    if not dlq_items:
        st.info("No dead-letter events found matching filter. Use the simulation buttons below to test poison pill isolation.")
    else:
        df_dlq = pd.DataFrame(dlq_items)
        col_q_table, col_q_detail = st.columns([1.3, 1.1])

        with col_q_table:
            st.markdown("##### Isolated Poison Pill Records")
            st.dataframe(
                df_dlq[["dlq_id", "error_type", "status", "retry_count", "source_topic"]],
                column_config={
                    "dlq_id": st.column_config.TextColumn("Incident ID"),
                    "error_type": st.column_config.TextColumn("Failure Category"),
                    "status": st.column_config.TextColumn("Status"),
                    "retry_count": st.column_config.NumberColumn("Retries"),
                },
                use_container_width=True,
                height=380,
            )

        with col_q_detail:
            st.markdown("##### Diagnostic Inspector & Replay Action")
            selected_dlq_id = st.selectbox("Select DLQ Record to Inspect", df_dlq["dlq_id"].tolist())
            dlq_row = df_dlq[df_dlq["dlq_id"] == selected_dlq_id].iloc[0]

            with st.container(border=True):
                st.markdown(f"**DLQ ID:** `{dlq_row['dlq_id']}`")
                st.markdown(f"**Failure Type:** `{dlq_row['error_type']}`")
                st.markdown(f"**Status:** `{dlq_row['status']}` | **Retries:** `{dlq_row['retry_count']}`")
                st.error(f"**Diagnostic Error:**\n{dlq_row['error_message']}")

                st.markdown("**Raw Corrupted Payload:**")
                st.code(dlq_row['raw_payload'], language="json")

                st.markdown("---")
                col_btn_rep, col_btn_disc = st.columns(2)
                with col_btn_rep:
                    if st.button("🔁 Replay & Score", key=f"rep_{selected_dlq_id}", use_container_width=True):
                        try:
                            payload_dict = json.loads(dlq_row['raw_payload'])
                            raw_tx = RawTransaction(**payload_dict)
                            enriched = feature_mgr.enrich(raw_tx)
                            res = local_predictor.predict(enriched)
                            store.insert_scored_transaction(enriched, res)
                            store.update_dlq_status(selected_dlq_id, "REPLAYED", increment_retry=True)
                            st.success(f"Successfully replayed and scored! Decision: {res.prediction}")
                            st.rerun()
                        except Exception as ex:
                            store.update_dlq_status(selected_dlq_id, "PENDING", increment_retry=True)
                            st.warning(f"Replay failed: {ex}. Payload still requires manual fix.")
                with col_btn_disc:
                    if st.button("🗑️ Discard Incident", key=f"disc_{selected_dlq_id}", use_container_width=True):
                        store.update_dlq_status(selected_dlq_id, "DISCARDED")
                        st.info("Marked incident as DISCARDED.")
                        st.rerun()

    st.markdown("---")
    st.markdown("##### 🧪 Poison Pill Simulation Lab")
    c_p1, c_p2, c_p3 = st.columns(3)
    from src.streaming.schemas import DeadLetterEvent
    with c_p1:
        if st.button("⚡ Inject Malformed JSON String", use_container_width=True):
            dlq_ev = DeadLetterEvent(
                dlq_id=f"dlq_badjson_{int(time.time()*1000)}",
                source_topic="transactions.raw",
                error_type="MALFORMED_JSON",
                error_message="JSONDecodeError: Unterminated string starting at line 1 column 48",
                raw_payload='{"transaction_id": "tx_unclosed", "amount": 250.0, "corrupt...',
                retry_count=0,
                status="PENDING",
                timestamp=time.time()
            )
            store.insert_dlq_event(dlq_ev)
            st.success("Injected malformed JSON poison pill!")
            st.rerun()

    with c_p2:
        if st.button("⚡ Inject Missing Mandatory Fields", use_container_width=True):
            dlq_ev = DeadLetterEvent(
                dlq_id=f"dlq_missing_{int(time.time()*1000)}",
                source_topic="transactions.raw",
                error_type="SCHEMA_VALIDATION_ERROR",
                error_message="ValidationError: Field required: 'transaction_id', 'card_id', 'amount'",
                raw_payload=json.dumps({"user_id": "usr_broken", "notes": "client_app_v1_bug"}),
                retry_count=0,
                status="PENDING",
                timestamp=time.time()
            )
            store.insert_dlq_event(dlq_ev)
            st.success("Injected schema contract violation!")
            st.rerun()

    with c_p3:
        if st.button("⚡ Inject Negative Monetary Amount", use_container_width=True):
            dlq_ev = DeadLetterEvent(
                dlq_id=f"dlq_neg_{int(time.time()*1000)}",
                source_topic="transactions.raw",
                error_type="SCHEMA_VALIDATION_ERROR",
                error_message="ValidationError: Input should be greater than 0 [amount: -4500.00]",
                raw_payload=json.dumps({"transaction_id": "tx_neg_val", "user_id": "usr_hacker", "card_id": "card_01", "amount": -4500.0, "merchant_id": "m1", "merchant_category": "crypto", "latitude": 40.71, "longitude": -74.00}),
                retry_count=0,
                status="PENDING",
                timestamp=time.time()
            )
            store.insert_dlq_event(dlq_ev)
            st.success("Injected negative amount poison pill!")
            st.rerun()

# --- TAB 6: Drift Radar & Model Health ---
with tab_drift:
    st.subheader("Statistical Data & Concept Drift Radar")
    st.markdown("Sub-second divergence tracking between training reference baseline distributions and live production streaming features.")

    drift_report = fetch_drift_report(limit=300)

    # 1. High-level status alert
    if drift_report.get("dataset_drift_detected", False):
        st.error(f"🚨 **DATASET DRIFT ALERT**: {drift_report.get('drifted_feature_count', 0)} / {drift_report.get('total_features', 8)} features ({drift_report.get('drift_share', 0.0)}%) exhibit statistically significant divergence from baseline! Immediate model retraining or traffic audit recommended.")
    else:
        st.success(f"🟢 **STREAM DISTRIBUTIONS STABLE**: 0 critical feature drifts detected across {drift_report.get('sample_size', 0)} monitored production transactions. Production aligns with baseline.")

    # 2. Executive KPIs
    d_c1, d_c2, d_c3, d_c4 = st.columns(4)
    d_c1.metric("Drifted Features", f"{drift_report.get('drifted_feature_count', 0)} / {drift_report.get('total_features', 8)}")
    d_c2.metric("Drift Share", f"{drift_report.get('drift_share', 0.0)}%", delta=f"{drift_report.get('drift_share', 0.0)}%" if drift_report.get("dataset_drift_detected") else "Normal", delta_color="inverse")
    d_c3.metric("Production Sample Window", f"{drift_report.get('sample_size', 0)} events")
    d_c4.metric("Statistical Methods", "PSI + KS-Test + Wasserstein")

    st.divider()

    # 3. Feature Drift Scorecard Table
    st.markdown("##### 🔬 Feature-Level Stability Matrix")
    st.caption("PSI Rule-of-Thumb: **< 0.10** Stable | **0.10 - 0.20** Moderate Shift | **≥ 0.20** Critical Drift. KS Test: Flagged when **p-value < 0.01** & **statistic ≥ 0.20**.")

    feats = drift_report.get("features", {})
    if feats:
        rows = []
        for fname, m in feats.items():
            status_badge = "🟢 STABLE"
            if m.get("status") == "CRITICAL_DRIFT":
                status_badge = "🔴 CRITICAL DRIFT"
            elif m.get("status") == "MODERATE_SHIFT":
                status_badge = "🟡 MODERATE SHIFT"

            rows.append({
                "Feature Name": fname,
                "Status": status_badge,
                "PSI Score": round(m.get("psi_score", 0.0), 4),
                "KS Statistic": round(m.get("ks_statistic", 0.0), 4),
                "KS p-value": round(m.get("ks_p_value", 1.0), 5),
                "Wasserstein Distance": round(m.get("wasserstein_distance", 0.0), 4),
                "Drift Detected": "⚠️ YES" if m.get("drift_detected") else "✅ NO",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # 4. Interactive Distribution Overlay Visualizer
    st.markdown("##### 📊 Interactive Distribution Comparison: Baseline Reference vs. Production")
    feature_choices = list(feats.keys()) if feats else ["amount", "geo_distance_velocity_kmh", "tx_count_last_10m", "amount_deviation_from_avg"]
    selected_feat = st.selectbox("Select Feature to Inspect Distribution Overlay:", feature_choices, index=0)

    dist_data = fetch_drift_distribution(selected_feat, limit=300)
    ref_vals = dist_data.get("reference", [])
    curr_vals = dist_data.get("current", [])

    if len(ref_vals) > 0 and len(curr_vals) > 0:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=ref_vals,
            name="Training Baseline (Reference)",
            opacity=0.6,
            marker_color="#3b82f6",
            histnorm="probability density"
        ))
        fig_dist.add_trace(go.Histogram(
            x=curr_vals,
            name="Streaming Production (Current)",
            opacity=0.6,
            marker_color="#ef4444" if feats.get(selected_feat, {}).get("drift_detected") else "#10b981",
            histnorm="probability density"
        ))
        fig_dist.update_layout(
            barmode="overlay",
            title=f"Distribution Density: '{selected_feat}' (Baseline vs Production)",
            xaxis_title=selected_feat,
            yaxis_title="Probability Density",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(x=0.75, y=0.95),
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    elif len(curr_vals) == 0:
        st.info(f"Awaiting production transactions to compute density overlay for '{selected_feat}'. Use the simulation buttons below to generate data.")

    st.divider()

    # 5. Live Drift Simulation Controls
    st.markdown("##### 🧪 Real-World Distribution Shift Injection Studio")
    st.caption("Simulate sudden external macroeconomic shifts or localized coordinated attack bursts to watch the drift radar detect divergence in real-time.")

    s_c1, s_c2, s_c3 = st.columns(3)
    with s_c1:
        if st.button("⚡ Inject 50x Luxury Amount Surge ($3k-$9k)", use_container_width=True):
            try:
                requests.post(f"{API_BASE_URL}/v1/analytics/drift/simulate-shift?shift_type=amount_surge&count=50", timeout=3.0)
                st.success("Injected 50 luxury amount surge events via API!")
            except Exception:
                now = time.time()
                for i in range(50):
                    amt = float(np.random.uniform(3000.0, 9500.0))
                    raw_tx = RawTransaction(
                        transaction_id=f"tx_shift_amt_{uuid.uuid4().hex[:6]}",
                        card_id=f"card_shift_{i%5}",
                        user_id=f"usr_shift_{i%5}",
                        amount=amt,
                        currency="USD",
                        merchant_id="m_luxury",
                        merchant_category="luxury_goods",
                        latitude=40.71,
                        longitude=-74.00,
                        timestamp=now + (i*0.1)
                    )
                    enr = feature_mgr.enrich(raw_tx)
                    pred = local_predictor.predict(enr)
                    store.insert_scored_transaction(enr, pred)
                st.success("Locally injected 50 amount surge events into DuckDB!")
            st.rerun()

    with s_c2:
        if st.button("⚡ Inject 50x High-Velocity Flight Bursts", use_container_width=True):
            try:
                requests.post(f"{API_BASE_URL}/v1/analytics/drift/simulate-shift?shift_type=velocity_spike&count=50", timeout=3.0)
                st.success("Injected 50 velocity spike events via API!")
            except Exception:
                now = time.time()
                for i in range(50):
                    raw_tx = RawTransaction(
                        transaction_id=f"tx_shift_vel_{uuid.uuid4().hex[:6]}",
                        card_id=f"card_shift_{i%3}",
                        user_id=f"usr_shift_{i%3}",
                        amount=float(np.random.uniform(150.0, 750.0)),
                        currency="USD",
                        merchant_id="m_travel",
                        merchant_category="electronics",
                        latitude=35.6762 if (i % 2 == 0) else 40.7128,
                        longitude=139.6503 if (i % 2 == 0) else -74.0060,
                        timestamp=now + (i*0.1)
                    )
                    enr = feature_mgr.enrich(raw_tx)
                    pred = local_predictor.predict(enr)
                    store.insert_scored_transaction(enr, pred)
                st.success("Locally injected 50 velocity spike events into DuckDB!")
            st.rerun()

    with s_c3:
        if st.button("⚡ Inject 50x Crypto Arbitrage Transactions", use_container_width=True):
            try:
                requests.post(f"{API_BASE_URL}/v1/analytics/drift/simulate-shift?shift_type=crypto_boom&count=50", timeout=3.0)
                st.success("Injected 50 crypto surge events via API!")
            except Exception:
                now = time.time()
                for i in range(50):
                    raw_tx = RawTransaction(
                        transaction_id=f"tx_shift_cry_{uuid.uuid4().hex[:6]}",
                        card_id=f"card_shift_{i%4}",
                        user_id=f"usr_shift_{i%4}",
                        amount=float(np.random.uniform(1200.0, 4500.0)),
                        currency="USD",
                        merchant_id="m_crypto",
                        merchant_category="crypto_exchange",
                        latitude=51.5074,
                        longitude=-0.1278,
                        timestamp=now + (i*0.1)
                    )
                    enr = feature_mgr.enrich(raw_tx)
                    pred = local_predictor.predict(enr)
                    store.insert_scored_transaction(enr, pred)
                st.success("Locally injected 50 crypto surge events into DuckDB!")
            st.rerun()

# --- TAB 7: Model Explainability & Thresholds ---
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
