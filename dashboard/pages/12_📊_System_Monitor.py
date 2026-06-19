"""
System Monitor - Health, Resources, and Logs
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_client import AgenticAPIClient
from utils.data_controls import render_data_controls
from components import render_metric_card, render_status_card, create_gauge_chart

st.set_page_config(page_title="System Monitor - Agentic AI", page_icon="📊", layout="wide")

@st.cache_resource
def get_api():
    return AgenticAPIClient()

api = get_api()

st.title("📊 System Monitor")
st.caption("Real-time system health, resource utilization, and monitoring")

with st.expander("ℹ️ System Monitoring Guide", expanded=False):
    st.markdown("""
    **What System Monitor Tracks**

    This page provides real-time visibility into infrastructure health, CPU/memory/disk usage,
    and the status of all external services powering the Agentic AI Marketing Platform.

    **Components Monitored**
    | Component | Purpose |
    |-----------|---------|
    | API Server | FastAPI backend serving all platform requests |
    | PostgreSQL | Primary database with pgvector for embeddings |
    | Redis | Semantic caching layer for LLM responses |
    | Ollama LLM | Local large language model inference service |
    | Prometheus | Time-series metrics collection (scrape every 15s) |
    | Grafana | Dashboard visualization for metrics & alerts |

    **Container Metrics** — Docker container resource consumption is tracked via cAdvisor,
    showing per-container CPU, memory, and network I/O.

    **Resource Trends** — Historical CPU and memory data is pulled from Prometheus for
    capacity planning. Use the time range selector (1h–7d) to spot usage patterns.

    **Alert Thresholds**
    - 🟡 **Warning:** CPU >80%, Memory >75%
    - 🔴 **Critical:** CPU >90%, Memory >90%, Disk <10% free

    **External Services** — Links to Grafana (visualization at `:3000`), Prometheus
    (metrics at `:9090`), and MLflow (experiment tracking).

    **How to Use This Page**
    1. Check system health daily — all indicators should be green ✅
    2. Investigate any red ❌ indicators immediately
    3. Monitor resource trends for capacity planning — look for sustained upward trends
    4. Review container metrics to identify resource-hungry services

    **When to Worry**
    - Sustained high CPU (>80%) or memory (>75%) over 30+ minutes
    - Database latency consistently >100ms (ideal is <10ms)
    - Redis cache hit rate dropping below 50% (target is >80%)
    - Any component showing ❌ disconnected status
    """)

tab1, tab2, tab3, tab4 = st.tabs([
    "🏥 System Health",
    "📊 Resource Utilization",
    "🔗 External Services",
    "📈 Metrics & Monitoring"
])

with tab1:
    st.subheader("System Health Dashboard")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.caption("Real-time health status of all system components — green means healthy, red requires attention")
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True, help="Re-fetch health status from all components. Triggers live checks against database, Redis, and Ollama."):
            st.rerun()
    
    try:
        health = api.get_health()
        
        if health.get("status") == "healthy":
            st.success("✅ All systems operational")
        else:
            st.error("⚠️ System degraded - review details below")
        
        detailed_health = api.get_detailed_health()
        
        if detailed_health:
            st.markdown("---")
            st.markdown("### Component Status")
            
            components = detailed_health.get("components", {})
            
            if components:
                cols = st.columns(min(len(components), 4))
                
                for idx, (component_name, component_status) in enumerate(components.items()):
                    col_idx = idx % 4
                    
                    with cols[col_idx]:
                        status = component_status.get("status", "unknown")
                        
                        if status == "healthy":
                            st.success(f"✅ **{component_name.upper()}**")
                        elif status == "degraded":
                            st.warning(f"⚠️ **{component_name.upper()}**")
                        else:
                            st.error(f"❌ **{component_name.upper()}**")
                        
                        message = component_status.get("message")
                        if message:
                            st.caption(message)
                        
                        latency = component_status.get("latency_ms")
                        if latency is not None:
                            st.caption(f"Latency: {latency}ms")
            
            st.markdown("---")
            st.markdown("### System Metrics")
            
            uptime = detailed_health.get("uptime_seconds", 0)
            uptime_hours = uptime / 3600
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Uptime", f"{uptime_hours:.1f}h", help="Time since the API server was last restarted. A sudden reset to 0 indicates an unexpected restart.")
            
            with col2:
                version = detailed_health.get("version", "N/A")
                st.metric("Version", version, help="Current API server version. Check this matches your expected deployment.")
            
            with col3:
                timestamp = detailed_health.get("timestamp", "N/A")
                if timestamp != "N/A":
                    st.metric("Checked", timestamp[:19], help="Timestamp of the last health check. Should be within the last few seconds.")
                else:
                    st.metric("Checked", "N/A", help="Timestamp of the last health check.")
            
            with col4:
                environment = detailed_health.get("environment", "production")
                st.metric("Environment", environment.capitalize(), help="Deployment environment (production, staging, development). Affects logging verbosity and feature flags.")
        
        st.markdown("---")
        st.markdown("### Readiness Status")
        
        readiness = api.get_readiness()
        
        if readiness.get("ready"):
            st.success("✅ System is ready to accept requests")
        else:
            st.error("❌ System is not ready")
            
            reasons = readiness.get("reasons", [])
            if reasons:
                st.markdown("**Reasons:**")
                for reason in reasons:
                    st.write(f"- {reason}")
    
    except Exception as e:
        st.error(f"Failed to load system health: {str(e)}")

with tab2:
    st.subheader("Resource Utilization")
    st.caption("CPU, Memory, Disk, and Network usage — live metrics from psutil with historical trends from Prometheus")
    
    cpu_usage = None
    memory_usage = None
    disk_usage = None
    network_data = None
    has_real_metrics = False
    
    try:
        system_metrics = api.get_system_metrics()
        
        if system_metrics and not system_metrics.get("error"):
            cpu_usage = system_metrics.get("cpu", {}).get("percent")
            memory_usage = system_metrics.get("memory", {}).get("percent")
            disk_usage = system_metrics.get("disk", {}).get("percent")
            network_data = system_metrics.get("network", {})
            
            has_real_metrics = cpu_usage is not None and memory_usage is not None
            
            if has_real_metrics:
                st.success("✅ **LIVE METRICS** | Real-time system metrics from psutil + Prometheus integration.")
            else:
                st.error("❌ **DATA UNAVAILABLE** | Unable to fetch real system metrics. Check API connection.")
        else:
            st.error(f"❌ **DATA UNAVAILABLE** | {system_metrics.get('error', 'Unknown error')}")
            
    except Exception as e:
        st.error(f"❌ **DATA UNAVAILABLE** | Unable to fetch real system metrics: {str(e)[:100]}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if cpu_usage is not None:
            fig = create_gauge_chart(
                value=cpu_usage,
                max_value=100,
                title="CPU Usage (%)",
                thresholds={'low': 50, 'medium': 75, 'high': 100}
            )
            st.plotly_chart(fig, use_container_width=True, key="sysmon_api_latency")
        else:
            st.metric("CPU Usage (%)", "N/A", help="Percentage of CPU capacity in use. <50% healthy, 50–80% moderate, >80% warning, >90% critical.")
            st.caption("❌ Data unavailable")
    
    with col2:
        if memory_usage is not None:
            fig = create_gauge_chart(
                value=memory_usage,
                max_value=100,
                title="Memory Usage (%)",
                thresholds={'low': 50, 'medium': 75, 'high': 100}
            )
            st.plotly_chart(fig, use_container_width=True, key="sysmon_requests")
            mem_info = system_metrics.get("memory", {})
            if mem_info.get("used_gb"):
                st.caption(f"Used: {mem_info['used_gb']} / {mem_info['total_gb']} GB")
        else:
            st.metric("Memory Usage (%)", "N/A", help="Percentage of RAM in use. <50% healthy, 50–75% moderate, >75% warning, >90% critical.")
            st.caption("❌ Data unavailable")
    
    with col3:
        if disk_usage is not None:
            fig = create_gauge_chart(
                value=disk_usage,
                max_value=100,
                title="Disk Usage (%)",
                thresholds={'low': 50, 'medium': 75, 'high': 100}
            )
            st.plotly_chart(fig, use_container_width=True, key="sysmon_errors")
            disk_info = system_metrics.get("disk", {})
            if disk_info.get("used_gb"):
                st.caption(f"Used: {disk_info['used_gb']} / {disk_info['total_gb']} GB")
        else:
            st.metric("Disk Usage (%)", "N/A", help="Percentage of disk space consumed. Keep below 90% to avoid write failures. Alert triggers at <10% free.")
            st.caption("❌ Data unavailable")
    
    with col4:
        if network_data:
            bytes_sent = network_data.get("bytes_sent", 0)
            bytes_recv = network_data.get("bytes_recv", 0)
            sent_gb = bytes_sent / (1024**3)
            recv_gb = bytes_recv / (1024**3)
            st.metric("Network Sent", f"{sent_gb:.2f} GB", help="Total bytes sent since container start. Useful for tracking outbound API/LLM traffic.")
            st.metric("Network Recv", f"{recv_gb:.2f} GB", help="Total bytes received since container start. High values may indicate heavy inbound data ingestion.")
            st.caption(f"Errors: {network_data.get('errors_in', 0)} in / {network_data.get('errors_out', 0)} out")
        else:
            st.metric("Network", "N/A", help="Network I/O statistics unavailable.")
            st.caption("❌ Data unavailable")
    
    st.markdown("---")
    
    st.markdown("### Resource Trends (Historical)")
    
    trend_time = st.selectbox("Time Range", ["1h", "6h", "24h", "7d"], index=0, key="trend_time", help="Select the historical window for CPU and memory trend charts. Longer ranges help spot sustained usage patterns for capacity planning.")
    
    try:
        col_cpu, col_mem = st.columns(2)
        
        with col_cpu:
            cpu_history = api.get_metrics_history(metric="cpu", time_range=trend_time)
            if cpu_history and cpu_history.get("values") and len(cpu_history["values"]) > 0:
                cpu_df = pd.DataFrame(cpu_history["values"])
                cpu_df['timestamp'] = pd.to_datetime(cpu_df['timestamp'])
                cpu_df['value'] = cpu_df['value'].fillna(0)
                
                fig_cpu = go.Figure()
                fig_cpu.add_trace(go.Scatter(
                    x=cpu_df['timestamp'], 
                    y=cpu_df['value'],
                    mode='lines',
                    name='CPU %',
                    fill='tozeroy',
                    line=dict(color='#3b82f6')
                ))
                fig_cpu.update_layout(
                    title=f"CPU Usage ({trend_time})",
                    yaxis_title="Usage (%)",
                    yaxis_range=[0, 100],
                    height=300
                )
                st.plotly_chart(fig_cpu, use_container_width=True, key="sysmon_cpu")
            else:
                st.error(f"❌ CPU history unavailable: {cpu_history.get('error', cpu_history.get('message', 'No data'))}")
        
        with col_mem:
            mem_history = api.get_metrics_history(metric="memory", time_range=trend_time)
            if mem_history and mem_history.get("values") and len(mem_history["values"]) > 0:
                mem_df = pd.DataFrame(mem_history["values"])
                mem_df['timestamp'] = pd.to_datetime(mem_df['timestamp'])
                mem_df['value'] = mem_df['value'].fillna(0)
                
                fig_mem = go.Figure()
                fig_mem.add_trace(go.Scatter(
                    x=mem_df['timestamp'],
                    y=mem_df['value'],
                    mode='lines',
                    name='Memory %',
                    fill='tozeroy',
                    line=dict(color='#10b981')
                ))
                fig_mem.update_layout(
                    title=f"Memory Usage ({trend_time})",
                    yaxis_title="Usage (%)",
                    yaxis_range=[0, 100],
                    height=300
                )
                st.plotly_chart(fig_mem, use_container_width=True, key="sysmon_memory")
            else:
                st.error(f"❌ Memory history unavailable: {mem_history.get('error', mem_history.get('message', 'No data'))}")
                
    except Exception as e:
        st.error(f"❌ Failed to fetch historical metrics: {str(e)[:100]}")
    
    st.markdown("---")
    
    st.markdown("### Container Metrics")
    
    try:
        container_metrics = api.get_container_metrics()
        containers = container_metrics.get("containers", [])
        
        if containers:
            container_df = pd.DataFrame(containers)
            
            column_mapping = {
                "name": "Container",
                "cpu_percent": "CPU %",
                "memory_mb": "Memory (MB)",
                "network_rx_kbps": "Net RX (KB/s)",
                "network_tx_kbps": "Net TX (KB/s)"
            }
            
            available_cols = [col for col in column_mapping.keys() if col in container_df.columns]
            container_df = container_df[available_cols].rename(columns=column_mapping)
            
            if "Container" in container_df.columns:
                container_df["Container"] = container_df["Container"].str.replace("agentic_", "")
            
            st.dataframe(container_df, use_container_width=True, hide_index=True)
        else:
            error_msg = container_metrics.get("error", "No container data available")
            st.error(f"❌ Container metrics unavailable: {error_msg}")
            
    except Exception as e:
        st.error(f"❌ Failed to fetch container metrics: {str(e)[:100]}")

with tab3:
    st.subheader("External Service Status")
    st.caption("Database, Redis, GPU, and LLM service connectivity — checks are performed live on each page load")
    
    try:
        detailed_health = api.get_detailed_health()
        components = detailed_health.get("components", {})
        
        st.markdown("### 🗄️ PostgreSQL Database")
        
        db_status = components.get("database", {})
        
        if db_status:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status = db_status.get("status", "unknown")
                if status == "healthy":
                    st.success("✅ Connected")
                else:
                    st.error("❌ Disconnected")
            
            with col2:
                latency = db_status.get("latency_ms", 0)
                st.metric("Latency", f"{latency:.1f}ms", help="Round-trip time for a simple database health query. <10ms ideal, <50ms acceptable, >100ms indicates potential issues.")
            
            with col3:
                pool_info = "N/A"
                db_pool = None
                try:
                    db_pool = api.request("GET", "/operations/database/health")
                    if db_pool and db_pool.get("pool_size"):
                        pool_info = f"{db_pool.get('checked_out', 0)}/{db_pool.get('pool_size', 0)}"
                except Exception:
                    pass
                st.metric("Pool Size", pool_info, help="Active/Total connections in pool")
            
            with st.expander("Database Details"):
                st.write(f"**Host:** {db_status.get('host', 'localhost:5432')}")
                db_name = db_pool.get('database_name', 'agentic') if db_pool else 'agentic'
                pg_version = db_pool.get('pg_version', 'PostgreSQL') if db_pool else 'PostgreSQL'
                st.write(f"**Database:** {db_name}")
                st.write(f"**Extensions:** pgvector, uuid-ossp")
                st.write(f"**Version:** {pg_version}")
                if db_pool:
                    st.markdown("---")
                    st.markdown("**Connection Pool Stats:**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"• Pool Size: {db_pool.get('pool_size', 'N/A')}")
                        st.write(f"• Max Overflow: {db_pool.get('max_overflow', 'N/A')}")
                        st.write(f"• Checked Out: {db_pool.get('checked_out', 'N/A')}")
                    with col_b:
                        st.write(f"• Idle Connections: {db_pool.get('idle_connections', 'N/A')}")
                        st.write(f"• Active Queries: {db_pool.get('active_queries', 'N/A')}")
                        st.write(f"• Avg Query Time: {db_pool.get('avg_query_ms', 'N/A')}ms")
                    st.write(f"**Database Size:** {db_pool.get('database_size', 'Unknown')}")
        
        st.markdown("---")
        
        st.markdown("### 📦 Redis Cache")
        
        redis_status = components.get("redis", {})
        
        if redis_status:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status = redis_status.get("status", "unknown")
                if status == "healthy":
                    st.success("✅ Connected")
                else:
                    st.error("❌ Disconnected")
            
            with col2:
                latency = redis_status.get("latency_ms", 0)
                st.metric("Latency", f"{latency:.1f}ms", help="Round-trip time for a Redis PING command. <5ms ideal, <20ms acceptable. High latency impacts LLM cache performance.")
            
            with col3:
                try:
                    cache_metrics = api.request("GET", "/costs/cache-metrics")
                    if cache_metrics and cache_metrics.get("hit_rate") is not None:
                        hit_rate = cache_metrics.get("hit_rate", 0)
                        st.metric("Hit Rate", f"{hit_rate:.1f}%", help="Percentage of LLM requests served from cache. Higher is better — target >80%. Below 50% means most requests hit the LLM directly, increasing cost and latency.")
                    else:
                        st.metric("Hit Rate", "N/A", help="Cache hit rate unavailable. Check Redis connectivity.")
                except Exception:
                    st.metric("Hit Rate", "N/A", help="Cache hit rate unavailable. Check Redis connectivity.")
            
            with st.expander("Redis Details"):
                st.write(f"**Host:** {redis_status.get('host', 'localhost:6379')}")
                st.write(f"**Mode:** Standalone")
                used_memory = redis_status.get("used_memory_mb")
                if used_memory:
                    st.write(f"**Memory Used:** {used_memory} MB")
                try:
                    if cache_metrics:
                        st.write(f"**Semantic Cache Entries:** {cache_metrics.get('total_entries', 'N/A')}")
                        st.write(f"**Cache Hits:** {cache_metrics.get('cache_hits', 0)}")
                        st.write(f"**Cache Misses:** {cache_metrics.get('cache_misses', 0)}")
                        st.write(f"**Est. Cost Savings:** €{cache_metrics.get('estimated_cost_savings', 0):.2f}")
                except Exception:
                    pass
        
        st.markdown("---")

        st.markdown("### 🖥️ GPU Status")

        gpu_status = components.get("gpu", {})
        ollama_status = components.get("ollama", {})

        if gpu_status:
            status = gpu_status.get("status", "unknown")

            if status == "healthy":
                st.success(f"✅ GPU Available: {gpu_status.get('model', 'NVIDIA GPU')}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("GPU Model", gpu_status.get("model", "NVIDIA GPU")[:30], help="Detected GPU hardware model used for LLM inference acceleration.")

                with col2:
                    st.metric("VRAM", gpu_status.get("memory_gb", "N/A"), help="Total video RAM available for model loading. Larger models require more VRAM.")

                with col3:
                    st.metric("CUDA Version", gpu_status.get("cuda_version", "N/A"), help="Installed CUDA toolkit version. Required for GPU-accelerated inference.")

                with st.expander("GPU Details"):
                    st.write(f"**GPU Model:** {gpu_status.get('model', 'Unknown')}")
                    st.write(f"**VRAM:** {gpu_status.get('memory_gb', 'Unknown')}")
                    st.write(f"**CUDA Version:** {gpu_status.get('cuda_version', 'Unknown')}")
                    st.write(f"**Status:** {gpu_status.get('message', 'Available')}")
            else:
                st.info(f"ℹ️ GPU Status: {gpu_status.get('message', 'Not available')}")
        else:
            st.info("ℹ️ No GPU detected - running in CPU mode")
        
        st.markdown("---")
        
        st.markdown("### 🤖 Ollama LLM Service")
        
        if ollama_status:
            status = ollama_status.get("status", "unknown")
            
            if status == "healthy":
                st.success("✅ Ollama Service Running")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    models_available = ollama_status.get("models_available", 0)
                    st.metric("Models Available", models_available, help="Total number of Ollama models downloaded and ready for use.")
                
                with col2:
                    models_loaded = ollama_status.get("models_loaded", 0)
                    st.metric("In VRAM", models_loaded, help="Models currently loaded in GPU memory. Models load automatically on first use and unload after idle timeout.")
                
                with col3:
                    configured_model = ollama_status.get("configured_model", "N/A")
                    st.metric("Active Model", configured_model[:20], help="The model currently configured for content generation. Change this on the LLM Management page.")
                
                model_list = ollama_status.get("model_list", [])
                if model_list:
                    st.markdown("**📦 Installed Models:**")
                    for m in model_list:
                        loaded_indicator = "🟢" if m.get("name") in ollama_status.get("loaded_models", []) else "⚪"
                        active_indicator = "✓" if m.get("name") == configured_model else ""
                        st.markdown(f"  {loaded_indicator} `{m.get('name')}` ({m.get('size')}) {active_indicator}")
                
                if models_available > 0 and models_loaded == 0:
                    st.info("ℹ️ **Note:** Models load into VRAM on first request. Generate content to load a model.")
                
                with st.expander("Ollama Details"):
                    st.write(f"**Host:** {ollama_status.get('host', 'Unknown')}")
                    st.write(f"**Status:** {ollama_status.get('message', 'Running')}")
                    st.write(f"**Configured Model:** {configured_model}")
                    st.write(f"**Models Available:** {models_available}")
                    st.write(f"**Models in VRAM:** {models_loaded}")
                    if ollama_status.get("loaded_models"):
                        st.write(f"**Loaded Models:** {', '.join(ollama_status.get('loaded_models', []))}")
                    if ollama_status.get("gpu_info"):
                        st.markdown("---")
                        st.markdown("**GPU Inference:**")
                        gpu = ollama_status.get("gpu_info")
                        st.write(f"• Model: {gpu.get('model', 'N/A')}")
                        st.write(f"• VRAM Used: {gpu.get('vram_used', 'N/A')}")
                
                st.markdown("---")
                st.info("💡 **Manage models and prompts:** Use the **🤖 LLM Management** page to download, change, or test models and configure prompts.")
            else:
                st.error(f"❌ Ollama Unavailable: {ollama_status.get('message', 'Service not reachable')}")
        else:
            st.warning("⚠️ Ollama status unknown")
    
    except Exception as e:
        st.error(f"Failed to load service status: {str(e)}")

with tab4:
    st.subheader("Metrics & Monitoring")
    st.caption("Prometheus metrics collection and Grafana dashboards — external tools for deep observability")
    
    st.markdown("""
    
    **Prometheus** - Metrics collection and alerting
    - Endpoint: `http://localhost:9090`
    - Scrape interval: 15s
    - Retention: 15 days
    
    **Grafana** - Visualization and dashboards
    - Endpoint: `http://localhost:3000`
    - Default credentials: admin/admin
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Prometheus")
        
        if st.button("🔗 Open Prometheus", use_container_width=True, help="Opens Prometheus query interface for ad-hoc metric exploration and PromQL queries."):
            st.info("Open http://localhost:9090 in your browser")
        
        st.markdown("**Key Metrics:**")
        st.write("- `http_requests_total` - Total HTTP requests")
        st.write("- `http_request_duration_seconds` - Request latency")
        st.write("- `campaign_count` - Active campaigns")
        st.write("- `agent_memory_ops` - Memory operations")
        st.write("- `ope_evaluations_total` - OPE evaluations")
    
    with col2:
        st.markdown("#### 📈 Grafana")
        
        if st.button("🔗 Open Grafana", use_container_width=True, help="Opens Grafana dashboard interface for pre-built visualizations of system and application metrics."):
            st.info("Open http://localhost:3000 in your browser")
        
        st.markdown("**Available Dashboards:**")
        st.write("- System Overview")
        st.write("- Campaign Performance")
        st.write("- Agent Metrics")
        st.write("- Database Performance")
        st.write("- Cost Tracking")
    
    st.markdown("---")
    
    st.markdown("### Alert Configuration")
    
    st.info("""
    **Active Alerts:**
    - High error rate (>5% for 5m)
    - High latency (P95 > 500ms for 5m)
    - Database connection pool exhaustion
    - Memory usage > 90%
    - Disk space < 10%
    """)
    
    st.markdown("#### Recent Alerts")
    
    try:
        alerts_data = api.get_active_alerts()
        real_alerts = alerts_data.get('alerts', {})
        has_real_alerts = False
        
        for severity in ['critical', 'error', 'warning', 'info']:
            alert_list = real_alerts.get(severity, [])
            if alert_list:
                has_real_alerts = True
                for alert in alert_list[:3]:
                    timestamp = alert.get('timestamp', 'N/A')[:19] if alert.get('timestamp') else 'N/A'
                    message = alert.get('message', alert.get('title', 'Unknown alert'))
                    
                    if severity == 'critical':
                        st.error(f"🔴 **{timestamp}** - {message}")
                    elif severity == 'error':
                        st.warning(f"🟠 **{timestamp}** - {message}")
                    elif severity == 'warning':
                        st.warning(f"🟡 **{timestamp}** - {message}")
                    else:
                        st.info(f"🔵 **{timestamp}** - {message}")
        
        if not has_real_alerts:
            st.success("✅ No active alerts")
            
    except Exception as e:
        st.info("ℹ️ Alert data available on System Transparency page")

st.markdown("---")
st.caption(f"System Monitor | Last updated: {datetime.now().strftime('%H:%M:%S')}")
