"""
LLM Management Dashboard Page
Manage Ollama models, test prompts, and configure LLM settings.
"""
import streamlit as st
from utils.api_client import AgenticAPIClient


st.set_page_config(
    page_title="LLM Management - Agentic",
    page_icon="🤖",
    layout="wide"
)

from utils.llm_checks import check_llm_readiness, render_llm_status_banner

st.title("🤖 LLM Management")
st.caption("Manage Ollama models, configure prompts, and test generation")

with st.expander("ℹ️ LLM Management Guide", expanded=False):
    st.markdown("""
    **What this page does:** Manage local LLM models via [Ollama](https://ollama.com), test prompts interactively, manage reusable prompt templates, and configure generation settings.

    **Key concepts:**
    - **Ollama** — A local LLM inference server. Runs on your machine, fully private, no API costs.
    - **Models** — e.g. `gemma2`, `llama3`, `mistral`. Different sizes suit different tasks (smaller = faster, larger = higher quality).
    - **Model Testing** — Send test prompts with adjustable temperature/token settings to preview output before deploying to production.
    - **Prompt Templates** — Reusable, version-controlled templates for content generation, safety checks, and claim validation.
    - **Template lifecycle** — `Draft → Testing → Validated → Production`. Templates must pass validation before deployment (safe promotion flow).
    - **Configuration** — Set the active model, safety scoring parameters, Ollama connection, and response parsing patterns.

    **Tips:**
    - Use smaller models (2B–7B parameters) for speed and lower resource usage; larger models (13B+) for higher quality output.
    - Always test prompts after editing to ensure the response format matches expected parsing patterns.
    - Back up prompt templates before making changes — use the Reset button to restore defaults if needed.
    """)

@st.cache_resource
def get_api_client():
    return AgenticAPIClient()

api = get_api_client()

# ── Global LLM readiness check ─────────────────────────────────────────
_llm_status = render_llm_status_banner(api, context="general")

_ollama_available = _llm_status.get("ollama_available", False)

tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Model Management",
    "🧪 Model Testing",
    "📝 Prompt Templates",
    "⚙️ Configuration"
])

with tab1:
    st.subheader("Installed Models")
    st.caption("View, activate, delete, and download Ollama models. The active model is used for all LLM operations.")
    
    if not _ollama_available:
        st.warning("🤖 Ollama is not available. Ensure Ollama is installed and running, then configure `OLLAMA_HOST` in ⚙️ Operations → System Settings → llm category.")
    
    try:
        models_data = api.request("GET", "/llm/models")
        
        if models_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Models Installed", models_data.get("total", 0), help="Total number of Ollama models downloaded to this server.")
            with col2:
                loaded = models_data.get("loaded_count", 0)
                st.metric("In VRAM", loaded, help="Models currently loaded in GPU memory. Models load automatically on first use and unload after idle timeout.")
            with col3:
                st.metric("Active Model", models_data.get("configured_model", "N/A"), help="The model currently configured for all LLM operations (content generation, safety scoring, etc.).")
            
            # Detect mismatch: configured model not in installed models
            _configured = models_data.get("configured_model", "")
            _installed_names = [m.get("name") for m in models_data.get("models", [])]
            _has_active = any(m.get("is_active") for m in models_data.get("models", []))
            if _configured and _installed_names and _configured not in _installed_names:
                st.error(
                    f"🚨 **Active model '{_configured}' is NOT installed!** "
                    f"All content generation and safety checks will fail.\n\n"
                    f"**Installed models:** {', '.join(_installed_names)}\n\n"
                    f"**Fix:** Click **Set Active** on one of the installed models below, "
                    f"or download `{_configured}` using the Download section."
                )
            elif not _has_active and _installed_names:
                st.warning(
                    "⚠️ **No model is marked as active.** "
                    "Click **Set Active** on one of the installed models below to enable content generation."
                )
            
            st.markdown("---")
            
            models = models_data.get("models", [])
            if models:
                for model in models:
                    with st.container():
                        col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 2])
                        
                        with col_a:
                            if model.get("is_active") and model.get("is_loaded"):
                                status_icon = "🟢"
                            elif model.get("is_active"):
                                status_icon = "🟡"
                            else:
                                status_icon = "⚪"
                            active_badge = " ✓ Active" if model.get("is_active") else ""
                            st.markdown(f"**{status_icon} {model.get('name')}**{active_badge}")
                        
                        with col_b:
                            st.write(f"{model.get('size_gb', 0)} GB")
                        
                        with col_c:
                            if model.get("is_loaded"):
                                st.success("In VRAM")
                            else:
                                st.caption("Idle")
                        
                        with col_d:
                            btn_col1, btn_col2 = st.columns(2)
                            with btn_col1:
                                if not model.get("is_active"):
                                    if st.button("Set Active", key=f"activate_{model.get('name')}"):
                                        with st.spinner(f"Activating & loading {model.get('name')}..."):
                                            result = api.request("PUT", "/llm/models/active", {
                                                "model_name": model.get("name")
                                            })
                                        if result and result.get("status") == "updated":
                                            st.toast(f"✅ {model.get('name')} is now active and loaded into VRAM", icon="✅")
                                            st.rerun()
                            with btn_col2:
                                if st.button("🗑️ Delete", key=f"delete_{model.get('name')}"):
                                    result = api.request("DELETE", f"/llm/models/{model.get('name')}")
                                    if result and result.get("status") == "deleted":
                                        st.toast(f"✅ Deleted {model.get('name')}", icon="✅")
                                        st.rerun()
                        
                        st.markdown("---")
                st.caption("🟢 Active & in VRAM — 🟡 Active, loads on first request — ⚪ Inactive")
            else:
                st.warning("No models installed. Download a model below.")
            
            st.subheader("📥 Download New Model")
            
            popular = api.request("GET", "/llm/models/popular")
            popular_models = popular.get("models", []) if popular else []
            
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                model_options = [m.get("name") for m in popular_models]
                selected_model = st.selectbox(
                    "Select model to download",
                    options=[""] + model_options,
                    format_func=lambda x: x if x else "Choose a model...",
                    help="Popular pre-configured models. Size and capability vary — check the details shown after selection."
                )
                
                if selected_model:
                    model_info = next((m for m in popular_models if m.get("name") == selected_model), None)
                    if model_info:
                        st.info(f"""
                        **{model_info.get('description')}**
                        - Size: ~{model_info.get('size_gb')} GB
                        - Parameters: {model_info.get('parameters')}
                        - Use case: {model_info.get('use_case')}
                        """)
                
                st.markdown("**Or enter custom model name:**")
                custom_model = st.text_input("Custom model name", placeholder="e.g., llama3:latest", help="Enter any model name from the Ollama model library (e.g., 'phi3:mini', 'codellama:7b'). Must match the exact Ollama model tag.")
            
            with col_right:
                st.markdown("<br>", unsafe_allow_html=True)
                model_to_download = custom_model if custom_model else selected_model
                
                if st.button("📥 Download Model", disabled=not model_to_download, type="primary"):
                    if model_to_download:
                        with st.spinner(f"Starting download of {model_to_download}..."):
                            result = api.request("POST", "/llm/models/pull", {
                                "model_name": model_to_download
                            })
                            if result and result.get("status") == "started":
                                st.success(f"""
                                ✅ Download started for **{model_to_download}**
                                
                                The model is downloading in the background. 
                                Refresh this page to check progress.
                                """)
                            else:
                                st.error(f"Failed to start download: {result}")
    
    except Exception as e:
        st.error(f"Failed to load model data: {str(e)}")


with tab2:
    st.subheader("Test Model Generation")
    st.caption("Send a test prompt to verify model output. Experiment with different models and parameters before using them in production.")
    
    if not _ollama_available:
        st.warning("🤖 Ollama is not available. Model testing requires a running Ollama instance. Configure in ⚙️ Operations → System Settings → llm category.")
    else:
        try:
            models_data = api.request("GET", "/llm/models")
            installed_models = [m.get("name") for m in models_data.get("models", [])] if models_data else []
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                test_model = st.selectbox(
                    "Select model",
                    options=installed_models if installed_models else ["No models installed"],
                    index=0,
                    help="Choose which LLM model to use. Smaller models are faster, larger models produce better quality."
                )
            
            with col2:
                col_temp, col_tokens = st.columns(2)
                with col_temp:
                    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, help="Controls randomness. 0=deterministic, 1=creative, 2=very random. Recommended: 0.7 for content, 0.1 for safety.")
                with col_tokens:
                    max_tokens = st.slider("Max Tokens", 50, 2000, 500, 50, help="Maximum response length. Higher = longer output but slower and more expensive.")
            
            preset_prompts = {
                "Simple greeting": "Hello! Please introduce yourself briefly.",
                "Content generation": "Write a short LinkedIn post about AI in marketing (2-3 sentences).",
                "Claim validation": "Analyze this claim for accuracy: 'AI can reduce marketing costs by 50%'",
                "Safety check": "Is the following content appropriate for B2B marketing? 'Our product is the best!'",
                "Custom": ""
            }
            
            selected_preset = st.selectbox("Choose preset prompt", list(preset_prompts.keys()), help="Pre-configured test prompts for common tasks. Choose 'Custom' to write your own.")
            
            if selected_preset == "Custom":
                test_prompt = st.text_area(
                    "Enter your test prompt",
                    height=150,
                    placeholder="Enter any prompt to test the model...",
                    help="Write any prompt to test how the model responds. Use this to experiment before creating prompt templates."
                )
            else:
                test_prompt = st.text_area(
                    "Test prompt",
                    value=preset_prompts[selected_preset],
                    height=150
                )
            
            if st.button("🚀 Generate", type="primary", disabled=not test_prompt or not installed_models):
                with st.spinner("Generating response..."):
                    result = api.request("POST", "/llm/models/test", {
                        "prompt": test_prompt,
                        "model": test_model,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    })
                    
                    if result:
                        st.success(f"✅ Generated in {result.get('elapsed_seconds', 0):.1f}s ({result.get('tokens_per_second', 0):.1f} tokens/sec)")
                        
                        st.markdown("### Response:")
                        st.markdown(result.get("response", "No response"))
                        
                        with st.expander("Generation Details"):
                            st.write(f"**Model:** {result.get('model')}")
                            st.write(f"**Tokens Generated:** {result.get('tokens_generated', 0)}")
                            st.write(f"**Time:** {result.get('elapsed_seconds', 0):.1f}s")
                            st.write(f"**Speed:** {result.get('tokens_per_second', 0):.1f} tokens/sec")
                    else:
                        st.error("Generation failed")
        except Exception as e:
            st.error(f"Failed to load test interface: {str(e)}")


with tab3:
    st.subheader("Prompt Templates")
    st.caption("View and edit system prompts used for content generation and validation. Changes require validation before deployment.")
    
    try:
        prompts_data = api.request("GET", "/llm/prompts")
        
        if prompts_data:
            prompt_files = prompts_data.get("prompts", [])
            
            file_options = {p.get("file"): p.get("category") for p in prompt_files}
            selected_file = st.selectbox(
                "Select prompt file",
                options=list(file_options.keys()),
                format_func=lambda x: f"{file_options[x]} ({x})",
                help="Each file contains prompt templates for a specific category (e.g., content generation, safety scoring, claim validation)."
            )
            
            if selected_file:
                file_data = api.request("GET", f"/llm/prompts/{selected_file}")
                
                if file_data:
                    content = file_data.get("content", {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Version:** {content.get('version', 'N/A')}")
                    with col2:
                        st.write(f"**Last Updated:** {content.get('last_updated', 'N/A')}")
                    
                    st.markdown("---")
                    
                    templates = [k for k in content.keys() if k not in ['version', 'last_updated', 'review_required_after', 'validation_rules', 'final_assessment', 'scoring_methodology', 'decision_rules', 'review_priority', 'dimensions']]
                    
                    if templates:
                        selected_template = st.selectbox(
                            "Select template to view/edit",
                            options=templates,
                            help="Choose a specific template within this file. Each template serves a different purpose (e.g., system prompt, user prompt, scoring rubric)."
                        )
                        
                        if selected_template:
                            template_data = content.get(selected_template, {})
                            
                            st.markdown(f"### {selected_template}")
                            
                            import yaml
                            template_yaml = yaml.dump({selected_template: template_data}, default_flow_style=False, allow_unicode=True, sort_keys=False)
                            
                            edited_yaml = st.text_area(
                                "Template Content (YAML)",
                                value=template_yaml,
                                height=400,
                                help="Edit the YAML content carefully. Changes must pass validation before deployment."
                            )
                            
                            st.warning("⚠️ **Production Safety:** All prompt changes must pass comprehensive validation before deployment.")
                            
                            col_validate, col_deploy, col_reset = st.columns(3)
                            
                            with col_validate:
                                if st.button("🧪 Validate & Test", type="secondary"):
                                    with st.spinner("Running comprehensive validation tests (this takes ~45-60s)..."):
                                        try:
                                            result = api.request(
                                                "POST",
                                                f"/llm/prompts/{selected_file}/{selected_template}/validate",
                                                {"prompt_content": edited_yaml},
                                                timeout=180
                                            )
                                        except Exception as e:
                                            result = None
                                            st.error(f"Validation request failed: {e}")
                                        
                                        if result:
                                            st.session_state["validation_result"] = result
                                            
                                            if result.get("can_deploy"):
                                                st.success(f"✅ All tests passed! Ready for deployment.")
                                            else:
                                                st.error(f"❌ Validation failed - cannot deploy")
                                            
                                            with st.expander("📊 Validation Results", expanded=True):
                                                st.write(f"**Test ID:** {result.get('test_id')}")
                                                st.write(f"**Status:** {result.get('overall_status')}")
                                                st.write(f"**Can Deploy:** {'✅ Yes' if result.get('can_deploy') else '❌ No'}")
                                                
                                                for phase_name, phase_data in result.get("phases", {}).items():
                                                    st.markdown(f"**{phase_name}:** {phase_data.get('status', 'N/A')}")
                                                    if phase_data.get("pass_rate") is not None:
                                                        st.write(f"  Pass Rate: {phase_data.get('pass_rate')*100:.0f}%")
                                                
                                                if result.get("blocking_issues"):
                                                    st.markdown("**🚫 Blocking Issues:**")
                                                    for issue in result.get("blocking_issues"):
                                                        st.error(f"• {issue}")
                                        else:
                                            st.error("Validation request failed")
                            
                            with col_deploy:
                                validation_passed = st.session_state.get("validation_result", {}).get("can_deploy", False)
                                
                                if st.button("🚀 Deploy to Production", type="primary", disabled=not validation_passed):
                                    with st.spinner("Deploying validated prompt (re-running validation)..."):
                                        result = api.request(
                                            "POST",
                                            f"/llm/prompts/{selected_file}/{selected_template}/deploy",
                                            {"prompt_content": edited_yaml},
                                            timeout=180
                                        )
                                        
                                        if result and result.get("deployed"):
                                            st.success(f"✅ {result.get('message')}")
                                            st.balloons()
                                            st.session_state.pop("validation_result", None)
                                        else:
                                            st.error(f"❌ Deployment blocked: {result.get('message', 'Unknown error')}")
                                            if result.get("blocking_issues"):
                                                for issue in result.get("blocking_issues"):
                                                    st.error(f"• {issue}")
                                
                                if not validation_passed:
                                    st.caption("Run validation first to enable deployment")
                            
                            with col_reset:
                                if st.button("🔄 Reset to Default"):
                                    with st.spinner("Resetting template to default..."):
                                        result = api.request("POST", f"/llm/prompts/{selected_file}/{selected_template}/reset")
                                        if result and result.get("success"):
                                            st.success(f"✅ {result.get('message')}")
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Reset failed: {result.get('detail', 'Unknown error') if result else 'No response'}")
                            
                            template_info = api.request("GET", f"/llm/prompts/{selected_file}/{selected_template}")
                            if template_info and template_info.get("variables"):
                                with st.expander("📌 Variables Used in Template"):
                                    for var in template_info.get("variables", []):
                                        st.code(f"{{{var}}}")
                    else:
                        st.info("No editable templates in this file")
                        
                        with st.expander("View Raw Content"):
                            st.code(file_data.get("raw_yaml", ""), language="yaml")
    
    except Exception as e:
        st.error(f"Failed to load prompts: {str(e)}")


with tab4:
    st.subheader("LLM Configuration")
    st.caption("Configure Ollama connection, default model, generation parameters, and response parsing settings.")
    
    try:
        config_list = api.request("GET", "/config/category/llm")
        llm_configs = {}
        
        if config_list and isinstance(config_list, list):
            for item in config_list:
                key = item.get("key", "")
                if "OLLAMA" in key or "LLM" in key:
                    llm_configs[key] = item.get("display_value", "")
        
        st.markdown("### Current Settings")
        
        if llm_configs:
            for key, value in llm_configs.items():
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.write(f"**{key}**")
                
                with col2:
                    if key == "OLLAMA_MODEL":
                        models_data = api.request("GET", "/llm/models")
                        installed = [m.get("name") for m in models_data.get("models", [])] if models_data else []
                        
                        current_idx = installed.index(value) if value in installed else 0
                        new_value = st.selectbox(
                            f"Select {key}",
                            options=installed if installed else [value],
                            index=current_idx,
                            key=f"config_{key}",
                            label_visibility="collapsed",
                            help="The default model used for all LLM operations. Change this to switch which model handles content generation and safety scoring."
                        )
                    elif key == "OLLAMA_TEMPERATURE":
                        new_value = st.slider(
                            f"Set {key}",
                            0.0, 2.0, float(value) if value else 0.7, 0.1,
                            key=f"config_{key}",
                            label_visibility="collapsed",
                            help="Default temperature for all LLM calls. 0=deterministic, 1=creative, 2=very random. Recommended: 0.7 for content, 0.1 for safety scoring."
                        )
                    elif key == "OLLAMA_MAX_TOKENS":
                        new_value = st.number_input(
                            f"Set {key}",
                            min_value=100, max_value=4000, value=int(value) if value else 2000,
                            key=f"config_{key}",
                            label_visibility="collapsed",
                            help="Default maximum response length for all LLM calls. Higher values allow longer output but increase latency and memory usage."
                        )
                    elif key == "USE_LOCAL_LLM":
                        new_value = st.checkbox(
                            "Enable",
                            value=str(value).lower() == "true",
                            key=f"config_{key}",
                            label_visibility="collapsed",
                            help="When enabled, the platform uses the local Ollama LLM for content generation and safety scoring instead of external APIs."
                        )
                    else:
                        new_value = st.text_input(
                            f"Set {key}",
                            value=str(value) if value else "",
                            key=f"config_{key}",
                            label_visibility="collapsed",
                            help=f"Configuration value for {key}. Refer to the Operations guide for details on this setting."
                        )
                
                with col3:
                    if st.button("Save", key=f"save_{key}"):
                        result = api.request("PUT", f"/config/value/{key}", {"value": str(new_value)})
                        if result:
                            st.toast(f"✅ Saved", icon="✅")
                            st.rerun()
                        else:
                            st.error("Failed to save")
        else:
            st.info("No LLM configurations found")
        
        st.markdown("---")
        
        st.markdown("### Connection Status")
        
        health = api.get_detailed_health()
        if health:
            ollama = health.get("components", {}).get("ollama", {})
            
            if ollama.get("status") == "healthy":
                st.success(f"✅ Connected to Ollama at {ollama.get('host', 'unknown')}")
                st.write(f"• Models Available: {ollama.get('models_available', 0)}")
                st.write(f"• Models Loaded: {ollama.get('models_loaded', 0)}")
            else:
                st.error(f"❌ Ollama Unavailable: {ollama.get('message', 'Unknown error')}")
        
        st.markdown("---")
        st.markdown("### Response Parsing Patterns")
        st.caption("These regex patterns are used to extract data from LLM responses. Changing prompts may break parsing if these patterns are not matched.")
        
        try:
            patterns_resp = api.request("GET", "/llm/patterns")
            if patterns_resp and patterns_resp.get("patterns"):
                patterns = patterns_resp.get("patterns")
            else:
                patterns = {
                    "Toxicity Score": r"\*{0,2}TOXICITY_SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)",
                    "Factuality Score": r"\*{0,2}FACTUALITY_SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)",
                    "Brand Score": r"\*{0,2}BRAND_SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)",
                    "Compliance Score": r"\*{0,2}COMPLIANCE_SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)",
                    "Claim Citation": r"\[CLM_\d{3}\]",
                    "Headline Extract": r"Headline:\s*(.+?)(?:\n|$)",
                }
        except:
            patterns = {
                "Toxicity Score": r"\*{0,2}TOXICITY_SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)",
                "Factuality Score": r"\*{0,2}FACTUALITY_SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)",
                "Brand Score": r"\*{0,2}BRAND_SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)",
                "Compliance Score": r"\*{0,2}COMPLIANCE_SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)",
                "Claim Citation": r"\[CLM_\d{3}\]",
                "Headline Extract": r"Headline:\s*(.+?)(?:\n|$)",
            }
        
        for name, pattern in patterns.items():
            with st.expander(name):
                st.code(pattern, language="regex")
                st.caption("LLM responses must include this pattern for parsing to work correctly.")
    
    except Exception as e:
        st.error(f"Failed to load configuration: {str(e)}")


st.markdown("---")
st.caption("💡 **Tip:** Always test prompts after making changes to ensure proper response format.")
