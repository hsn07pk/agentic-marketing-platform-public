"""
Data Configuration Management Dashboard

Manage all data configuration files from a unified interface:
- Claims Library: Verified claims for content grounding
- Brand Voice: Brand identity and style guidelines
- Competitors: Competitor intelligence database
- Product Catalog: Product modules, packages, and governance
"""
import streamlit as st
import pandas as pd
import json

from datetime import datetime

st.set_page_config(
    page_title="Data Configuration | Agentic AI",
    page_icon="📚",
    layout="wide"
)

from utils.api_client import AgenticAPIClient

@st.cache_resource
def get_api():
    return AgenticAPIClient()

api = get_api()

st.title("📚 Data Configuration")
st.caption("Manage claims, brand voice, competitors, and product catalog")

with st.expander("ℹ️ Data Configuration Guide", expanded=False):
    st.markdown("""
**What this page manages:** All source data used by AI agents to generate accurate marketing content.

| Section | Purpose |
|---------|---------|
| **📝 Claims Library** | Verified marketing claims with confidence scores and evidence citations. AI uses these to ground content in facts. |
| **🎨 Brand Voice** | Tone, style, and messaging guidelines that AI follows when generating content. Ensures consistent brand identity. |
| **🏢 Competitors** | Competitor profiles used for competitive differentiation in content. Helps AI highlight your unique advantages. |
| **📦 Product Catalog** | Product features, benefits, modules, and details referenced in marketing content. |
| **📋 Version History** | Automatic version tracking — every edit is versioned, so you can restore any previous state. |

**How data flows into AI content:**
These files feed into **RAG (Retrieval Augmented Generation)** — when AI creates content, it retrieves relevant claims, brand guidelines, competitor intel, and product details to produce accurate, on-brand marketing materials.

**Best practices:**
- Keep claims updated with source citations — stale claims reduce content quality
- Review brand voice quarterly to ensure it reflects current positioning
- Add new competitors as they emerge in the market
- Use confidence scores honestly — AI prioritizes higher-confidence claims

**Safety net:** Every change is auto-versioned. If something goes wrong, switch to the Version History tab and restore any previous state.
    """)


summary = api.get_data_config_summary()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📝 Claims", summary.get("claims", {}).get("total", 0), help="Total verified marketing claims in the library. These ground AI-generated content in facts.")
with col2:
    st.metric("🎨 Brand Voice", f"{len(summary.get('brand_voice', {}).get('sections', []))} sections", help="Number of brand voice sections defined (e.g., tone, style, messaging). AI follows these when generating content.")
with col3:
    st.metric("🏢 Competitors", summary.get("competitors", {}).get("total", 0), help="Number of tracked competitors. AI uses these profiles to differentiate your content.")
with col4:
    st.metric("📦 Modules", summary.get("products", {}).get("modules", 0), help="Number of product modules in the catalog. AI references these when describing product capabilities.")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Claims Library",
    "🎨 Brand Voice",
    "🏢 Competitors",
    "📦 Product Catalog",
    "📋 Version History"
])

with tab1:
    st.subheader("📝 Claims Library")
    st.caption("Verified claims for content grounding and governance — AI retrieves these to back up marketing statements with evidence")
    
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 1, 1, 1])
    
    with ctrl_col1:
        search_claims = st.text_input("🔍 Search claims", placeholder="Search by claim text...", key="claims_search", help="Search by claim text, type, or evidence. Matches anywhere in the claim.")
    
    with ctrl_col2:
        claim_types = api.get_claim_types()
        filter_type = st.selectbox("Type", ["All"] + claim_types, key="claims_type_filter", help="Filter claims by category: product, industry, research, testimonial, statistical.")
    
    with ctrl_col3:
        filter_confidence = st.selectbox("Min Confidence", [None, 3, 4, 5], format_func=lambda x: "All" if x is None else f"≥{x}", key="claims_conf_filter", help="Filter to show only claims at or above this confidence level. Higher confidence claims are prioritized by AI.")
    
    with ctrl_col4:
        if st.button("➕ Add Claim", type="primary", use_container_width=True):
            st.session_state.show_claim_form = True
    
    if st.session_state.get("show_claim_form"):
        with st.expander("➕ New Claim", expanded=True):
            with st.form("new_claim_form"):
                new_claim_text = st.text_area("Claim Text *", height=100, help="The marketing claim statement. Be specific and factual — AI will use this verbatim or paraphrase it.")
                
                form_col1, form_col2 = st.columns(2)
                with form_col1:
                    new_claim_type = st.selectbox("Type", ["qualitative", "quantitative", "methodological", "process"], help="Category of claim: qualitative (opinion-based), quantitative (number-based), methodological (approach-based), or process (workflow-based).")
                    new_confidence = st.slider("Confidence", 1, 5, 4, help="How confident we are in this claim (1-5). Higher confidence claims are used more by AI.")
                    new_personas = st.multiselect("Personas", ["decision_maker", "practitioner", "researcher"], help="Which target personas this claim is relevant for.")
                
                with form_col2:
                    new_source_title = st.text_input("Source Title", help="Title of the source document, study, or article backing this claim.")
                    new_source_url = st.text_input("Source URL", help="URL to the original source. Helps AI validate the claim before using it.")
                    new_tags = st.text_input("Tags (comma-separated)", help="Categorization tags for organizing claims. Separate multiple tags with commas.")
                
                new_evidence = st.text_area("Evidence Excerpt", height=80, help="Supporting evidence or source URL. Helps AI validate the claim before using it.")
                
                submit_col1, submit_col2 = st.columns([1, 4])
                with submit_col1:
                    if st.form_submit_button("Create Claim", type="primary"):
                        if new_claim_text:
                            result = api.create_claim({
                                "claim_text": new_claim_text,
                                "claim_type": new_claim_type,
                                "confidence": new_confidence,
                                "personas": new_personas,
                                "tags": [t.strip() for t in new_tags.split(",") if t.strip()],
                                "source_title": new_source_title,
                                "source_url": new_source_url,
                                "evidence_excerpt": new_evidence
                            })
                            if result.get("success"):
                                st.toast(f"Created claim {result.get('claim_id')}", icon="✅")
                                st.session_state.show_claim_form = False
                                st.rerun()
                            else:
                                st.error(f"Failed: {result.get('error')}")
                        else:
                            st.error("Claim text is required")
                with submit_col2:
                    if st.form_submit_button("Cancel"):
                        st.session_state.show_claim_form = False
                        st.rerun()
    
    claims_data = api.list_claims(
        limit=100,
        claim_type=filter_type if filter_type != "All" else None,
        confidence_min=filter_confidence,
        search=search_claims if search_claims else None
    )
    
    claims = claims_data.get("claims", [])
    
    if claims:
        st.info(f"Showing {len(claims)} of {claims_data.get('total', 0)} claims")
        
        for claim in claims:
            with st.expander(f"**{claim.get('id')}** | {claim.get('claim_text', '')[:80]}... ({claim.get('claim_type', 'N/A')})"):
                view_col1, view_col2 = st.columns([3, 1])
                
                with view_col1:
                    st.markdown(f"**Claim:** {claim.get('claim_text', '')}")
                    st.markdown(f"**Evidence:** {claim.get('evidence_excerpt', 'N/A')}")
                    if claim.get('source_url'):
                        st.markdown(f"**Source:** [{claim.get('source_title', 'Link')}]({claim.get('source_url')})")
                
                with view_col2:
                    st.metric("Confidence", f"{claim.get('confidence', 0)}/5", help="Claim confidence score. 5 = fully verified with strong evidence, 1 = unverified or anecdotal.")
                    st.caption(f"Type: {claim.get('claim_type', 'N/A')}")
                    st.caption(f"Personas: {claim.get('personas', '[]')}")
                
                btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
                with btn_col1:
                    if st.button("✏️ Edit", key=f"edit_claim_{claim.get('id')}"):
                        st.session_state[f"editing_claim_{claim.get('id')}"] = True
                        st.rerun()
                with btn_col2:
                    if st.button("🗑️ Delete", key=f"delete_claim_{claim.get('id')}"):
                        result = api.delete_claim(claim.get('id'))
                        if result.get("success"):
                            st.toast("Deleted!", icon="✅")
                            st.rerun()
                        else:
                            st.error(f"Failed: {result.get('error')}")
                
                if st.session_state.get(f"editing_claim_{claim.get('id')}"):
                    with st.form(f"edit_claim_form_{claim.get('id')}"):
                        edit_text = st.text_area("Claim Text", value=claim.get('claim_text', ''), height=100, help="The marketing claim statement. Be specific and factual.")
                        edit_confidence = st.slider("Confidence", 1, 5, int(claim.get('confidence', 3)), help="How confident we are in this claim (1-5). Higher confidence claims are used more by AI.")
                        edit_evidence = st.text_area("Evidence", value=claim.get('evidence_excerpt', ''), height=80, help="Supporting evidence or source URL. Helps AI validate the claim before using it.")
                        
                        if st.form_submit_button("Save Changes"):
                            result = api.update_claim(claim.get('id'), {
                                "claim_text": edit_text,
                                "confidence": edit_confidence,
                                "evidence_excerpt": edit_evidence
                            })
                            if result.get("success"):
                                st.toast("Updated!", icon="✅")
                                del st.session_state[f"editing_claim_{claim.get('id')}"]
                                st.rerun()
                            else:
                                st.error(f"Failed: {result.get('error')}")
    else:
        st.info("No claims found matching your filters")


with tab2:
    st.subheader("🎨 Brand Voice Configuration")
    st.caption("Define brand identity, tone, style, and content guidelines — AI follows these rules to maintain consistent brand messaging")
    
    brand_voice = api.get_brand_voice()
    
    if not brand_voice:
        st.warning("Could not load brand voice configuration")
    else:
        bv_sections = list(brand_voice.keys())
        bv_tab_names = [s.replace("_", " ").title() for s in bv_sections]
        
        section_tabs = st.tabs(bv_tab_names)
        
        for idx, section in enumerate(bv_sections):
            with section_tabs[idx]:
                section_data = brand_voice.get(section, {})
                
                st.markdown(f"### {section.replace('_', ' ').title()}")
                
                with st.expander("📝 Edit Section", expanded=False):
                    edited_json = st.text_area(
                        f"Edit {section}",
                        value=json.dumps(section_data, indent=2),
                        height=400,
                        key=f"bv_edit_{section}",
                        help="Edit this section as JSON. Changes are validated before saving. Invalid JSON will be rejected."
                    )
                    
                    if st.button(f"💾 Save {section.title()}", key=f"save_bv_{section}"):
                        try:
                            parsed = json.loads(edited_json)
                            result = api.update_brand_voice_section(section, parsed)
                            if result.get("success"):
                                st.toast(f"Section '{section}' updated!", icon="✅")
                                st.rerun()
                            else:
                                st.error(f"Failed: {result.get('error')}")
                        except json.JSONDecodeError as e:
                            st.error(f"Invalid JSON: {e}")
                
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        if isinstance(value, list):
                            st.markdown(f"**{key.replace('_', ' ').title()}:**")
                            for item in value:
                                if isinstance(item, dict):
                                    st.json(item)
                                else:
                                    st.markdown(f"- {item}")
                        elif isinstance(value, dict):
                            st.markdown(f"**{key.replace('_', ' ').title()}:**")
                            st.json(value)
                        else:
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                else:
                    st.write(section_data)


with tab3:
    st.subheader("🏢 Competitor Intelligence")
    st.caption("Track competitor features, claims, and differentiators — AI uses this to highlight your unique advantages in content")
    
    comp_ctrl1, comp_ctrl2, comp_ctrl3 = st.columns([2, 2, 1])
    
    with comp_ctrl1:
        search_comp = st.text_input("🔍 Search competitors", placeholder="Search by name or features...", key="comp_search", help="Search by company name, features, or claims. Matches anywhere in competitor profiles.")
    
    with comp_ctrl2:
        comp_data = api.list_competitors()
        categories = list(set(c.get("category", "") for c in comp_data.get("competitors", []) if c.get("category")))
        filter_category = st.selectbox("Category", ["All"] + sorted(categories), key="comp_category", help="Filter competitors by market category (e.g., Engagement & Performance, Learning & Development).")
    
    with comp_ctrl3:
        if st.button("➕ Add Competitor", type="primary", use_container_width=True):
            st.session_state.show_competitor_form = True
    
    if st.session_state.get("show_competitor_form"):
        with st.expander("➕ New Competitor", expanded=True):
            with st.form("new_competitor_form"):
                new_comp_name = st.text_input("Company Name *", help="Official company name of the competitor.")
                new_comp_category = st.text_input("Category", placeholder="e.g., Engagement & Performance", help="Market category this competitor operates in.")
                new_comp_url = st.text_input("Website URL", help="Competitor's website URL for reference.")
                new_comp_features = st.text_area("Key Features", height=80, help="Main product features and capabilities of this competitor.")
                new_comp_claims = st.text_area("Typical Claims", height=80, help="Marketing claims this competitor commonly makes. Helps AI counter-position.")
                new_comp_diff = st.text_area("Differentiators vs Us", height=80, help="How our product differs from this competitor. AI uses this for competitive positioning.")
                new_comp_risky = st.text_input("Risky Topics to Avoid", help="Topics to avoid when mentioning this competitor (e.g., ongoing litigation, sensitive comparisons).")
                
                submit_col1, submit_col2 = st.columns([1, 4])
                with submit_col1:
                    if st.form_submit_button("Create", type="primary"):
                        if new_comp_name:
                            result = api.create_competitor({
                                "name": new_comp_name,
                                "category": new_comp_category,
                                "url": new_comp_url,
                                "key_features": new_comp_features,
                                "typical_claims": new_comp_claims,
                                "differentiators_vs_us": new_comp_diff,
                                "risky_topics": new_comp_risky
                            })
                            if result.get("success"):
                                st.toast(f"Added {new_comp_name}", icon="✅")
                                st.session_state.show_competitor_form = False
                                st.rerun()
                            else:
                                st.error(f"Failed: {result.get('error')}")
                        else:
                            st.error("Company name is required")
                with submit_col2:
                    if st.form_submit_button("Cancel"):
                        st.session_state.show_competitor_form = False
                        st.rerun()
    
    competitors_data = api.list_competitors(
        category=filter_category if filter_category != "All" else None,
        search=search_comp if search_comp else None
    )
    
    competitors = competitors_data.get("competitors", [])
    
    if competitors:
        st.info(f"Showing {len(competitors)} competitors")
        
        for comp in competitors:
            with st.expander(f"**{comp.get('name')}** | {comp.get('category', 'N/A')}"):
                if comp.get('url'):
                    st.markdown(f"🔗 [{comp.get('url')}]({comp.get('url')})")
                
                st.markdown("**Key Features:**")
                st.write(comp.get('key_features', 'N/A'))
                
                st.markdown("**Typical Claims:**")
                st.write(comp.get('typical_claims', 'N/A'))
                
                st.markdown("**Our Differentiators:**")
                st.write(comp.get('differentiators_vs_us', 'N/A'))
                
                if comp.get('risky_topics'):
                    st.warning(f"⚠️ Risky Topics: {comp.get('risky_topics')}")
                
                st.caption(f"Last checked: {comp.get('last_checked', 'N/A')}")
                
                btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
                with btn_col1:
                    if st.button("✏️ Edit", key=f"edit_comp_{comp.get('name')}"):
                        st.session_state[f"editing_comp_{comp.get('name')}"] = True
                        st.rerun()
                with btn_col2:
                    if st.button("🗑️ Delete", key=f"delete_comp_{comp.get('name')}"):
                        result = api.delete_competitor(comp.get('name'))
                        if result.get("success"):
                            st.toast("Deleted!", icon="✅")
                            st.rerun()
                        else:
                            st.error(f"Failed: {result.get('error')}")
                
                if st.session_state.get(f"editing_comp_{comp.get('name')}"):
                    with st.form(f"edit_comp_form_{comp.get('name')}"):
                        edit_cat = st.text_input("Category", value=comp.get('category', ''), help="Market category this competitor operates in.")
                        edit_features = st.text_area("Key Features", value=comp.get('key_features', ''), height=80, help="Main product features and capabilities of this competitor.")
                        edit_claims = st.text_area("Typical Claims", value=comp.get('typical_claims', ''), height=80, help="Marketing claims this competitor commonly makes.")
                        edit_diff = st.text_area("Differentiators", value=comp.get('differentiators_vs_us', ''), height=80, help="How our product differs from this competitor.")
                        
                        if st.form_submit_button("Save Changes"):
                            result = api.update_competitor(comp.get('name'), {
                                "category": edit_cat,
                                "key_features": edit_features,
                                "typical_claims": edit_claims,
                                "differentiators_vs_us": edit_diff
                            })
                            if result.get("success"):
                                st.toast("Updated!", icon="✅")
                                del st.session_state[f"editing_comp_{comp.get('name')}"]
                                st.rerun()
                            else:
                                st.error(f"Failed: {result.get('error')}")
    else:
        st.info("No competitors found")


with tab4:
    st.subheader("📦 Product Catalog")
    st.caption("Manage product modules, packages, and governance rules — AI references these when describing product capabilities in content")
    
    catalog = api.get_product_catalog()
    
    if not catalog:
        st.warning("Could not load product catalog")
    else:
        prod_tab1, prod_tab2, prod_tab3, prod_tab4 = st.tabs([
            "📦 Modules",
            "💼 Packages",
            "⚖️ Governance",
            "📋 Full Catalog"
        ])
        
        with prod_tab1:
            st.markdown("### Product Modules")
            modules = catalog.get("modules", [])
            
            for module in modules:
                with st.expander(f"**{module.get('name')}** ({module.get('id')})"):
                    st.markdown(f"**Description:** {module.get('description', 'N/A')}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Features:**")
                        for feat in module.get('features', []):
                            st.markdown(f"- {feat}")
                    
                    with col2:
                        st.markdown("**KPIs:**")
                        for kpi in module.get('kpis', []):
                            st.markdown(f"- {kpi}")
                    
                    st.markdown("**Inputs:** " + ", ".join(module.get('inputs', [])))
                    st.markdown("**Outputs:** " + ", ".join(module.get('outputs', [])))
                    
                    if st.button("✏️ Edit Module", key=f"edit_mod_{module.get('id')}"):
                        st.session_state[f"editing_mod_{module.get('id')}"] = True
                        st.rerun()
                    
                    if st.session_state.get(f"editing_mod_{module.get('id')}"):
                        with st.form(f"edit_mod_form_{module.get('id')}"):
                            edited_mod = st.text_area(
                                "Module JSON",
                                value=json.dumps(module, indent=2),
                                height=400,
                                help="Edit the module definition as JSON. Includes features, KPIs, inputs, and outputs."
                            )
                            if st.form_submit_button("Save Module"):
                                try:
                                    parsed = json.loads(edited_mod)
                                    result = api.update_product_module(module.get('id'), parsed)
                                    if result.get("success"):
                                        st.toast("Module updated!", icon="✅")
                                        del st.session_state[f"editing_mod_{module.get('id')}"]
                                        st.rerun()
                                    else:
                                        st.error(f"Failed: {result.get('error')}")
                                except json.JSONDecodeError as e:
                                    st.error(f"Invalid JSON: {e}")
        
        with prod_tab2:
            st.markdown("### Product Packages")
            packages = catalog.get("packages", [])
            
            for pkg in packages:
                with st.container(border=True):
                    st.markdown(f"### {pkg.get('name')} ({pkg.get('id')})")
                    st.markdown(f"_{pkg.get('notes', '')}_")
                    
                    st.markdown("**Includes:**")
                    for mod_id in pkg.get('includes', []):
                        st.markdown(f"- ✅ {mod_id}")
                    
                    pricing = pkg.get('unit_pricing', {})
                    if pricing:
                        st.metric(
                            "Price",
                            f"€{pricing.get('amount_eur', 0)} / {pricing.get('model', 'unit').replace('_', ' ')}",
                            help="Package price per unit. Displayed in marketing materials and proposals."
                        )
        
        with prod_tab3:
            st.markdown("### Governance Rules")
            governance = catalog.get("governance", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Min Confidence to Publish", governance.get("min_confidence_to_publish", "N/A"), help="Minimum claim confidence score required before AI can include a claim in published content.")
                st.metric("Preferred Confidence", governance.get("preferred_confidence", "N/A"), help="Preferred confidence level for claims. AI prioritizes claims at or above this threshold.")
            
            with col2:
                st.markdown("**Forbidden Phrases:**")
                for phrase in governance.get("forbidden_phrases", []):
                    st.markdown(f"- ❌ {phrase}")
            
            st.markdown("**Reviewer Required For:**")
            for item in governance.get("reviewer_required_for", []):
                st.markdown(f"- 👤 {item}")
            
            with st.expander("✏️ Edit Governance Rules"):
                with st.form("edit_governance_form"):
                    edited_gov = st.text_area(
                        "Governance JSON",
                        value=json.dumps(governance, indent=2),
                        height=400,
                        help="Edit governance rules as JSON. Controls confidence thresholds, forbidden phrases, and review requirements."
                    )
                    if st.form_submit_button("Save Governance"):
                        try:
                            parsed = json.loads(edited_gov)
                            result = api.update_product_governance(parsed)
                            if result.get("success"):
                                st.toast("Governance rules updated!", icon="✅")
                                st.rerun()
                            else:
                                st.error(f"Failed: {result.get('error')}")
                        except json.JSONDecodeError as e:
                            st.error(f"Invalid JSON: {e}")
        
        with prod_tab4:
            st.markdown("### Full Product Catalog")
            st.caption("Raw JSON view of the complete catalog — for advanced users who need to edit the full structure directly")
            
            with st.expander("View/Edit Full Catalog", expanded=False):
                edited_catalog = st.text_area(
                    "Product Catalog JSON",
                    value=json.dumps(catalog, indent=2),
                    height=600,
                    help="Complete product catalog as JSON. Includes all modules, packages, and governance rules. Edit with care."
                )
                
                if st.button("💾 Save Full Catalog"):
                    try:
                        parsed = json.loads(edited_catalog)
                        result = api.update_product_catalog(parsed)
                        if result.get("success"):
                            st.toast("Catalog saved!", icon="✅")
                            st.rerun()
                        else:
                            st.error(f"Failed: {result.get('error')}")
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON: {e}")


with tab5:
    st.subheader("📋 Version History & Backup")
    st.caption("Automatic versioning protects your data — every edit creates a backup before saving, and you can restore any previous state")

    bcol1, bcol2 = st.columns([1, 3])
    with bcol1:
        if st.button("📦 Backup All Now", type="primary", use_container_width=True):
            result = api.create_data_config_backup()
            if result.get("success"):
                st.toast(f"✅ Created {len(result.get('versions', []))} backups")
                st.rerun()
            else:
                st.error(f"Backup failed: {result.get('error', 'Unknown error')}")
    with bcol2:
        filter_config = st.selectbox(
            "Filter by config",
            ["All", "claims", "brand_voice", "competitors", "products"],
            key="version_filter",
            help="Show version history for a specific config file, or 'All' to see everything."
        )

    config_filter = None if filter_config == "All" else filter_config
    versions_data = api.list_data_config_versions(config_name=config_filter)
    versions = versions_data.get("versions", [])

    if versions:
        st.info(f"📂 {len(versions)} version(s) available")
        for v in versions:
            vid = v.get("version_id", "")
            created = v.get("created_at", "")[:19].replace("T", " ")
            reason = v.get("reason", "auto")
            config = v.get("config_name", "")
            size_kb = (v.get("file_size", 0) / 1024)

            reason_icon = {"auto": "🔄", "manual_backup": "📦", "pre_restore": "⏪"}.get(reason, "📝")

            with st.expander(f"{reason_icon} {config} — {created} ({reason})", expanded=False):
                st.caption(f"Version ID: `{vid}`  |  Size: {size_kb:.1f} KB")
                rcol1, rcol2 = st.columns(2)
                with rcol1:
                    if st.button("⏪ Restore this version", key=f"restore_{vid}"):
                        result = api.restore_data_config_version(vid)
                        if result.get("success"):
                            st.toast(f"✅ Restored {config} from {created}")
                            st.rerun()
                        else:
                            st.error(f"Restore failed: {result.get('error', 'Unknown')}")
                with rcol2:
                    if st.button("🗑️ Delete backup", key=f"del_{vid}"):
                        result = api.delete_data_config_version(vid)
                        if result.get("success"):
                            st.toast("Backup deleted")
                            st.rerun()
                        else:
                            st.error(f"Delete failed: {result.get('error', 'Unknown')}")
    else:
        st.info("No version history yet. Versions are created automatically when you edit any config data.")


st.divider()

footer_col1, footer_col2, footer_col3 = st.columns([1, 1, 2])

with footer_col1:
    if st.button("🔄 Reload All Data", type="secondary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with footer_col2:
    st.caption("All data persists in `data/` directory")

with footer_col3:
    with st.expander("📁 File Locations"):
        st.code("""data/claim_library/claims.csv
data/company/brand_voice.json
data/competitors/competitors.csv
data/products/catalog.json""")
