"""
Governance & Safety - Complete HITL and Content Review System
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import re
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_client import AgenticAPIClient
from utils.data_controls import render_data_controls
from components import (
    render_metric_card, render_status_badge, render_status_card,
    create_bar_chart, create_line_chart, create_gauge_chart,
    render_copy_button, render_linkedin_copy_section, render_platform_copy_section, expand_claim_citations
)
from app_config import get_governance_thresholds, SAFETY_SCORE_TIERS


def parse_content_fields(body_text: str) -> dict:
    """
    Parse headline, CTA, and clean body from content that has them embedded.
    Many content generators output: **Headline:** X\n**Body:** Y\n**CTA:** Z
    This extracts them into separate fields.
    """
    if not body_text:
        return {"headline": None, "body": body_text, "cta": None}
    
    headline = None
    cta = None
    clean_body = body_text
    
    headline_match = re.search(
        r'(?:^|\n)\*\*Headline:?\*\*\s*(.+?)(?=\n\n|\n\*\*Body|\n\*\*CTA|\Z)', 
        body_text, 
        re.IGNORECASE | re.DOTALL
    )
    if headline_match:
        headline = headline_match.group(1).strip()
    
    # Must be **CTA:** or **CTA** at start of line, NOT [CTA Button] in body
    cta_match = re.search(
        r'(?:^|\n)\*\*CTA:?\*\*\s*(.+?)(?=\n\*\*Claims|\n\nClaims|\nClaims Used|\Z)', 
        body_text, 
        re.IGNORECASE | re.DOTALL
    )
    if cta_match:
        cta = cta_match.group(1).strip()
    
    if not cta:
        cta_patterns = [
            r'(Schedule a call[^.!]*[.!])',
            r'(Learn [Mm]ore[^.!]*[.!])',
            r'(Sign up[^.!]*[.!])',
            r'(Get [Ss]tarted[^.!]*[.!])',
            r'(Contact us[^.!]*[.!])',
            r'(Don\'t miss out[^.!]*[.!])'
        ]
        for pattern in cta_patterns:
            match = re.search(pattern, body_text, re.IGNORECASE)
            if match:
                cta = match.group(1).strip()
                break
    
    body_match = re.search(
        r'(?:^|\n)\*\*Body:?\*\*\s*(.+?)(?=\n\*\*CTA|\Z)', 
        body_text, 
        re.IGNORECASE | re.DOTALL
    )
    if body_match:
        clean_body = body_match.group(1).strip()
    else:
        clean_body = re.sub(r'\*\*Headline:?\*\*[^\n]*\n?', '', body_text)
        clean_body = re.sub(r'\*\*CTA:?\*\*.*$', '', clean_body, flags=re.DOTALL)
        clean_body = clean_body.strip()
    
    return {
        "headline": headline,
        "body": clean_body if clean_body else body_text,
        "cta": cta
    }


st.set_page_config(page_title="Governance - Agentic AI", page_icon="👁️", layout="wide")

@st.cache_resource
def get_api():
    return AgenticAPIClient()

api = get_api()

st.title("👁️ Governance & Content Safety")
st.caption("Human-in-the-loop review, safety validation, and compliance management")

with st.expander("ℹ️ Content Governance & Safety Guide", expanded=False):
    st.markdown("""
    **What is Content Governance?**
    Governance ensures all AI-generated content meets quality, safety, and brand standards before deployment.

    **Human-in-the-Loop (HITL)**
    All AI-generated content goes through human review before publishing. Reviewers can approve, reject, or request regeneration.

    **Safety Scoring**
    - **Toxicity** (0–1): Lower is safer. Measures hate speech, profanity, and harmful language.
    - **Factuality** (0–1): Higher is better. Measures accuracy of claims and statements.
    - **Brand Alignment** (0–1): Higher is better. Measures consistency with brand voice and guidelines.

    **Review Workflow**
    Content Generated → Safety Check → HITL Queue → Human Review → Approve / Reject / Regenerate → Deploy

    **Override Rate**
    Percentage of times humans reject AI-recommended content. Lower values indicate better AI alignment with human standards. Target: < 5%.

    **Best Practices**
    - Review the queue regularly to avoid content bottlenecks.
    - Provide specific feedback when rejecting content — this helps the AI improve.
    - Use "Regenerate" for content that needs more than minor edits.
    - Rejected content is marked as failed; the campaign can generate new content to replace it.
    """)


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 HITL Review Queue",
    "🛡️ Safety Validation",
    "📊 Safety Statistics",
    "🧪 Golden Tests",
    "📜 Review History",
    "📈 Override Rate",
    "⚖️ Rules & Config"
])

with tab7:
    st.subheader("⚖️ Governance Rules & Configuration")
    st.caption("Active safety thresholds and compliance policies. These rules determine which content is auto-approved, flagged for review, or auto-rejected.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🛡️ Safety Thresholds")
        st.info("Content failing these checks requires manual review")
        
        gov_thresholds = get_governance_thresholds()
        
        thresholds = {
            "Minimum Safety Score": gov_thresholds['safety_score'],
            "Maximum Toxicity": gov_thresholds['toxicity'],
            "Minimum Factuality": gov_thresholds['factuality'],
            "Brand Alignment": gov_thresholds['brand_alignment']
        }
        
        for name, value in thresholds.items():
            st.markdown(f"**{name}**")
            st.progress(value)
            st.caption(f"Threshold: {value:.2f}")

    with col2:
        st.markdown("### 🚦 Approval Policies")
        
        auto_approve = gov_thresholds['auto_approve']
        min_safety = gov_thresholds['min_safety']
        
        st.markdown(f"""
        **✅ Auto-Approval Criteria**
        - Safety Score > {auto_approve}
        - No detected toxicity
        - High brand alignment (> 0.90)
        
        **⚠️ Manual Review Required**
        - Any claims without citations
        - Safety Score between {min_safety:.2f} and {auto_approve}
        - Sensitive topics detected
        
        **❌ Auto-Rejection**
        - Safety Score < 0.60
        - High toxicity detected
        - Competitor mentions without context
        """)
        
        st.markdown("### 📝 Compliance Rules")
        st.json({
            "industry": "Software/Technology",
            "regulatory_framework": "Standard Marketing Ethics",
            "sensitive_topics": ["pricing", "guarantees", "competitors"],
            "required_disclaimers": True
        })

with tab1:
    st.subheader("📋 Human-in-the-Loop Review Queue")
    st.caption("Review and approve AI-generated content before deployment. Items are prioritized by safety score and urgency.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        queue_status = st.selectbox(
            "Queue Status",
            ["pending", "completed", "all"],
            index=0,
            key="queue_status",
            help="pending=awaiting review, completed=reviewed (approved/rejected)"
        )
    
    with col2:
        queue_limit = st.number_input(
            "Items to Load",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Number of queue items to fetch. Increase to see more items at once."
        )
    
    with col3:
        if st.button("🔄 Refresh Queue", use_container_width=True):
            st.rerun()
    
    try:
        hitl_items = api.get_hitl_queue(status=queue_status, limit=queue_limit)
        
        if hitl_items:
            st.success(f"✅ Found {len(hitl_items)} items in {queue_status} queue")
            
            filtered_items = render_data_controls(
                data=hitl_items,
                search_fields=['campaign_name', 'headline', 'body', 'content_id', 'campaign_id'],
                filter_configs=[
                    {'field': 'campaign_name', 'label': 'Campaign', 'type': 'select', 'options': 'auto'},
                    {'field': 'status', 'label': 'Status', 'type': 'select', 'options': 'auto'},
                    {'field': 'priority', 'label': 'Priority', 'type': 'select', 'options': 'auto'},
                ],
                sort_options=['created_at', 'priority', 'campaign_name', 'safety_score'],
                key_prefix="hitl_queue"
            )

            if queue_status == "pending" and filtered_items:
                low_risk_items = [item for item in filtered_items
                                  if item.get('safety_score', 0) > 0.9 or
                                     item.get('toxicity_score', 1) < 0.1]

                if low_risk_items:
                    col_batch, col_count = st.columns([2, 1])
                    with col_batch:
                        if st.button(f"✅ Approve All Low-Risk ({len(low_risk_items)} items)",
                                     type="primary", use_container_width=True):
                            approved_count = 0
                            with st.spinner("Approving low-risk items..."):
                                for item in low_risk_items:
                                    try:
                                        api.submit_review({
                                            'hitl_id': item.get('id'),
                                            'content_id': item.get('content_id'),
                                            'decision': 'approved',
                                            'reviewer_notes': 'Auto-approved (low-risk batch)'
                                        })
                                        approved_count += 1
                                    except Exception:
                                        pass
                            st.toast(f"✅ Approved {approved_count}/{len(low_risk_items)} low-risk items", icon="✅")
                            st.rerun()
                    with col_count:
                        st.caption(f"Low-risk: safety score > 0.9")

            
            for idx, item in enumerate(filtered_items):
                item_id = item.get('id', f'item_{idx}')
                content_id = item.get('content_id', 'N/A')
                priority = item.get('priority', 5)
                created_at = item.get('created_at', 'N/A')
                
                if priority >= 8:
                    priority_color = "🔴"
                    priority_label = "HIGH"
                elif priority >= 5:
                    priority_color = "🟡"
                    priority_label = "MEDIUM"
                else:
                    priority_color = "🟢"
                    priority_label = "LOW"
                
                _item_platform = item.get('platform', 'linkedin')
                _plat_icons = {"linkedin": "💼", "twitter": "🐦", "email": "📧", "blog": "📝"}
                _plat_icon = _plat_icons.get(_item_platform, "📋")
                with st.expander(
                    f"{priority_color} **Item {idx+1}** | {_plat_icon} {_item_platform.title()} | Priority: {priority_label} ({priority}) | "
                    f"Created: {created_at[:19] if created_at != 'N/A' else 'N/A'}",
                    expanded=(idx == 0 and queue_status == "pending")
                ):
                    col_content, col_safety, col_actions = st.columns([2, 1, 1])
                    
                    with col_content:
                        st.markdown("**Content Details**")
                        
                        campaign_id = item.get('campaign_id', 'N/A')
                        campaign_name = item.get('campaign_name', 'Unknown Campaign')
                        item_platform = item.get('platform', 'linkedin')
                        platform_icons = {"linkedin": "💼", "twitter": "🐦", "email": "📧", "blog": "📝"}
                        platform_names = {"linkedin": "LinkedIn", "twitter": "Twitter/X", "email": "Email", "blog": "Blog"}
                        st.markdown(f"**Campaign:** {campaign_name}")
                        st.markdown(f"**Platform:** {platform_icons.get(item_platform, '📋')} {platform_names.get(item_platform, item_platform.title())}")
                        st.markdown(f"**Campaign ID:** `{campaign_id}`")
                        st.markdown("---")
                        
                        st.markdown(f"**Content ID:** `{content_id}`")
                        st.markdown(f"**Queue Item ID:** `{item_id}`")

                        raw_headline = item.get('headline')
                        raw_body = item.get('body', '')
                        raw_cta = item.get('cta')
                        claims = item.get('claims_used', [])
                        
                        if (not raw_headline or raw_headline == 'N/A') and raw_body:
                            parsed = parse_content_fields(raw_body)
                            headline = parsed['headline'] or 'N/A'
                            body = parsed['body'] or raw_body
                            cta = parsed['cta'] or 'N/A'
                        else:
                            headline = raw_headline or 'N/A'
                            body = raw_body
                            cta = raw_cta or 'N/A'

                        st.markdown(f"**Headline:** {headline}")
                        st.markdown(f"**CTA:** {cta}")

                        st.markdown("---")
                        st.markdown("**📄 Content Body:**")
                        st.text_area("Content Body", body, height=150, disabled=True, key=f"body_{item_id}", label_visibility="collapsed")

                        st.markdown("---")
                        render_platform_copy_section(
                            headline=headline if headline != 'N/A' else "",
                            body=body,
                            cta=cta if cta != 'N/A' else "",
                            platform=item_platform,
                            key=f"platform_{item_id}"
                        )

                        if claims:
                            st.markdown("---")
                            st.markdown("**📌 Claim Citations:**")
                            for claim_idx, claim in enumerate(claims):
                                if isinstance(claim, dict):
                                    claim_text = claim.get('text', str(claim))
                                    claim_source = claim.get('source', 'Unknown')
                                    verified = claim.get('verified', True)
                                else:
                                    claim_text = str(claim)
                                    claim_source = "Claim Library"
                                    verified = True

                                if verified:
                                    st.markdown(f"✅ **Claim {claim_idx+1}:** {claim_text}")
                                    st.caption(f"   Source: {claim_source}")
                                else:
                                    st.markdown(f"⚠️ **Claim {claim_idx+1}:** {claim_text}")
                                    st.caption(f"   Source: {claim_source} (UNVERIFIED)")
                        else:
                            st.info("No claims cited in this content")

                    
                    with col_safety:
                        st.markdown("**Safety Scores**")

                        overall = item.get('safety_score')
                        toxicity = item.get('toxicity_score')
                        factuality = item.get('factuality_score')
                        brand_alignment = item.get('brand_score')

                        has_individual_scores = any(s is not None for s in [toxicity, factuality, brand_alignment])

                        if overall is not None and overall > 0:
                            fig = create_gauge_chart(
                                value=overall,
                                max_value=1.0,
                                title="Overall Safety"
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"gov_safety_radar_{item_id}")
                        elif overall is not None:
                            st.metric("Overall Safety", f"{overall:.2%}", help="Composite safety score (0–1). Higher is safer.")
                        else:
                            st.metric("Overall Safety", "N/A", help="Composite safety score (0–1). Higher is safer.")

                        if has_individual_scores:
                            if toxicity is not None:
                                st.metric("Toxicity", f"{toxicity:.2f}", help="Toxicity score (0–1). Lower is safer. Measures harmful language.")
                            else:
                                st.metric("Toxicity", "N/A", help="Toxicity score (0–1). Lower is safer.")

                            if factuality is not None:
                                st.metric("Factuality", f"{factuality:.2f}", help="Factuality score (0–1). Higher means more accurate claims.")
                            else:
                                st.metric("Factuality", "N/A", help="Factuality score (0–1). Higher means more accurate claims.")

                            if brand_alignment is not None:
                                st.metric("Brand", f"{brand_alignment:.2f}", help="Brand alignment (0–1). Higher means better fit with brand voice.")
                            else:
                                st.metric("Brand", "N/A", help="Brand alignment (0–1). Higher means better fit with brand voice.")

                        reason = item.get('reason', '')
                        if reason:
                            st.caption(f"📝 {reason}")
                    
                    with col_actions:
                        st.markdown("**Actions**")

                        if queue_status == "pending":
                            overall_safety = item.get('safety_score', 0) or 0

                            if overall_safety < 0.7 and overall_safety > 0:
                                st.warning(f"⚠️ Low Safety Score: {overall_safety:.2f} - Consider regenerating content")

                            with st.form(f"review_form_{item_id}", clear_on_submit=True):
                                feedback = st.text_area(
                                    "Feedback (optional)",
                                    placeholder="Provide feedback on this content...",
                                    key=f"feedback_{item_id}",
                                    help="Specific feedback helps the AI improve. Mention what's wrong and how to fix it."
                                )

                                reviewer_email = st.text_input(
                                    "Your Email",
                                    value="reviewer@example.com",
                                    key=f"email_{item_id}",
                                    help="Email of the reviewer. Used for audit trail and accountability."
                                )

                                st.markdown("**Choose Action:**")

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    approve_btn = st.form_submit_button(
                                        "✅ Approve",
                                        type="primary",
                                        use_container_width=True
                                    )

                                with col2:
                                    reject_btn = st.form_submit_button(
                                        "❌ Reject",
                                        type="secondary",
                                        use_container_width=True
                                    )

                                with col3:
                                    regenerate_btn = st.form_submit_button(
                                        "🔄 Regenerate",
                                        type="secondary",
                                        use_container_width=True
                                    )

                                if approve_btn:
                                    decision = "approve"
                                    submit_review = True
                                elif reject_btn:
                                    decision = "reject"
                                    submit_review = True
                                elif regenerate_btn:
                                    decision = "regenerate"
                                    submit_review = True
                                else:
                                    submit_review = False

                                if submit_review:
                                    try:
                                        if decision == "regenerate":
                                            with st.spinner("🔄 Regenerating content... This may take up to 2 minutes. Please wait..."):
                                                result = api.regenerate_content(
                                                    content_id=content_id,
                                                    feedback=feedback or "Safety score too low - regenerating"
                                                )

                                            if result and result.get('status') == 'success':
                                                st.toast(f"🔄 Content regenerated successfully! New content will appear in the queue.", icon="✅")
                                                st.rerun()
                                            else:
                                                st.error(f"Failed to regenerate: {result.get('message', 'Unknown error')}")
                                        else:
                                            review_data = {
                                                "queue_item_id": item_id,
                                                "content_id": content_id,
                                                "decision": decision,  # Already lowercase from button handler
                                                "feedback": feedback,
                                                "reviewer_email": reviewer_email
                                            }

                                            result = api.submit_review(review_data)

                                            if result:
                                                if decision == "approve":
                                                    st.toast(f"✅ Content approved!", icon="✅")
                                                else:
                                                    st.info(f"❌ Content rejected")

                                                st.rerun()
                                            else:
                                                st.error("Failed to submit review")

                                    except Exception as e:
                                        st.error(f"Error submitting review: {str(e)}")
                        
                        else:
                            item_status = item.get('status', 'unknown').upper()
                            st.info(f"Status: {item_status}")

                            reviewer = item.get('reviewed_by')
                            reviewed_at = item.get('reviewed_at')
                            decision = item.get('decision')
                            feedback_text = item.get('feedback')

                            if reviewer:
                                st.write(f"**Reviewer:** {reviewer}")
                            else:
                                st.write(f"**Reviewer:** N/A")

                            if reviewed_at:
                                st.write(f"**Reviewed:** {reviewed_at[:19]}")
                            else:
                                st.write(f"**Reviewed:** N/A")

                            if decision and decision != 'N/A':
                                decision_emoji = "✅" if decision == "approve" else "❌" if decision == "reject" else "🔄"
                                st.write(f"**Decision:** {decision_emoji} {decision.upper()}")

                            if feedback_text:
                                st.markdown("**Feedback:**")
                                st.info(feedback_text)
        
        else:
            st.info(f"📭 No items in {queue_status} queue")
            
            if queue_status == "pending":
                st.success("🎉 All content reviewed! Great job!")
    
    except Exception as e:
        st.error(f"Failed to load HITL queue: {str(e)}")

with tab2:
    st.subheader("🛡️ Content Safety Validation")
    st.caption("Test content safety scores before adding to the review queue. Paste any content to see how it scores.")
    
    st.markdown("""
    **Safety Validation Checks:**
    - ✅ Toxicity detection (hate speech, profanity)
    - ✅ Factual accuracy verification
    - ✅ Brand alignment assessment
    - ✅ Compliance with guidelines
    """)
    
    with st.form("safety_validation_form"):
        st.markdown("#### Test Content Safety")
        
        test_headline = st.text_input(
            "Headline",
            placeholder="Enter headline to test...",
            help="The headline of the content to validate for safety and brand alignment."
        )
        
        test_body = st.text_area(
            "Body Content",
            placeholder="Enter body content with claim citations like [CLM_003]...\n\nExample: Our platform delivers results [CLM_003] with seamless integration [CLM_032].",
            height=150,
            help="Include claim citations in [CLM_XXX] format. Claims will be auto-extracted and validated against the claim library."
        )
        
        validate_button = st.form_submit_button("🔍 Validate Content", type="primary", use_container_width=True)
        
        if validate_button:
            if not test_headline or not test_body:
                st.error("Both headline and body are required")
            else:
                try:
                    with st.spinner("Validating content (this may take up to 60 seconds)..."):
                        import re
                        found_claims = re.findall(r'\[([A-Z0-9_]+)\]', test_body)
                        
                        validation_data = {
                            "headline": test_headline,
                            "body": test_body,
                            "claims_used": list(set(found_claims))
                        }

                        result = api.validate_content(validation_data)

                    if result and result.get('overall_score') is not None:
                        st.success("✅ Validation completed!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        overall = result.get('overall_score', 0)
                        toxicity = result.get('toxicity_score', 0)
                        factuality = result.get('factuality_score', 0)
                        brand = result.get('brand_alignment_score', 0)
                        
                        with col1:
                            st.metric("Overall Score", f"{overall:.2f}", help="Composite safety score (0–1). Higher is safer.")
                        
                        with col2:
                            color = "normal" if toxicity >= 0.8 else "inverse"
                            st.metric("Toxicity", f"{toxicity:.2f}", delta_color=color, help="Toxicity score (0–1). Lower is safer.")
                        
                        with col3:
                            color = "normal" if factuality >= 0.8 else "inverse"
                            st.metric("Factuality", f"{factuality:.2f}", delta_color=color, help="Factuality score (0–1). Higher means more accurate.")
                        
                        with col4:
                            color = "normal" if brand >= 0.8 else "inverse"
                            st.metric("Brand Alignment", f"{brand:.2f}", delta_color=color, help="Brand alignment (0–1). Higher means better brand fit.")
                        
                        st.markdown("---")
                        
                        passed = result.get('passed', False)
                        requires_review = result.get('requires_review', True)
                        
                        if passed and not requires_review:
                            st.success("✅ **PASSED** - Content meets all safety criteria")
                        elif passed and requires_review:
                            st.warning("⚠️ **PASSED WITH REVIEW** - Content is safe but requires human review")
                        else:
                            st.error("❌ **FAILED** - Content does not meet safety criteria")
                        
                        issues = result.get('issues', [])
                        if issues:
                            st.markdown("**Issues Found:**")
                            for issue in issues:
                                st.markdown(f"- ⚠️ {issue}")
                        
                        recommendations = result.get('recommendations', [])
                        if recommendations:
                            st.markdown("**Recommendations:**")
                            for rec in recommendations:
                                st.markdown(f"- 💡 {rec}")
                
                except Exception as e:
                    st.error(f"Validation failed: {str(e)}")

with tab3:
    st.subheader("📊 Safety Statistics & Trends")
    st.caption("Aggregated safety metrics and score distributions across all reviewed content.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        stats_days = st.selectbox("Time Period", [7, 14, 30, 60, 90], index=2, key="stats_days", help="Number of days to include in the statistics.")
    
    with col2:
        if st.button("🔄 Refresh Stats", use_container_width=True):
            st.rerun()
    
    try:
        safety_stats = api.get_safety_stats(days=stats_days)
        
        st.markdown("#### Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            render_metric_card("Total Content", safety_stats.get('total_content', 0), help_text="Total content items generated in the selected time period.")
        
        with col2:
            approved = safety_stats.get('approved', 0)
            render_metric_card("Approved", approved, delta=f"+{approved}", help_text="Content approved by human reviewers and cleared for deployment.")
        
        with col3:
            rejected = safety_stats.get('rejected', 0)
            render_metric_card("Rejected", rejected, delta=f"-{rejected}", delta_color="inverse", help_text="Content rejected by reviewers. Rejected content is marked as failed.")
        
        with col4:
            pending = safety_stats.get('pending_review', 0)
            render_metric_card("Pending", pending, help_text="Content awaiting human review in the HITL queue.")
        
        with col5:
            approval_rate = safety_stats.get('approval_rate', 0)
            render_metric_card("Approval Rate", f"{approval_rate:.1f}", suffix="%", help_text="Percentage of reviewed content that was approved. Higher is better.")
        
        st.markdown("---")
        
        st.markdown("#### Average Safety Scores")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_safety = safety_stats.get('average_safety_score', 0)
            fig = create_gauge_chart(avg_safety, 1.0, "Overall Safety")
            st.plotly_chart(fig, use_container_width=True, key="gov_safety_dist")
        
        with col2:
            avg_toxicity = safety_stats.get('average_toxicity_score', 0)
            fig = create_gauge_chart(avg_toxicity, 1.0, "Toxicity")
            st.plotly_chart(fig, use_container_width=True, key="gov_toxicity_dist")
        
        with col3:
            avg_factuality = safety_stats.get('average_factuality_score', 0)
            fig = create_gauge_chart(avg_factuality, 1.0, "Factuality")
            st.plotly_chart(fig, use_container_width=True, key="gov_factuality_dist")
        
        with col4:
            high_risk = safety_stats.get('high_risk_content', 0)
            st.metric("High Risk Items", high_risk, help="Content with safety scores below the minimum threshold.")
            if high_risk > 0:
                st.warning(f"⚠️ {high_risk} items flagged")
            else:
                st.success("✅ No high risk items")
        
        st.markdown("---")
        st.markdown("#### Content Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            approved_count = safety_stats.get('approved', 0)
            rejected_count = safety_stats.get('rejected', 0)
            pending_count = safety_stats.get('pending_review', 0)

            status_data = []
            if approved_count > 0:
                status_data.append({'Status': 'Approved', 'Count': approved_count})
            if rejected_count > 0:
                status_data.append({'Status': 'Rejected', 'Count': rejected_count})
            if pending_count > 0:
                status_data.append({'Status': 'Pending', 'Count': pending_count})

            if status_data:
                import plotly.graph_objects as go
                
                # Use explicit values with go.Pie to prevent data interpretation issues
                statuses = [d['Status'] for d in status_data]
                counts = [d['Count'] for d in status_data]
                colors = {'Approved': '#10b981', 'Rejected': '#ef4444', 'Pending': '#f59e0b'}
                marker_colors = [colors.get(s, '#6b7280') for s in statuses]
                
                fig = go.Figure(data=[go.Pie(
                    labels=statuses,
                    values=counts,
                    marker=dict(colors=marker_colors),
                    textinfo='percent+label',
                    hovertemplate='%{label}: %{value} items<extra></extra>'
                )])
                fig.update_layout(
                    title="Content Status Distribution",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True, key="gov_compliance_timeline")
            else:
                st.info("No content data available")
        
        with col2:
            try:
                all_content_data = api.list_contents(limit=1000)

                if all_content_data:
                    low_count = sum(1 for c in all_content_data if c.get('safety_score') is not None and 0 <= c.get('safety_score') < 0.5)
                    medium_count = sum(1 for c in all_content_data if c.get('safety_score') is not None and 0.5 <= c.get('safety_score') < 0.7)
                    good_count = sum(1 for c in all_content_data if c.get('safety_score') is not None and 0.7 <= c.get('safety_score') < 0.85)
                    excellent_count = sum(1 for c in all_content_data if c.get('safety_score') is not None and 0.85 <= c.get('safety_score') <= 1.0)

                    score_ranges = pd.DataFrame({
                        'Range': ['0.0-0.5 (Low)', '0.5-0.7 (Medium)', '0.7-0.85 (Good)', '0.85-1.0 (Excellent)'],
                        'Count': [low_count, medium_count, good_count, excellent_count]
                    })

                    score_ranges = score_ranges[score_ranges['Count'] > 0].reset_index(drop=True)

                    if len(score_ranges) == 0:
                        score_ranges = pd.DataFrame({
                            'Range': ['No Data'],
                            'Count': [0]
                        })
                else:
                    score_ranges = pd.DataFrame({
                        'Range': ['No Data'],
                        'Count': [0]
                    })
            except Exception as e:
                st.error(f"Error calculating distribution: {str(e)}")
                score_ranges = pd.DataFrame({
                    'Range': ['No Data'],
                    'Count': [0]
                })

            if len(score_ranges) > 0 and score_ranges['Count'].max() > 0:
                import plotly.graph_objects as go

                ranges = score_ranges['Range'].tolist()
                counts = score_ranges['Count'].tolist()

                fig = go.Figure(data=[
                    go.Bar(
                        x=ranges,
                        y=counts,
                        text=[f'{c}' for c in counts],
                        textposition='outside',
                        marker=dict(color='lightblue')
                    )
                ])

                max_count = max(counts)
                fig.update_layout(
                    title="Safety Score Distribution",
                    xaxis_title="Range",
                    yaxis_title="Count",
                    yaxis=dict(range=[0, max_count * 1.3]),
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="gov_brand_alignment")
            else:
                st.info("No safety score data available")
    
    except Exception as e:
        st.error(f"Failed to load safety statistics: {str(e)}")

with tab4:
    st.subheader("🧪 Golden Test Suite")
    st.caption("Mandatory test suite for production deployment validation. 100% pass rate required before deploying.")

    try:
        test_results = api.get_golden_test_results()
        if test_results:
            pass_rate_raw = test_results.get('pass_rate', 0)
            pass_rate = pass_rate_raw * 100 if pass_rate_raw <= 1 else pass_rate_raw
            total_tests = test_results.get('total_tests', 0)
            passed_tests = test_results.get('passed_tests', 0)
            failed_tests = test_results.get('failed_tests', 0)

            col_gauge, col_status = st.columns([2, 1])

            with col_gauge:
                fig_pass_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pass_rate,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Golden Test Pass Rate", 'font': {'size': 20}},
                    number={'suffix': '%', 'font': {'size': 40}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#22c55e" if pass_rate >= 100 else "#ef4444"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 95], 'color': '#fee2e2'},
                            {'range': [95, 99], 'color': '#fef3c7'},
                            {'range': [99, 100], 'color': '#d1fae5'}
                        ],
                        'threshold': {
                            'line': {'color': "green", 'width': 4},
                            'thickness': 0.75,
                            'value': 100
                        }
                    }
                ))
                fig_pass_gauge.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_pass_gauge, use_container_width=True, key="gov_pass_rate_gauge")

            with col_status:
                st.markdown("### Deployment Status")
                if pass_rate >= 100:
                    st.success("✅ **READY FOR PRODUCTION**")
                    st.markdown(f"**{passed_tests}/{total_tests}** tests passing")
                elif pass_rate >= 95:
                    st.warning("⚠️ **REVIEW REQUIRED**")
                    st.markdown(f"**{failed_tests}** tests failing")
                else:
                    st.error("❌ **BLOCKED**")
                    st.markdown(f"**{failed_tests}** tests failing")
                st.caption("Target: 100% pass rate")

            st.markdown("---")
    except Exception as e:
        st.warning(f"Could not load test results for gauge: {e}")

    st.markdown("""
    **Golden Test Requirements:**
    - ✅ 40 comprehensive test cases
    - ✅ 100% pass rate required for production
    - ✅ Validates claim citations, safety, persona matching, compliance
    """)

    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Latest Test Results")
    
    with col2:
        if st.button("▶️ Run Golden Tests", type="primary", use_container_width=True):
            with st.spinner("Running golden test suite... (this may take 2-3 minutes)"):
                try:
                    result = api.run_golden_tests()
                    
                    if result.get('status') == 'completed':
                        st.toast("✅ Golden tests completed!", icon="✅")
                        st.rerun()
                    else:
                        st.error(f"Test run failed: {result.get('message', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"Failed to run tests: {str(e)}")
    
    try:
        test_results = api.get_golden_test_results()
        
        if test_results:
            col1, col2, col3, col4 = st.columns(4)
            
            total_tests = test_results.get('total_tests', 0)
            passed_tests = test_results.get('passed_tests', 0)
            failed_tests = test_results.get('failed_tests', 0)
            # pass_rate is stored as a decimal (0.0-1.0), convert to percentage (0-100)
            pass_rate_raw = test_results.get('pass_rate', 0)
            pass_rate = pass_rate_raw * 100 if pass_rate_raw <= 1 else pass_rate_raw
            
            with col1:
                render_metric_card("Total Tests", total_tests, help_text="Total golden test cases in the suite.")
            
            with col2:
                render_metric_card("Passed", passed_tests, delta=f"+{passed_tests}", delta_color="normal", help_text="Tests that passed all validation criteria.")
            
            with col3:
                render_metric_card("Failed", failed_tests, delta=f"-{failed_tests}" if failed_tests > 0 else "0", delta_color="inverse", help_text="Tests that failed. All must pass before production deployment.")
            
            with col4:
                pass_color = "normal" if pass_rate >= 100 else "inverse"
                render_metric_card("Pass Rate", f"{pass_rate:.1f}", suffix="%", delta_color=pass_color, help_text="Percentage of tests passing. 100% required for production.")
            
            st.markdown("---")
            
            if pass_rate >= 100:
                st.success("🎉 **ALL TESTS PASSED** - System ready for production deployment")
            elif pass_rate >= 95:
                st.warning(f"⚠️ **{failed_tests} tests failing** - Review and fix before deployment")
            else:
                st.error(f"❌ **{failed_tests} tests failing** - System not ready for production")
            
            last_run = test_results.get('last_run')
            if last_run:
                st.caption(f"Last run: {last_run}")
            
            st.markdown("---")
            st.markdown("#### Test Details")
            
            test_details = test_results.get('test_details', [])
            if test_details:
                test_df = pd.DataFrame(test_details)
                
                def color_status(val):
                    if val == 'PASSED':
                        return 'background-color: #d1fae5'
                    elif val == 'FAILED':
                        return 'background-color: #fee2e2'
                    else:
                        return ''
                
                if 'status' in test_df.columns:
                    styled_df = test_df.style.map(color_status, subset=['status'])
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(test_df, use_container_width=True, hide_index=True)
            else:
                st.info("No detailed test results available")
        
        else:
            st.info("📋 No test results found. Run the golden test suite to see results.")
    
    except Exception as e:
        st.error(f"Failed to load golden test results: {str(e)}")

with tab5:
    st.subheader("📜 Review History")
    st.caption("Historical record of all content reviews. Use this to audit past decisions and identify patterns.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        history_days = st.selectbox("Time Period", [7, 14, 30, 60, 90], index=2, key="history_days", help="Number of days of review history to display.")
    
    with col2:
        history_limit = st.number_input("Items to Load", min_value=10, max_value=200, value=50, step=10, key="history_limit", help="Maximum number of review records to load.")
    
    with col3:
        if st.button("🔄 Refresh History", use_container_width=True):
            st.rerun()
    
    try:
        review_history = api.get_review_history(limit=history_limit, days=history_days)

        if review_history:
            st.success(f"✅ Loaded {len(review_history)} review records")

            for idx, review in enumerate(review_history):
                decision = review.get('decision', 'N/A').upper()

                if decision == 'APPROVE':
                    decision_emoji = "✅"
                    decision_color = "green"
                elif decision == 'REJECT':
                    decision_emoji = "❌"
                    decision_color = "red"
                else:
                    decision_emoji = "🔄"
                    decision_color = "orange"

                raw_headline = review.get('headline')
                raw_body = review.get('body', '')
                
                if (not raw_headline or raw_headline == 'N/A') and raw_body:
                    parsed = parse_content_fields(raw_body)
                    display_headline = parsed['headline'] or 'No headline'
                else:
                    display_headline = raw_headline or 'No headline'
                display_body = raw_body

                with st.expander(
                    f"{decision_emoji} **{decision}** | {display_headline} | {review.get('reviewed_at', 'N/A')[:16]}",
                    expanded=False
                ):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown("**Content Details**")
                        campaign_id = review.get('campaign_id', 'N/A')
                        campaign_name = review.get('campaign_name', 'Unknown Campaign')
                        review_platform = review.get('platform', 'linkedin')
                        _rp_icons = {"linkedin": "💼", "twitter": "🐦", "email": "📧", "blog": "📝"}
                        _rp_names = {"linkedin": "LinkedIn", "twitter": "Twitter/X", "email": "Email", "blog": "Blog"}
                        st.write(f"**Campaign:** {campaign_name}")
                        st.write(f"**Platform:** {_rp_icons.get(review_platform, '📋')} {_rp_names.get(review_platform, review_platform.title())}")
                        st.write(f"**Campaign ID:** `{campaign_id}`")
                        st.markdown("---")
                        st.write(f"**Headline:** {display_headline}")
                        st.markdown("**Body:**")
                        st.text_area("Body Content", display_body, height=120, disabled=True, key=f"review_body_{review.get('id', idx)}", label_visibility="collapsed")
                        st.write(f"**Type:** {review.get('content_type', 'N/A')}")
                        st.write(f"**Status:** {review.get('status', 'N/A')}")

                    with col2:
                        st.markdown("**Review Information**")
                        st.write(f"**Decision:** {decision}")
                        st.write(f"**Reviewed By:** {review.get('reviewed_by', 'N/A')}")
                        st.write(f"**Date:** {review.get('reviewed_at', 'N/A')[:16]}")
                        st.write(f"**Priority:** {review.get('priority', 'N/A')}")

                    st.markdown("**Safety Scores**")
                    score_col1, score_col2, score_col3, score_col4 = st.columns(4)

                    with score_col1:
                        safety = review.get('safety_score')
                        st.metric("Safety", f"{safety:.2f}" if safety is not None else "N/A", help="Overall safety score at time of review.")

                    with score_col2:
                        toxicity = review.get('toxicity_score')
                        st.metric("Toxicity", f"{toxicity:.2f}" if toxicity is not None else "N/A", help="Toxicity score (0–1). Lower is safer.")

                    with score_col3:
                        factuality = review.get('factuality_score')
                        st.metric("Factuality", f"{factuality:.2f}" if factuality is not None else "N/A", help="Factuality score (0–1). Higher is better.")

                    with score_col4:
                        brand = review.get('brand_alignment_score')
                        st.metric("Brand", f"{brand:.2f}" if brand is not None else "N/A", help="Brand alignment (0–1). Higher is better.")

                    feedback = review.get('feedback') or review.get('review_notes')
                    if feedback:
                        st.markdown("**Human Feedback**")
                        st.info(feedback)
                    else:
                        st.caption("_No feedback provided_")

                    st.markdown("---")
                    st.caption(f"Campaign ID: {review.get('campaign_id', 'N/A')} | Content ID: {review.get('content_id', 'N/A')}")

            st.markdown("---")
            st.markdown("#### Summary Table")
            history_data = []
            for review in review_history:
                raw_headline = review.get('headline')
                raw_body = review.get('body', '')
                if (not raw_headline or raw_headline == 'N/A') and raw_body:
                    parsed = parse_content_fields(raw_body)
                    headline = parsed['headline'] or 'No headline'
                else:
                    headline = raw_headline or 'No headline'
                    
                history_data.append({
                    'Date': review.get('reviewed_at', 'N/A')[:16],
                    'Headline': headline,
                    'Decision': review.get('decision', 'N/A').upper(),
                    'Safety': f"{review.get('safety_score', 0):.2f}",
                    'Reviewer': review.get('reviewed_by', 'N/A'),
                    'Feedback': '✅' if (review.get('feedback') or review.get('review_notes')) else '❌'
                })

            if history_data:
                df = pd.DataFrame(history_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                if st.button("📥 Export to CSV"):
                    from components import export_to_csv
                    export_to_csv(df, f"review_history_{datetime.now().strftime('%Y%m%d')}.csv")
                
                st.markdown("---")
                st.markdown("#### Review Statistics")

                approved_count = len([r for r in review_history if r.get('decision') == 'approve'])
                rejected_count = len([r for r in review_history if r.get('decision') == 'reject'])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Reviews", len(review_history), help="Total number of reviews in the selected time period.")
                
                with col2:
                    st.metric("Approved", approved_count, help="Number of content items approved by human reviewers.")
                
                with col3:
                    st.metric("Rejected", rejected_count, help="Number of content items rejected by human reviewers.")
        
        else:
            st.info("📭 No review history found for the selected period")
    
    except Exception as e:
        st.error(f"Failed to load review history: {str(e)}")

with tab6:
    st.subheader("📈 Human Override Rate")
    st.caption("Track percentage of AI-generated content rejected by humans (Target: < 5%). Lower override rates indicate better AI–human alignment.")

    try:
        override_data = api.get_override_rate()
        
        st.markdown("### Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        current_rate = override_data.get('current_rate', 0)
        target_rate = override_data.get('target_rate', 5.0)
        total_reviews = override_data.get('total_reviews', 0)
        total_overrides = override_data.get('total_overrides', 0)
        passes_target = override_data.get('passes_target', True)
        
        with col1:
            delta = target_rate - current_rate
            delta_color = "normal" if passes_target else "inverse"
            st.metric(
                "Current Override Rate",
                f"{current_rate:.1f}%",
                delta=f"{delta:+.1f}% from target",
                delta_color=delta_color,
                help="Percentage of AI content rejected by humans. Calculated as (rejections / total reviews) × 100."
            )
        
        with col2:
            st.metric("Target Rate", f"< {target_rate:.0f}%", help="Maximum acceptable override rate. Content quality should keep rejections below this.")
        
        with col3:
            st.metric("Total Reviews", f"{total_reviews:,}", help="Total number of human reviews performed.")
        
        with col4:
            st.metric("Total Overrides", f"{total_overrides:,}", help="Number of times humans rejected AI-generated content.")
        
        st.markdown("---")
        if passes_target:
            st.success(f"✅ **PASSING** - Override rate ({current_rate:.1f}%) is below target ({target_rate:.0f}%)")
            st.balloons()
        else:
            st.error(f"❌ **FAILING** - Override rate ({current_rate:.1f}%) exceeds target ({target_rate:.0f}%)")
            st.warning("**Recommended Actions:**")
            st.markdown("""
            - Review content generation prompts for quality improvements
            - Analyze rejection reasons to identify patterns
            - Consider increasing safety thresholds
            - Improve claim library coverage
            """)
        
        st.markdown("---")
        st.markdown("### Override Rate Gauge")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_rate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Override Rate (%)"},
                delta={'reference': target_rate, 'decreasing': {'color': "green"}, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [0, 20], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 5], 'color': '#22c55e'},
                        {'range': [5, 10], 'color': '#f59e0b'},
                        {'range': [10, 20], 'color': '#ef4444'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': target_rate
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key="gov_audit_timeline")
        
        with col2:
            trend_data = api.get_override_rate_trend(days=30)
            trend_points = trend_data.get('trend_data', [])
            
            if trend_points:
                trend_df = pd.DataFrame(trend_points)
                if 'date' in trend_df.columns and 'rate' in trend_df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trend_df['date'].tolist(),
                        y=trend_df['rate'].tolist(),
                        mode='lines+markers',
                        name='Override Rate',
                        line=dict(color='steelblue', width=2)
                    ))
                    fig.add_hline(y=target_rate, line_dash="dash", line_color="red", 
                                  annotation_text=f"Target: {target_rate}%")
                    fig.update_layout(
                        title="Override Rate Trend (30 Days)",
                        xaxis_title="Date",
                        yaxis_title="Override Rate (%)",
                        height=300,
                        yaxis=dict(rangemode='tozero')
                    )
                    st.plotly_chart(fig, use_container_width=True, key="gov_audit_action_dist")
            else:
                st.info("No trend data available. Run more reviews to see trend.")
        
        st.markdown("---")
        st.markdown("### Override Breakdown")
        
        st.info("""
        **Research Plan Reference:** Section 10.2 - "Human Override Rate: Percentage of content 
        routed to HITL that is rejected by the human. Target: < 5%"
        
        This metric measures the quality of AI-generated content by tracking how often human 
        reviewers override (reject) the AI's output. A lower rate indicates better AI performance 
        and alignment with brand guidelines.
        """)
        
    except Exception as e:
        st.error(f"Failed to load override rate data: {str(e)}")

st.markdown("---")
st.caption(f"Governance & Safety | Last updated: {datetime.now().strftime('%H:%M:%S')}")
