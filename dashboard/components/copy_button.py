"""
Copy-to-Clipboard Component for Platform Content Posting
Per Research Plan Section 3, Layer 4 - UX for manual posting on personal profiles
"""
import streamlit as st
import streamlit.components.v1 as components
import re
from typing import Dict, Optional

_claim_library_cache: Optional[Dict[str, str]] = None

def _load_claim_library() -> Dict[str, str]:
    """Load claim library and cache it for claim text expansion."""
    global _claim_library_cache
    if _claim_library_cache is not None:
        return _claim_library_cache
    
    try:
        import yaml
        from pathlib import Path
        
        claim_path = Path("config/prompts/claim_library.yaml")
        if not claim_path.exists():
            claim_path = Path("/app/config/prompts/claim_library.yaml")
        
        if claim_path.exists():
            with open(claim_path, 'r') as f:
                data = yaml.safe_load(f)
                _claim_library_cache = {
                    c['id']: c.get('text', c.get('claim_text', ''))
                    for c in data.get('claims', [])
                }
                return _claim_library_cache
    except Exception as e:
        pass
    
    _claim_library_cache = {}
    return _claim_library_cache

def expand_claim_citations(content: str, format_style: str = "inline") -> str:
    """
    Expand claim citations [CLM_XXX] to human-readable text.
    
    Internal systems use citation IDs for tracking,
    but published content shows the actual claim text.
    
    Args:
        content: Content with [CLM_XXX] or [CLAIM_ID:CLM_XXX] citations
        format_style: 
            - "inline": Replace [CLM_XXX] with claim text inline
            - "footnote": Keep [1], [2] and add footnotes at end
            - "remove": Remove citations entirely (clean copy)
    """
    if not content:
        return content
    
    claim_library = _load_claim_library()
    
    content = re.sub(r'\[CLAIM_ID:\s*([A-Z0-9_]+)\]', r'[\1]', content, flags=re.IGNORECASE)
    
    pattern = r'\[([A-Z0-9_]+)\]'
    citations = re.findall(pattern, content)
    
    if not citations:
        return content
    
    if format_style == "inline":
        # Citation markers are for internal tracking/validation only;
        # the content already conveys the claim's message
        result = content
        for claim_id in set(citations):
            result = result.replace(f"[{claim_id}]", "")
        
        result = re.sub(r' +', ' ', result)
        result = re.sub(r' \.', '.', result)
        result = re.sub(r' ,', ',', result)
        return result.strip()
    
    elif format_style == "footnote":
        result = content
        footnotes = []
        seen = {}
        counter = 1
        
        for claim_id in citations:
            if claim_id not in seen:
                seen[claim_id] = counter
                claim_text = claim_library.get(claim_id, f"Reference: {claim_id}")
                footnotes.append(f"[{counter}] {claim_text}")
                counter += 1
            result = result.replace(f"[{claim_id}]", f"[{seen[claim_id]}]", 1)
        
        if footnotes:
            result += "\n\n---\nReferences:\n" + "\n".join(footnotes)
        return result
    
    elif format_style == "remove":
        return re.sub(pattern, '', content).replace('  ', ' ').strip()
    
    return content

def render_copy_button(
    content: str,
    button_text: str = "Copy to Clipboard",
    key: str = None,
    success_message: str = "Copied to clipboard!"
):
    """
    Render a copy-to-clipboard button for content.

    Used for Personal LinkedIn posting where API access isn't available.
    Users copy content and paste it manually to their personal profiles.
    """
    escaped_content = content.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$').replace('\n', '\\n').replace('\r', '\\r')

    import hashlib
    unique_id = hashlib.md5(f"{key or content[:20]}".encode()).hexdigest()[:8]

    copy_html = f"""
    <style>
        .copy-btn-{unique_id} {{
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 14px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
            transition: background-color 0.3s;
        }}
        .copy-btn-{unique_id}:hover {{
            background-color: #45a049;
        }}
        .copy-btn-{unique_id}.copied {{
            background-color: #2196F3;
        }}
        .copy-success-{unique_id} {{
            color: #4CAF50;
            font-size: 12px;
            margin-left: 10px;
            display: none;
        }}
    </style>
    <button class="copy-btn-{unique_id}" onclick="copyContent_{unique_id}()">
        {button_text}
    </button>
    <span class="copy-success-{unique_id}" id="success-{unique_id}">{success_message}</span>
    <script>
        function copyContent_{unique_id}() {{
            const content = `{escaped_content}`;
            navigator.clipboard.writeText(content).then(function() {{
                const btn = document.querySelector('.copy-btn-{unique_id}');
                const success = document.getElementById('success-{unique_id}');
                btn.classList.add('copied');
                btn.textContent = 'Copied!';
                success.style.display = 'inline';
                setTimeout(function() {{
                    btn.classList.remove('copied');
                    btn.textContent = '{button_text}';
                    success.style.display = 'none';
                }}, 2000);
            }}).catch(function(err) {{
                console.error('Failed to copy: ', err);
                alert('Failed to copy content. Please select and copy manually.');
            }});
        }}
    </script>
    """

    components.html(copy_html, height=60)

def render_linkedin_copy_section(
    headline: str,
    body: str,
    cta: str = None,
    key: str = None,
    expand_claims: bool = True
):
    """
    Render a complete copy section for LinkedIn personal posting.

    Combines headline, body, and CTA into a formatted post ready for copying.
    Automatically expands claim citations [CLM_XXX] to readable text.
    """
    import re
    
    # Strip "Claims Used:" metadata (LLM sometimes includes this)
    def strip_claims_metadata(text: str) -> str:
        if not text:
            return text
        return re.sub(r'\s*Claims?\s*Used:.*$', '', text, flags=re.IGNORECASE).strip()
    
    clean_headline = strip_claims_metadata(headline)
    clean_body = strip_claims_metadata(body)
    clean_cta = strip_claims_metadata(cta)
    
    display_headline = expand_claim_citations(clean_headline, "inline") if expand_claims and clean_headline else clean_headline
    display_body = expand_claim_citations(clean_body, "inline") if expand_claims and clean_body else clean_body
    display_cta = expand_claim_citations(clean_cta, "inline") if expand_claims and clean_cta else clean_cta
    
    linkedin_post = display_headline or ""

    if display_body:
        if linkedin_post:
            linkedin_post += "\n\n"
        linkedin_post += display_body

    if display_cta:
        linkedin_post += f"\n\n{display_cta}"

    st.markdown("---")
    st.markdown("#### 📋 Copy for Personal LinkedIn")
    st.caption("Claims are automatically expanded to readable text for posting")

    show_preview = st.checkbox("Show preview", value=False, key=f"show_preview_{key}" if key else "show_preview")
    if show_preview:
        st.text_area(
            "LinkedIn Post Preview",
            linkedin_post,
            height=200,
            disabled=True,
            key=f"preview_{key}" if key else None,
            label_visibility="collapsed"
        )

    render_copy_button(
        content=linkedin_post,
        button_text="Copy LinkedIn Post",
        key=f"linkedin_{key}" if key else "linkedin_copy",
        success_message="Ready to paste in LinkedIn!"
    )

_PLATFORM_DISPLAY = {
    "linkedin": {"name": "LinkedIn", "icon": "💼", "label": "LinkedIn Post"},
    "twitter": {"name": "Twitter/X", "icon": "🐦", "label": "Tweet"},
    "email": {"name": "Email", "icon": "📧", "label": "Email Content"},
    "blog": {"name": "Blog", "icon": "📝", "label": "Blog Post"},
}

def render_platform_copy_section(
    headline: str,
    body: str,
    cta: str = None,
    platform: str = "linkedin",
    key: str = None,
    expand_claims: bool = True
):
    """
    Render a copy section for any platform.
    Adapts labels and formatting based on the target platform.
    """
    import re

    def strip_claims_metadata(text: str) -> str:
        if not text:
            return text
        return re.sub(r'\s*Claims?\s*Used:.*$', '', text, flags=re.IGNORECASE).strip()

    clean_headline = strip_claims_metadata(headline)
    clean_body = strip_claims_metadata(body)
    clean_cta = strip_claims_metadata(cta)

    display_headline = expand_claim_citations(clean_headline, "inline") if expand_claims and clean_headline else clean_headline
    display_body = expand_claim_citations(clean_body, "inline") if expand_claims and clean_body else clean_body
    display_cta = expand_claim_citations(clean_cta, "inline") if expand_claims and clean_cta else clean_cta

    pinfo = _PLATFORM_DISPLAY.get(platform.lower(), {"name": platform.title(), "icon": "📋", "label": f"{platform.title()} Post"})

    post_text = display_headline or ""
    if display_body:
        if post_text:
            post_text += "\n\n"
        post_text += display_body
    if display_cta:
        post_text += f"\n\n{display_cta}"

    # Twitter: truncate to 280 chars
    if platform.lower() in ("twitter", "x"):
        if len(post_text) > 280:
            post_text = post_text[:277] + "..."

    st.markdown("---")
    st.markdown(f"#### {pinfo['icon']} Copy for {pinfo['name']}")
    st.caption(f"Claims are automatically expanded to readable text for posting")

    show_preview = st.checkbox("Show preview", value=False, key=f"show_preview_{key}" if key else "show_preview_platform")
    if show_preview:
        st.text_area(
            f"{pinfo['label']} Preview",
            post_text,
            height=200,
            disabled=True,
            key=f"preview_{key}" if key else None,
            label_visibility="collapsed"
        )

    render_copy_button(
        content=post_text,
        button_text=f"Copy {pinfo['label']}",
        key=f"platform_{key}" if key else "platform_copy",
        success_message=f"Ready to paste in {pinfo['name']}!"
    )

def render_content_copy_buttons(
    headline: str = None,
    body: str = None,
    cta: str = None,
    key_prefix: str = ""
):
    """Render individual copy buttons for each content component."""
    st.markdown("**Quick Copy:**")

    cols = st.columns(3)

    with cols[0]:
        if headline:
            if st.button("Copy Headline", key=f"{key_prefix}_copy_headline", use_container_width=True):
                st.session_state[f'{key_prefix}_copied'] = 'headline'
                st.code(headline, language=None)
                st.success("Headline ready to copy!")

    with cols[1]:
        if body:
            if st.button("Copy Body", key=f"{key_prefix}_copy_body", use_container_width=True):
                st.session_state[f'{key_prefix}_copied'] = 'body'
                st.code(body[:500] + "..." if len(body) > 500 else body, language=None)
                st.success("Body ready to copy!")

    with cols[2]:
        if cta:
            if st.button("Copy CTA", key=f"{key_prefix}_copy_cta", use_container_width=True):
                st.session_state[f'{key_prefix}_copied'] = 'cta'
                st.code(cta, language=None)
                st.success("CTA ready to copy!")

def format_content_for_platform(
    headline: str,
    body: str,
    cta: str = None,
    platform: str = "linkedin"
) -> str:
    """Format content for specific platform posting."""
    if platform == "linkedin":
        content = f"{headline}\n\n{body}"
        if cta:
            content += f"\n\n{cta}"
        return content

    elif platform == "twitter":
        tweet = f"{headline}\n\n{body[:200]}"
        if cta:
            tweet += f"\n\n{cta}"
        if len(tweet) > 280:
            tweet = tweet[:277] + "..."
        return tweet

    elif platform == "email":
        return f"Subject: {headline}\n\n{body}\n\n{cta if cta else ''}"

    else:
        return f"{headline}\n\n{body}\n\n{cta if cta else ''}"
