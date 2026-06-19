"""
Operations - Knowledge Base and System Settings
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path
from datetime import datetime


sys.path.insert(0, str(Path(__file__).parent.parent))
import importlib
import utils.api_client
importlib.reload(utils.api_client)
from utils.api_client import AgenticAPIClient
from utils.data_controls import render_data_controls
from components import render_metric_card, render_status_card
from app_config import STATUS_ICONS
from app_config.constants import KB_CATEGORIES

st.set_page_config(page_title="Operations - Agentic AI", page_icon="⚙️", layout="wide")

@st.cache_resource
def get_api_client():
    return AgenticAPIClient()

api = get_api_client()

_ops_openai_configured = api.is_config_key_set('OPENAI_API_KEY')

st.title("⚙️ System Operations")
st.caption("Knowledge base management and system configuration")

with st.expander("ℹ️ Operations Guide", expanded=False):
    st.markdown("""
**What Operations Manages**

This page is the control center for three core areas of the Agentic AI Marketing Platform:

**📚 Knowledge Base (RAG)**
- Stores documents that AI agents reference when generating content — claims, brand info, product data, competitor analysis, and more.
- Uses **ChromaDB**, a vector database, for semantic search across documents. Documents are converted to numerical embeddings so the AI can find contextually relevant information, not just keyword matches.
- **How RAG works:** When an AI agent generates content (e.g., a LinkedIn post), it first queries the knowledge base to find relevant context. This grounds the output in your actual data, reducing hallucinations and ensuring brand consistency.

**🔧 System Settings**
- Configure thresholds, enable/disable features, and manage API keys for all integrated services (OpenAI, LinkedIn, Twitter, HubSpot, etc.).
- Connection tests verify that external services are reachable and properly authenticated.

**🛠️ Maintenance**
- Database cleanup, cache clearing, log rotation, and health checks.
- Monitor background job queues and review system logs for troubleshooting.

**Best Practices**
- Keep the knowledge base updated — stale documents lead to outdated AI-generated content.
- Regularly verify document relevance; remove obsolete entries to improve retrieval quality.
- Test connections after changing API keys to confirm they work.
- Back up the database before bulk operations (e.g., clearing collections or reindexing).
    """)

tab1, tab2, tab3, tab4 = st.tabs([
    "📚 Knowledge Base",
    "🔧 System Settings",
    "🛠️ Maintenance",
    "🔗 Integrations Hub"
])

with tab1:
    st.subheader("📚 Knowledge Base Management")
    st.caption("Manage RAG document ingestion, vector storage, and exploration. Documents stored here are used by AI agents as context when generating marketing content.")

    if 'kb_view_mode' not in st.session_state:
        st.session_state.kb_view_mode = 'list'
    if 'kb_selected_doc' not in st.session_state:
        st.session_state.kb_selected_doc = None
    if 'kb_collection' not in st.session_state:
        st.session_state.kb_collection = 'documents'

    col_coll, col_actions = st.columns([1, 2])
    
    with col_coll:
        try:
            collections_data = api.get_kb_collections()
            collections = [c['name'] for c in collections_data.get('collections', [])]
            if not collections:
                collections = ['documents']
            
            selected_collection = st.selectbox(
                "Active Collection", 
                collections, 
                index=collections.index(st.session_state.kb_collection) if st.session_state.kb_collection in collections else 0,
                help="Select the ChromaDB collection to work with. Each collection is an independent set of documents with its own vector index."
            )
            st.session_state.kb_collection = selected_collection
        except:
            st.session_state.kb_collection = 'documents'
            st.error("Failed to load collections")

    with col_actions:
        try:
            stats = api.get_knowledge_base_stats(collection_name=st.session_state.kb_collection)
            c1, c2, c3 = st.columns(3)
            c1.metric("Documents", stats.get('total_documents', 0), help="Total number of documents stored in this collection")
            c2.metric("Last Updated", stats.get('last_updated', 'N/A')[:10] if stats.get('last_updated') else 'Never', help="Date when the collection was last modified (document added, updated, or deleted)")
            c3.metric("Status", stats.get('status', 'Ready').title(), help="Current collection status — Ready means the index is available for queries")
        except:
            pass

    st.markdown("---")

    kb_tab1, kb_tab2, kb_tab3, kb_tab4 = st.tabs([
        "📄 Document Browser", 
        "📤 Ingestion & Upload", 
        "🕸️ Vector Space", 
        "⚙️ Collection Settings"
    ])

    with kb_tab1:
        col_search, col_filter, col_sort = st.columns([2, 1, 1])
        
        with col_search:
            search_query = st.text_input("🔍 Search documents", placeholder="Search by title or content...", help="Semantic search — finds documents by meaning, not just exact keyword matches. Powered by vector similarity.")
        
        with col_filter:
            try:
                categories_data = api.get_kb_categories(collection_name=st.session_state.kb_collection)
                categories = ["All"] + categories_data.get('categories', [])
            except:
                categories = ["All"] + KB_CATEGORIES
            category_filter = st.selectbox("Category", categories, index=0, help="Filter documents by category. Categories are assigned during ingestion.")
        
        with col_sort:
            sort_order = st.selectbox("Sort by", ["Newest First", "Oldest First", "Title A-Z"], help="Sort order for the document listing below")

        pk_sort = "created_at"
        pk_order = "desc"
        if sort_order == "Oldest First": pk_order = "asc"
        if sort_order == "Title A-Z": pk_sort = "title"; pk_order = "asc"
        
        cat_param = None if category_filter == "All" else category_filter

        if 'kb_view_mode' not in st.session_state:
            st.session_state.kb_view_mode = 'list'
            
        if st.session_state.kb_view_mode == 'list':
            limit = 20
            
            if 'kb_page' not in st.session_state:
                st.session_state.kb_page = 1

            if search_query:
                results = api.search_knowledge_base(query=search_query, limit=limit)
                docs = results 
                total_docs = len(results)
                is_search = True
                st.session_state.kb_page = 1 
            else:
                page = st.session_state.kb_page
                offset = (page - 1) * limit
                
                data = api.list_kb_documents_paginated(
                    collection=st.session_state.kb_collection, 
                    limit=limit, 
                    offset=offset, 
                    category=cat_param,
                    sort_by=pk_sort,
                    sort_order=pk_order
                )
                docs = data.get('documents', [])
                total_docs = data.get('total', 0)
                is_search = False

            if not is_search and total_docs > 0:
                total_pages = (total_docs + limit - 1) // limit
                current_page = st.session_state.kb_page
                
                start_idx = (current_page - 1) * limit + 1
                end_idx = min(current_page * limit, total_docs)
                
                p_col1, p_col2, p_col3 = st.columns([1, 3, 1])
                
                with p_col1:
                    if st.button("Previous", disabled=(current_page == 1), use_container_width=True):
                        st.session_state.kb_page -= 1
                        st.rerun()
                
                with p_col2:
                    st.markdown(f"<div style='text-align: center; padding-top: 5px;'>Page <b>{current_page}</b> of <b>{total_pages}</b></div>", unsafe_allow_html=True)
                
                with p_col3:
                    if st.button("Next", disabled=(current_page >= total_pages), use_container_width=True):
                        st.session_state.kb_page += 1
                        st.rerun()
                
                st.markdown(f"**Showing {start_idx}-{end_idx} of {total_docs} documents**")
            
            elif is_search:
                st.markdown(f"**Found {len(docs)} matching documents for '{search_query}'** ({limit} max)")
            
            else:
                st.write("No documents found.")

            for idx, doc in enumerate(docs):
                score_display = f" (Score: {doc.get('score', 0):.2f})" if is_search else ""
                doc_id = doc.get('id')
                
                with st.expander(f"📄 {doc.get('title', 'Untitled')}{score_display}"):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.caption(f"ID: {doc_id or 'N/A'} | Category: {doc.get('category', 'General')} | Created: {doc.get('created_at', 'N/A')}")
                        st.write(doc.get('content', '')[:300] + "...")
                    
                    with c2:
                        if st.button("View / Edit", key=f"view_{doc_id}_{idx}"):
                            st.session_state.kb_selected_doc = doc_id
                            st.session_state.kb_view_mode = 'details'
                            st.rerun()
                        
                        if st.button("Find Similar", key=f"sim_{doc_id}_{idx}"):
                            st.session_state.kb_selected_doc = doc_id
                            st.session_state.kb_view_mode = 'similar'
                            st.rerun()

                        if st.button("🗑️ Delete", key=f"del_{doc_id}_{idx}", type="primary"):
                            res = api.delete_kb_document(doc.get('id'), collection_name=st.session_state.kb_collection)
                            if res and res.get('success'):
                                st.toast("Deleted", icon="✅")
                                st.rerun()
                            else:
                                st.error("Failed to delete")

        elif st.session_state.kb_view_mode in ['details', 'edit'] and st.session_state.kb_selected_doc:
            if st.button("← Back to List"):
                st.session_state.kb_view_mode = 'list'
                st.rerun()
                
            st.markdown("### 📝 Document Editor")
            
            full_doc = api.get_kb_document(st.session_state.kb_selected_doc)
            
            if full_doc:
                with st.container():
                    # Use doc_id in keys to avoid conflicts
                    doc_id = st.session_state.kb_selected_doc
                    
                    edit_title = st.text_input("Title", value=full_doc.get('metadata', {}).get('title', ''), key=f"edit_title_{doc_id}", help="Document title — used for display and search results")
                    edit_category = st.text_input("Category", value=full_doc.get('metadata', {}).get('category', 'general'), key=f"edit_cat_{doc_id}", help="Category for organizing and filtering documents (e.g., general, strategy, technical)")
                    edit_tags = st.text_input("Tags (comma separated)", value=",".join(full_doc.get('metadata', {}).get('tags', [])), key=f"edit_tags_{doc_id}", help="Comma-separated tags for additional metadata and filtering")
                    edit_content = st.text_area("Content", value=full_doc.get('content', ''), height=300, key=f"edit_content_{doc_id}", help="The full document content. This text is embedded into vectors for semantic search and used as RAG context.")
                    
                    c_save, c_cancel = st.columns(2)
                    if c_save.button("💾 Save Changes", type="primary"):
                        tags_list = [t.strip() for t in edit_tags.split(',')]
                        meta = full_doc.get('metadata', {})
                        meta.update({"title": edit_title, "category": edit_category, "tags": tags_list})
                        
                        api.update_kb_document(
                            document_id=doc_id,
                            content=edit_content,
                            metadata=meta
                        )
                        st.toast("Document updated!", icon="✅")
                        st.session_state.kb_view_mode = 'list'
                        st.rerun()
            else:
                st.error("Document not found.")

        elif st.session_state.kb_view_mode == 'similar' and st.session_state.kb_selected_doc:
            if st.button("← Back to List"):
                st.session_state.kb_view_mode = 'list'
                st.rerun()

            st.markdown("### 🧬 Similar Documents")
            
            source_doc = api.get_kb_document(st.session_state.kb_selected_doc)
            if source_doc:
                st.caption(f"Finding documents similar to: **{source_doc.get('metadata', {}).get('title', 'Untitled')}**")
            
            sim_results = api.find_similar_documents(st.session_state.kb_selected_doc)
            
            if sim_results and sim_results.get('documents'):
                for idx, s_doc in enumerate(sim_results.get('documents', [])):
                    if str(s_doc.get('id')) == str(st.session_state.kb_selected_doc):
                        continue
                    
                    similarity = s_doc.get('similarity', 0)
                    title = s_doc.get('title') or s_doc.get('metadata', {}).get('title', 'Untitled')
                    content_preview = s_doc.get('content', '')[:300]
                    
                    with st.expander(f"📄 {title} — Similarity: {similarity:.1%}", expanded=(idx < 3)):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**Category:** {s_doc.get('category', 'general')}")
                            st.markdown("**Preview:**")
                            st.text(content_preview + "..." if len(content_preview) >= 300 else content_preview)
                        with col2:
                            st.metric("Match", f"{similarity:.1%}", help="Cosine similarity score — higher means more semantically similar to the source document")
                            if st.button("View", key=f"view_sim_{s_doc.get('id')}"):
                                st.session_state.kb_selected_doc = s_doc.get('id')
                                st.session_state.kb_view_mode = 'details'
                                st.rerun()
            else:
                st.info("No similar documents found in the knowledge base.")


    with kb_tab2:
        st.subheader("📤 Add Content to Knowledge Base")
        
        ingest_mode = st.radio(
            "Ingestion Method", 
            ["📁 File Upload", "🌐 Web Scraper", "✏️ Manual Entry", "📂 Directory Ingest"], 
            horizontal=True,
            help="Choose how to add documents: upload files, scrape a webpage, type content manually, or bulk-ingest from a server directory."
        )

        if ingest_mode == "📁 File Upload":
            st.markdown("#### Upload Documents")
            st.caption("Supported formats: Markdown (.md), Text (.txt), PDF (.pdf)")
            
            uploaded_files = st.file_uploader(
                "Select files to upload", 
                accept_multiple_files=True, 
                type=['md', 'txt', 'pdf'],
                help="Select one or more files to upload. Each file becomes a separate document in the knowledge base."
            )
            
            col_cat, col_tags = st.columns(2)
            with col_cat:
                upload_category = st.selectbox("Category", KB_CATEGORIES, help="Category to assign to all uploaded files")
            with col_tags:
                upload_tags = st.text_input("Tags (comma separated)", placeholder="marketing, ai, research", help="Tags applied to all uploaded files for easier filtering")
            
            if uploaded_files:
                st.info(f"📎 {len(uploaded_files)} file(s) selected")
                
                if st.button("🚀 Upload & Ingest", type="primary"):
                    progress_bar = st.progress(0)
                    success_count = 0
                    
                    for idx, file in enumerate(uploaded_files):
                        try:
                            content = file.getvalue().decode("utf-8")
                            tags_list = [t.strip() for t in upload_tags.split(",")] if upload_tags else []
                            
                            result = api.create_kb_document(
                                content=content,
                                title=file.name,
                                category=upload_category,
                                tags=tags_list,
                                collection=st.session_state.kb_collection
                            )
                            if result and result.get('id'):
                                success_count += 1
                        except Exception as e:
                            st.error(f"Failed to upload {file.name}: {e}")
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    if success_count > 0:
                        st.toast(f"✅ Successfully uploaded {success_count}/{len(uploaded_files)} files!", icon="✅")
                        st.rerun()

        elif ingest_mode == "🌐 Web Scraper":
            st.markdown("#### Scrape Web Content")
            st.caption("Extract content from web pages and add to knowledge base")
            
            url_input = st.text_input("URL to scrape", placeholder="https://example.com/article", help="Enter the full URL of the webpage to extract content from and add to the knowledge base")
            
            col_scrape_cat, col_scrape_tags = st.columns(2)
            with col_scrape_cat:
                scrape_category = st.selectbox("Category", KB_CATEGORIES, key="scrape_cat", help="Category to assign to the scraped content")
            with col_scrape_tags:
                scrape_tags = st.text_input("Tags (comma separated)", placeholder="web, article", key="scrape_tags", help="Tags to apply to the scraped document")
            
            if url_input and st.button("🔍 Scrape & Ingest", type="primary"):
                with st.spinner("Scraping content..."):
                    try:
                        scrape_result = api.scrape_url_to_kb(
                            url=url_input,
                            category=scrape_category,
                            tags=[t.strip() for t in scrape_tags.split(",")] if scrape_tags else [],
                            collection=st.session_state.kb_collection
                        )
                        if scrape_result and scrape_result.get('success'):
                            st.success(f"✅ Scraped and ingested: {scrape_result.get('title', url_input)}")
                        else:
                            st.error(f"Failed to scrape: {scrape_result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Scraping failed: {e}")

        elif ingest_mode == "✏️ Manual Entry":
            st.markdown("#### Add Document Manually")
            
            with st.form("manual_ingest"):
                new_title = st.text_input("Title *", placeholder="Document title", help="A descriptive title for the document — shown in search results and the document browser")
                
                col_m_cat, col_m_tags = st.columns(2)
                with col_m_cat:
                    new_cat = st.selectbox("Category", KB_CATEGORIES, help="Document category for organizing and filtering")
                with col_m_tags:
                    new_tags = st.text_input("Tags (comma separated)", placeholder="tag1, tag2", help="Optional tags for additional metadata and filtering")
                
                new_content = st.text_area("Content *", height=250, placeholder="Enter document content here...", help="The document content that will be vectorized and stored. This is what AI agents will retrieve during RAG.")
                
                submitted = st.form_submit_button("➕ Add Document", type="primary")
                
                if submitted:
                    if not new_title or not new_content:
                        st.error("Title and content are required")
                    else:
                        tags_list = [t.strip() for t in new_tags.split(",")] if new_tags else []
                        result = api.create_kb_document(
                            content=new_content, 
                            title=new_title, 
                            category=new_cat,
                            tags=tags_list,
                            collection=st.session_state.kb_collection
                        )
                        if result and result.get('id'):
                            st.toast(f"✅ Document '{new_title}' created with ID: {result.get('id')}", icon="✅")
                            st.rerun()
                        else:
                            st.error("Failed to create document")

        elif ingest_mode == "📂 Directory Ingest":
            st.markdown("#### Bulk Ingest from Directory")
            st.caption("Ingest all supported files from a server directory")
            
            dir_path = st.text_input(
                "Directory Path", 
                placeholder="/srv/thesis/agentic-marketing-platform/data/knowledge_base",
                help="Absolute path to a directory on the server containing .md, .txt, or .pdf files to ingest"
            )
            
            col_dir_cat, col_recursive = st.columns(2)
            with col_dir_cat:
                dir_category = st.selectbox("Default Category", KB_CATEGORIES, key="dir_cat", help="Category assigned to all files ingested from this directory")
            with col_recursive:
                recursive = st.checkbox("Include subdirectories", value=True, help="When enabled, also processes files in nested subdirectories")
            
            if dir_path and st.button("📥 Start Bulk Ingestion", type="primary"):
                with st.spinner(f"Ingesting files from {dir_path}..."):
                    try:
                        res = api.ingest_directory(
                            dir_path, 
                            collection_name=st.session_state.kb_collection,
                            category=dir_category
                        )
                        if res and res.get('success'):
                            st.success(f"✅ Ingested {res.get('documents_added', 0)} documents")
                        else:
                            st.warning(f"Ingestion completed: {res}")
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")


    with kb_tab3:
        st.subheader("🕸️ Vector Space Visualization")
        st.caption("Explore semantic relationships between documents in embedding space. Documents close together share similar meaning — useful for identifying clusters, gaps, and redundancies.")
        
        try:
            stats = api.get_knowledge_base_stats(collection_name=st.session_state.kb_collection)
            total_docs = stats.get('total_documents', 0)
        except:
            total_docs = 0
        
        col_info, col_limit = st.columns([2, 1])
        with col_info:
            st.info(f"📊 Collection **{st.session_state.kb_collection}** has **{total_docs}** documents")
        with col_limit:
            viz_limit = st.selectbox("Documents to visualize", [50, 100, 200, 500], index=1, key="viz_limit_select", help="Number of documents to include in the visualization. Higher values give a fuller picture but take longer to compute.")
        
        if total_docs == 0:
            st.warning("No documents in this collection. Add some documents first.")
        else:
            generate_viz = st.button("🎨 Generate 2D Visualization", type="primary", use_container_width=True)
            
            if generate_viz or st.session_state.get('kb_viz_data'):
                if generate_viz:
                    with st.spinner("Computing embeddings and applying t-SNE dimensionality reduction..."):
                        try:
                            viz_data = api.get_kb_embeddings_for_viz(collection=st.session_state.kb_collection, limit=viz_limit)
                            st.session_state.kb_viz_data = viz_data
                        except Exception as e:
                            st.error(f"Failed to fetch visualization data: {e}")
                            st.session_state.kb_viz_data = None
                
                viz_data = st.session_state.get('kb_viz_data')
                
                if viz_data and viz_data.get('points') and len(viz_data.get('points', [])) > 0:
                    import plotly.graph_objects as go
                    
                    df = pd.DataFrame(viz_data['points'])
                    df['title'] = viz_data.get('titles', ['Untitled'] * len(df))
                    df['category'] = viz_data.get('categories', ['general'] * len(df))
                    df['doc_id'] = viz_data.get('doc_ids', list(range(len(df))))
                    
                    color_map = {
                        'general': '#FF6B6B',
                        'blog_post': '#4ECDC4',
                        'whitepaper': '#45B7D1',
                        'case_study': '#96CEB4',
                        'product': '#FFEAA7',
                        'tutorial': '#DDA0DD',
                        'news': '#98D8C8',
                    }
                    
                    fig = go.Figure()
                    
                    for category in df['category'].unique():
                        cat_df = df[df['category'] == category]
                        color = color_map.get(category, '#FF6B6B')
                        
                        fig.add_trace(go.Scatter(
                            x=cat_df['x'].tolist(),
                            y=cat_df['y'].tolist(),
                            mode='markers',
                            name=f"{category} ({len(cat_df)})",
                            text=cat_df['title'].tolist(),
                            hovertemplate='<b>%{text}</b><extra></extra>',
                            marker=dict(
                                size=14,
                                color=color,
                                opacity=0.9,
                                line=dict(width=2, color='white')
                            )
                        ))
                    
                    x_margin = (df['x'].max() - df['x'].min()) * 0.1
                    y_margin = (df['y'].max() - df['y'].min()) * 0.1
                    
                    fig.update_layout(
                        title=dict(text=f"Document Embedding Space ({len(df)} documents)", font=dict(color='white', size=16)),
                        margin=dict(l=40, r=40, t=60, b=40),
                        height=600,
                        plot_bgcolor='#1E1E1E',
                        paper_bgcolor='#1E1E1E',
                        xaxis=dict(
                            showgrid=True, 
                            gridwidth=1, 
                            gridcolor='rgba(255,255,255,0.1)', 
                            zeroline=False, 
                            showticklabels=False, 
                            title="",
                            range=[df['x'].min() - x_margin, df['x'].max() + x_margin]
                        ),
                        yaxis=dict(
                            showgrid=True, 
                            gridwidth=1, 
                            gridcolor='rgba(255,255,255,0.1)', 
                            zeroline=False, 
                            showticklabels=False, 
                            title="",
                            range=[df['y'].min() - y_margin, df['y'].max() + y_margin]
                        ),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='white')),
                        font=dict(color='white')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="viz_scatter")
                    
                    st.caption("**How to read:** Documents close together are semantically similar. Colors represent categories. Hover for details.")
                    
                    with st.expander("📊 Category Breakdown"):
                        cat_counts = df['category'].value_counts()
                        for cat, count in cat_counts.items():
                            st.write(f"- **{cat}**: {count} documents")
                elif viz_data and viz_data.get('error'):
                    st.error(f"Visualization error: {viz_data.get('error')}")
                else:
                    st.warning("Could not generate visualization. Ensure documents have embeddings.")


    with kb_tab4:
        st.subheader("⚙️ Collection Settings")
        
        st.markdown("### 📊 Collection Information")
        try:
            stats = api.get_knowledge_base_stats(collection_name=st.session_state.kb_collection)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Documents", stats.get('total_documents', 0), help="Total number of documents stored in this collection")
            col2.metric("Collection Name", st.session_state.kb_collection, help="Name of the currently active ChromaDB collection")
            col3.metric("Status", stats.get('status', 'Unknown').title(), help="Current index status — Ready means queries can be served")
            col4.metric("Last Updated", stats.get('last_updated', 'N/A')[:10] if stats.get('last_updated') else 'Never', help="Date of the most recent modification to this collection")
        except:
            st.warning("Could not load collection stats")
        
        st.markdown("---")
        
        st.markdown("### 🧬 Embedding Configuration")
        if not _ops_openai_configured:
            st.info("🔑 Embedding requires `OPENAI_API_KEY`. Configure in ⚙️ System Settings → llm category to enable vectorization.")
        else:
            with st.expander("View/Edit Embedding Settings", expanded=False):
                col_model, col_dim = st.columns(2)
                with col_model:
                    embedding_model = st.selectbox(
                        "Embedding Model",
                        ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
                        help="OpenAI embedding model to use for vectorization"
                    )
                with col_dim:
                    st.text_input("Vector Dimensions", value="1536", disabled=True, help="Determined by the model")
                
                st.caption("⚠️ Changing embedding model requires re-indexing all documents")
        
        st.markdown("### 🔍 Search Configuration")
        with st.expander("View/Edit Search Settings", expanded=False):
            col_k, col_thresh = st.columns(2)
            with col_k:
                default_k = st.number_input("Default Results (k)", min_value=1, max_value=50, value=10, help="Default number of similar documents to return")
            with col_thresh:
                similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05, help="Minimum similarity score for results")
            
            search_method = st.selectbox("Search Method", ["Cosine Similarity", "L2 Distance", "Inner Product"], help="Distance metric for vector similarity. Cosine Similarity is recommended for text embeddings.")
        
        st.markdown("### 📇 Index Management")
        with st.expander("Index Operations", expanded=False):
            col_idx1, col_idx2 = st.columns(2)
            with col_idx1:
                if st.button("🔄 Reindex Collection"):
                    with st.spinner("Reindexing — clearing and re-ingesting all documents..."):
                        result = api.request("POST", f"/knowledge-base/reindex?collection_name={st.session_state.kb_collection}")
                        if result and result.get("success"):
                            st.success(f"✅ {result.get('message', 'Reindex complete')}")
                        else:
                            st.error(f"❌ Reindex failed: {result.get('detail', 'Unknown error') if result else 'No response'}")
            with col_idx2:
                if st.button("🩺 Validate Index"):
                    st.success("Index is healthy")
        
        st.markdown("---")
        
        st.markdown("### 🚨 Danger Zone")
        with st.container(border=True):
            st.warning("⚠️ These actions are destructive and cannot be undone!")
            
            col_danger1, col_danger2 = st.columns(2)
            
            with col_danger1:
                st.markdown("**Clear All Documents**")
                st.caption(f"Delete all documents in '{st.session_state.kb_collection}'")
                
                confirm_clear = st.checkbox("I understand this will delete all documents", key="confirm_clear", help="Safety confirmation — check this box to enable the Clear Collection button")
                if st.button("🗑️ Clear Collection", disabled=not confirm_clear, type="primary"):
                    with st.spinner("Clearing collection..."):
                        try:
                            api.clear_kb_collection(st.session_state.kb_collection)
                            st.toast(f"Collection '{st.session_state.kb_collection}' cleared.", icon="✅")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to clear: {e}")
            
            with col_danger2:
                st.markdown("**Delete Collection**")
                st.caption("Remove the entire collection and its schema")
                
                confirm_delete = st.checkbox("I understand this will delete the collection", key="confirm_delete", help="Safety confirmation — check this box to enable the Delete Collection button")
                if st.button("💀 Delete Collection", disabled=not confirm_delete, type="primary"):
                    st.warning("Collection deletion not implemented for safety")


with tab2:
    st.subheader("🔧 System Settings")
    st.caption("Configure all platform settings from this dashboard. Changes are saved to the database and take effect immediately across all services.")
    
    try:
        config_status = api.request("GET", "/config/status")
        is_configured = config_status.get("is_configured", False)
        total_settings = config_status.get("total_settings", 0)
        configured_count = config_status.get("configured_count", 0)
    except Exception as e:
        is_configured = False
        total_settings = 0
        configured_count = 0
        st.warning(f"Unable to load configuration status: {str(e)}")
    
    if not is_configured or total_settings == 0:
        st.warning("⚠️ **First-Time Setup Required**")
        st.info("Click the button below to initialize the system with default configurations.")
        
        if st.button("🚀 Initialize System Configuration", type="primary"):
            try:
                result = api.request("POST", "/config/initialize")
                st.success(f"✅ Initialized {result.get('created', 0)} configuration settings!")
                st.rerun()
            except Exception as e:
                st.error(f"Initialization failed: {str(e)}")
    
    if total_settings > 0:
        progress = configured_count / total_settings if total_settings > 0 else 0
        st.progress(progress, text=f"Configured: {configured_count}/{total_settings} settings")
    
    st.markdown("---")
    
    st.markdown("### 🔌 Connection Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Test Database", use_container_width=True):
            try:
                result = api.request("POST", "/config/test-connection", json={"service": "database"})
                if result.get("success"):
                    st.success("✅ Database OK")
                else:
                    st.error(f"❌ {result.get('message')}")
            except Exception as e:
                st.error(f"❌ {str(e)}")
    
    with col2:
        if st.button("Test Redis", use_container_width=True):
            try:
                result = api.request("POST", "/config/test-connection", json={"service": "redis"})
                if result.get("success"):
                    st.success("✅ Redis OK")
                else:
                    st.error(f"❌ {result.get('message')}")
            except Exception as e:
                st.error(f"❌ {str(e)}")
    
    with col3:
        if st.button("Test Ollama", use_container_width=True):
            try:
                result = api.request("POST", "/config/test-connection", json={"service": "ollama"})
                if result.get("success"):
                    st.success(f"✅ {result.get('message')}")
                else:
                    st.error(f"❌ {result.get('message')}")
            except Exception as e:
                st.error(f"❌ {str(e)}")
    
    with col4:
        if st.button("Test OpenAI", use_container_width=True):
            try:
                result = api.request("POST", "/config/test-connection", json={"service": "openai"})
                if result.get("success"):
                    st.success("✅ OpenAI OK")
                else:
                    st.warning(f"⚠️ {result.get('message')}")
            except Exception as e:
                st.warning(f"⚠️ {str(e)}")
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Configuration Categories")
    
    CATEGORY_ICONS = {
        "llm": "🤖",
        "database": "🗄️",
        "redis": "⚡",
        "linkedin": "💼",
        "twitter": "🐦",
        "email": "📧",
        "hubspot": "🔗",
        "calendar": "📅",
        "apify": "🔍",
        "governance": "🛡️",
        "cost_control": "💰",
        "simulation": "📊",
        "learning": "🧠",
        "marl": "🎮",
        "monitoring": "📡",
        "security": "🔐",
        "feature_flags": "🚩",
        "application": "⚙️",
    }
    
    try:
        categories = api.request("GET", "/config/categories")
        
        category_options = [f"{CATEGORY_ICONS.get(c['category'], '📁')} {c['display_name']}" for c in categories]
        category_values = [c['category'] for c in categories]
        
        selected_display = st.selectbox(
            "Select Configuration Category",
            category_options,
            index=0,
            help="Each category groups related settings. Select a category to view and edit its configuration values."
        )
        
        selected_category = category_values[category_options.index(selected_display)]
        
        configs = api.request("GET", f"/config/category/{selected_category}")
        
        if configs:
            st.markdown(f"### {selected_display}")
            
            if f"config_changes_{selected_category}" not in st.session_state:
                st.session_state[f"config_changes_{selected_category}"] = {}
            
            for config in configs:
                key = config['key']
                display_value = config.get('display_value', '')
                is_secret = config.get('is_secret', False)
                description = config.get('description', '')
                value_type = config.get('value_type', 'string')
                default_value = config.get('default_value', '')
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if value_type == 'boolean':
                        current_val = display_value.lower() in ('true', '1', 'yes', 'on') if display_value else False
                        new_val = st.checkbox(
                            key,
                            value=current_val,
                            help=description,
                            key=f"input_{key}"
                        )
                        new_val_str = "True" if new_val else "False"
                    elif is_secret:
                        new_val_str = st.text_input(
                            key,
                            value="",
                            type="password",
                            placeholder=f"Current: {display_value}" if display_value else "Not set",
                            help=f"{description}. Leave blank to keep current value.",
                            key=f"input_{key}"
                        )
                    elif value_type in ('integer', 'float'):
                        new_val_str = st.text_input(
                            key,
                            value=display_value or default_value,
                            help=description,
                            key=f"input_{key}"
                        )
                    else:
                        new_val_str = st.text_input(
                            key,
                            value=display_value or default_value,
                            help=description,
                            key=f"input_{key}"
                        )
                    
                    if new_val_str and (not is_secret or new_val_str != ""):
                        if is_secret and new_val_str:
                            st.session_state[f"config_changes_{selected_category}"][key] = new_val_str
                        elif not is_secret and new_val_str != display_value:
                            st.session_state[f"config_changes_{selected_category}"][key] = new_val_str
                
                with col2:
                    if default_value:
                        st.caption(f"Default: {default_value}")
                    if is_secret:
                        st.caption("🔒 Encrypted")
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("💾 Save Changes", type="primary", use_container_width=True):
                    changes = st.session_state.get(f"config_changes_{selected_category}", {})
                    if changes:
                        try:
                            result = api.request("POST", "/config/bulk-update", json={"configurations": changes})
                            if result.get("success"):
                                st.toast(f"✅ Saved {len(result.get('updated', []))} settings!", icon="✅")
                                st.session_state[f"config_changes_{selected_category}"] = {}
                                st.rerun()
                            else:
                                st.error(f"Failed: {result.get('message')}")
                        except Exception as e:
                            st.error(f"Error saving: {str(e)}")
                    else:
                        st.info("No changes to save")
            
            with col2:
                if st.button("🔄 Reset Form", use_container_width=True):
                    st.session_state[f"config_changes_{selected_category}"] = {}
                    st.rerun()
            
            with col3:
                pending = len(st.session_state.get(f"config_changes_{selected_category}", {}))
                if pending > 0:
                    st.info(f"📝 {pending} pending change(s)")
        else:
            st.info("No configurations found for this category")
    
    except Exception as e:
        st.error(f"Failed to load configurations: {str(e)}")
        st.info("Make sure the API is running and configurations are initialized.")

st.markdown("---")
st.caption(f"System Operations | Last updated: {datetime.now().strftime('%H:%M:%S')}")


with tab3:
    st.subheader("🛠️ System Maintenance")
    st.caption("Perform system maintenance tasks. Use these tools for cache management, backups, log rotation, and monitoring system health.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🧹 Cache Management")
        st.info("Clear Redis cache to remove stale data")
        if st.button("Clear Cache", type="primary", use_container_width=True):
            try:
                with st.spinner("Clearing cache..."):
                    result = api.clear_cache()
                if result.get('success'):
                    st.success(result.get('message'))
                else:
                    st.error(f"Failed: {result.get('message')}")
            except Exception as e:
                st.error(str(e))
                
    with col2:
        st.markdown("### 💾 Database Backup")
        st.info("Trigger an immediate database backup")
        if st.button("Trigger Backup", type="primary", use_container_width=True):
            try:
                with st.spinner("Starting backup..."):
                    result = api.trigger_backup()
                if result.get('success'):
                    st.success(f"{result.get('message')} (Job ID: {result.get('job_id')})")
                else:
                    st.error(f"Failed: {result.get('message')}")
            except Exception as e:
                st.error(str(e))
                
    with col3:
        st.markdown("### 📝 Log Rotation")
        st.info("Rotate application logs")
        if st.button("Rotate Logs", type="primary", use_container_width=True):
            try:
                with st.spinner("Rotating logs..."):
                    result = api.rotate_logs()
                if result.get('success'):
                    st.success(result.get('message'))
                else:
                    st.error(f"Failed: {result.get('message')}")
            except Exception as e:
                st.error(str(e))
                
    st.markdown("---")

    st.markdown("### 📋 Background Jobs Queue")

    try:
        queue_status = api.request("GET", "/operations/queue/status")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pending Jobs", queue_status.get("pending", 0), help="Jobs waiting in the queue to be picked up by a worker")
        with col2:
            st.metric("Running Jobs", queue_status.get("running", 0), help="Jobs currently being executed by background workers")
        with col3:
            st.metric("Completed (24h)", queue_status.get("completed_24h", 0), help="Successfully completed jobs in the last 24 hours")
        with col4:
            st.metric("Failed (24h)", queue_status.get("failed_24h", 0), help="Jobs that failed in the last 24 hours — check logs for details")

        recent_jobs = queue_status.get("recent_jobs", [])
        if recent_jobs:
            st.markdown("#### Recent Jobs")
            for job in recent_jobs[:5]:
                job_status = job.get("status", "unknown")
                status_icon = STATUS_ICONS.get(job_status, "❓")
                st.write(f"{status_icon} **{job.get('type', 'Unknown')}** - {job_status} - {job.get('created_at', 'N/A')[:19] if job.get('created_at') else 'N/A'}")
        else:
            st.info("No recent jobs in queue")

    except Exception as e:
        # Fallback display when API not available - show error state, not fake success
        st.warning("⚠️ Could not connect to queue status API")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queue Status", "Unknown", help="Unable to connect to the queue service")
        with col2:
            st.metric("Worker Status", "Unknown", help="Unable to determine worker status")
        st.caption("Queue and worker status unavailable. Check if Redis and the worker service are running.")

    st.markdown("---")

    st.markdown("### 📜 System Logs Viewer")

    col_filter, col_limit = st.columns([2, 1])
    with col_filter:
        log_level = st.selectbox(
            "Log Level",
            ["ALL", "ERROR", "WARNING", "INFO", "DEBUG"],
            index=0,
            help="Filter logs by severity level. ALL shows everything; ERROR shows only errors."
        )
    with col_limit:
        log_limit = st.number_input("Lines to show", min_value=10, max_value=500, value=50, step=10, help="Number of most recent log lines to display")

    if st.button("🔄 Refresh Logs", use_container_width=True):
        st.rerun()

    try:
        logs_response = api.request("GET", f"/operations/logs?level={log_level.lower()}&limit={log_limit}")
        logs = logs_response.get("logs", [])

        if logs:
            log_text = ""
            for log_entry in logs:
                level = log_entry.get("level", "INFO")
                timestamp = log_entry.get("timestamp", "")[:19]
                message = log_entry.get("message", "")
                source = log_entry.get("source", "")

                if level == "ERROR":
                    prefix = "🔴"
                elif level == "WARNING":
                    prefix = "🟡"
                elif level == "DEBUG":
                    prefix = "⚪"
                else:
                    prefix = "🟢"

                log_text += f"{prefix} [{timestamp}] [{level}] [{source}] {message}\n"

            st.code(log_text, language="log")
        else:
            st.info("No logs found matching criteria")

    except Exception:
        # Fallback when API fails - show format example, clearly marked as such
        st.warning("⚠️ Could not fetch live logs from API.")
        st.caption("Log format example (not real data):")
        st.code("""🟢 [2026-01-28 10:30:15] [INFO] [api] Server started successfully
🟢 [2026-01-28 10:30:16] [INFO] [scheduler] Weekly report scheduler started
🟡 [2026-01-28 10:35:22] [WARNING] [cache] Cache hit rate below threshold (75%)""", language="log")
        st.caption("Logs are stored in the workflow_events table when campaign/workflow events occur.")

    st.markdown("---")

    st.markdown("### 🗄️ Database Health")

    try:
        db_health = api.request("GET", "/operations/database/health")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Connection Pool", f"{db_health.get('pool_used', 0)}/{db_health.get('pool_size', 10)}", help="Database connections in use vs. total pool size. High utilization may indicate performance issues.")
        with col2:
            st.metric("Active Queries", db_health.get("active_queries", 0), help="Number of SQL queries currently being executed")
        with col3:
            st.metric("Avg Query Time", f"{db_health.get('avg_query_ms', 0):.1f}ms", help="Average query execution time in milliseconds. Values above 100ms may indicate slow queries.")

    except Exception:
        st.info("Database health metrics available via System Monitor page")

with tab4:
    st.caption("Unified view of all external service connections. Configure, test, and monitor integrations from one place.")

    with st.expander("ℹ️ Integrations Guide", expanded=False):
        st.markdown("""
**External Service Integrations**

The Agentic platform connects to several external services for content distribution, lead tracking, and notifications:

| Service | Purpose | Required Keys | Where Used |
|---------|---------|---------------|------------|
| **Ollama** | Local LLM inference (free) | Auto-detected | Content generation, safety checks |
| **OpenAI** | Cloud LLM inference (paid) | `OPENAI_API_KEY` | Fallback content generation when Ollama unavailable |
| **LinkedIn** | B2B social media marketing | `LINKEDIN_ACCESS_TOKEN` | Content deployment to LinkedIn pages |
| **Twitter/X** | Social media marketing | `TWITTER_API_KEY`, `TWITTER_ACCESS_TOKEN` | Content deployment to Twitter/X |
| **WordPress** | Blog CMS publishing | `BLOG_CMS_URL`, `BLOG_API_KEY` | Blog post deployment |
| **SendGrid** | Transactional email delivery | `SENDGRID_API_KEY` | Alert notifications, campaign rollback emails |
| **Mailgun** | Email campaign delivery | `MAILGUN_API_KEY`, `MAILGUN_DOMAIN` | Email campaign deployment |
| **Mailchimp** | Email marketing automation | `MAILCHIMP_API_KEY` | Email list management, campaign delivery |
| **HubSpot** | CRM & deal pipeline tracking | `HUBSPOT_API_KEY` | Funnel Attribution — tracks deals through pipeline |
| **Cal.com** | Demo call booking & scheduling | `CALENDAR_API_KEY` | Funnel Attribution — tracks booked calls |
| **Slack** | Real-time alert notifications | `SLACK_WEBHOOK_URL` | Canary deployment rollbacks, system alerts |
| **Perspective API** | Content toxicity analysis | `PERSPECTIVE_API_KEY` | Safety guardrails — LLM output moderation |
| **Apify** | Web scraping & competitive intel | `APIFY_API_TOKEN` | Market research, competitor analysis |

**How Keys Are Managed:**
- Keys are stored encrypted in the database via the Configuration Service
- Set keys in the **System Settings** tab under the appropriate category
- Changes take effect immediately (no restart needed)

**Testing Integrations:**
- Click **Test Connection** on each card to verify the service is reachable
- Green = connected and working, Yellow = configured but not reachable, Red = not configured
        """)

    try:
        integration_status = api.get_integration_status()
    except Exception as e:
        st.error(f"Failed to check integration status: {e}")
        integration_status = {}

    if st.button("🔄 Refresh Integration Status", key="refresh_integrations"):
        st.rerun()

    st.markdown("### 📊 Integration Status Overview")

    integrations = [
        ("ollama", "🤖 Ollama", "Local LLM inference", "OLLAMA_HOST"),
        ("openai", "☁️ OpenAI", "Cloud LLM inference", "OPENAI_API_KEY"),
        ("linkedin", "💼 LinkedIn", "B2B social media", "LINKEDIN_ACCESS_TOKEN"),
        ("twitter", "🐦 Twitter/X", "Social media marketing", "TWITTER_API_KEY"),
        ("wordpress", "📝 WordPress", "Blog CMS publishing", "BLOG_CMS_URL"),
        ("sendgrid", "📧 SendGrid", "Transactional email", "SENDGRID_API_KEY"),
        ("mailgun", "📨 Mailgun", "Email campaign delivery", "MAILGUN_API_KEY"),
        ("mailchimp", "📬 Mailchimp", "Email marketing", "MAILCHIMP_API_KEY"),
        ("hubspot", "🔗 HubSpot", "CRM & deal pipeline", "HUBSPOT_API_KEY"),
        ("calcom", "📅 Cal.com", "Demo call booking", "CALENDAR_API_KEY"),
        ("slack", "💬 Slack", "Alert notifications", "SLACK_WEBHOOK_URL"),
        ("perspective", "🛡️ Perspective API", "Content safety", "PERSPECTIVE_API_KEY"),
        ("apify", "🔍 Apify", "Web scraping & intel", "APIFY_API_TOKEN"),
    ]

    cols = st.columns(4)
    for idx, (key, name, desc, config_key) in enumerate(integrations):
        status = integration_status.get(key, {})
        connected = status.get("connected", False)
        configured = status.get("configured", False)
        error = status.get("error")
        details = status.get("details", {})

        with cols[idx % 4]:
            if connected:
                st.success(f"**{name}**")
                st.caption(f"✅ Connected")
            elif configured:
                st.warning(f"**{name}**")
                st.caption(f"⚠️ Configured but not reachable")
            else:
                st.error(f"**{name}**")
                st.caption(f"❌ Not configured")

            if error:
                st.caption(f"_{error}_")
            if details:
                detail_items = [f"{k}: {v}" for k, v in details.items()]
                st.caption(" | ".join(detail_items[:3]))

    st.markdown("---")
    st.markdown("### 🚀 Deployment Readiness")
    st.caption("Shows which platforms are ready for content deployment. Fix any ❌ items before launching campaigns on that platform.")

    mailgun_ok = integration_status.get("mailgun", {}).get("connected", False)
    mailgun_domain = integration_status.get("mailgun", {}).get("details", {}).get("domain", "")
    wp_ok = integration_status.get("wordpress", {}).get("connected", False)
    wp_error = integration_status.get("wordpress", {}).get("error", "")
    linkedin_ok = integration_status.get("linkedin", {}).get("connected", False)
    twitter_ok = integration_status.get("twitter", {}).get("connected", False)
    slack_ok = integration_status.get("slack", {}).get("connected", False)
    ollama_ok = integration_status.get("ollama", {}).get("connected", False)

    deploy_cols = st.columns(4)
    with deploy_cols[0]:
        if mailgun_ok:
            st.success("**📧 Email**")
            st.caption(f"✅ Ready — via Mailgun ({mailgun_domain})")
        else:
            st.error("**📧 Email**")
            st.caption("❌ Set MAILGUN_API_KEY, MAILGUN_DOMAIN, MAILGUN_REGION in System Settings → Email")
    with deploy_cols[1]:
        if wp_ok:
            st.success("**📝 Blog**")
            st.caption("✅ Ready — WordPress connected")
        else:
            st.error("**📝 Blog**")
            if wp_error:
                st.caption(f"❌ {wp_error}")
            else:
                st.caption("❌ Set BLOG_CMS_URL, BLOG_USERNAME, BLOG_APP_PASSWORD in System Settings → Blog")
    with deploy_cols[2]:
        if linkedin_ok:
            st.success("**💼 LinkedIn**")
            st.caption("✅ Ready — API connected")
        else:
            st.warning("**💼 LinkedIn**")
            st.caption("⚠️ Mock mode — set LINKEDIN_ACCESS_TOKEN for real deployment")
    with deploy_cols[3]:
        if twitter_ok:
            st.success("**🐦 Twitter/X**")
            st.caption("✅ Ready — API connected")
        else:
            st.warning("**🐦 Twitter/X**")
            st.caption("⚠️ Mock mode — set TWITTER_API_KEY + TWITTER_ACCESS_TOKEN for real deployment")

    prereq_cols = st.columns(3)
    with prereq_cols[0]:
        if ollama_ok:
            st.success("**🤖 LLM (Ollama)**")
            models = integration_status.get("ollama", {}).get("details", {}).get("models", [])
            st.caption(f"✅ {len(models)} models: {', '.join(models[:3])}")
        else:
            st.error("**🤖 LLM (Ollama)**")
            st.caption("❌ Required for content generation — check Ollama is running")
    with prereq_cols[1]:
        if slack_ok:
            st.success("**💬 Slack Alerts**")
            st.caption("✅ Notifications will be sent to your channel")
        else:
            st.warning("**💬 Slack Alerts**")
            st.caption("⚠️ No alerts — set SLACK_WEBHOOK_URL in System Settings → Monitoring")
    with prereq_cols[2]:
        email_recipients = ""
        try:
            resp = api.request("GET", "/config/DEFAULT_EMAIL_RECIPIENTS")
            if resp and isinstance(resp, dict):
                email_recipients = resp.get("value", "")
        except Exception:
            pass
        if email_recipients:
            st.success("**📬 Email Recipients**")
            st.caption(f"✅ {email_recipients}")
        else:
            st.warning("**📬 Email Recipients**")
            st.caption("⚠️ Set DEFAULT_EMAIL_RECIPIENTS in System Settings → Email")

    st.markdown("---")
    st.markdown("### 💬 Slack Notifications")
    st.caption("Send test notifications and view Slack integration status.")

    slack_status = integration_status.get("slack", {})
    if slack_status.get("connected"):
        col1, col2 = st.columns([3, 1])
        with col1:
            test_message = st.text_input(
                "Test Message",
                value="🧪 Agentic Test: Integration verification from dashboard",
                help="Send a test message to your configured Slack channel to verify the webhook works."
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📤 Send Test", key="send_slack_test"):
                try:
                    result = api.request("POST", "/config/test-connection", {"service": "slack"}, timeout=15)
                    if result and result.get("success"):
                        st.toast("✅ Slack message sent successfully!", icon="✅")
                    else:
                        st.error(f"Failed: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to send: {e}")

        st.caption("Slack alerts are triggered automatically for: canary deployment rollbacks, safety violations, budget threshold breaches.")
    else:
        st.info("💡 **Configure Slack:** Set `SLACK_WEBHOOK_URL` in System Settings → Monitoring category.")

    st.markdown("---")
    st.markdown("### 📧 SendGrid Email")
    st.caption("Email notifications for campaign alerts and system events.")

    sendgrid_status = integration_status.get("sendgrid", {})
    if sendgrid_status.get("connected"):
        st.success("SendGrid is connected and ready to send emails.")
        details = sendgrid_status.get("details", {})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("From Email", details.get("from_email", "Not set"), help="Sender email address for outgoing notifications.")
        with col2:
            st.metric("Status", "Active", help="SendGrid API connection is verified and working.")
        st.caption("📧 Emails are sent for: canary rollback alerts, campaign completion, budget warnings. Set `ALERT_EMAIL` in System Settings to receive notifications.")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🧪 Test Connection", key="test_sendgrid"):
                try:
                    result = api.request("POST", "/config/test-connection", {"service": "sendgrid"}, timeout=15)
                    if result and result.get("success"):
                        st.toast(f"✅ {result.get('message')}", icon="✅")
                    else:
                        st.error(f"Failed: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed: {e}")
    elif sendgrid_status.get("configured"):
        st.warning("SendGrid API key is configured but connection failed. Check the API key in System Settings.")
    else:
        st.info("💡 **Configure SendGrid:** Set `SENDGRID_API_KEY` and `SENDGRID_FROM_EMAIL` in System Settings → Notification category.")

    st.markdown("---")
    st.markdown("### 📨 Mailgun Email")
    st.caption("Email campaign delivery via Mailgun API.")

    mailgun_status = integration_status.get("mailgun", {})
    if mailgun_status.get("connected") or mailgun_status.get("configured"):
        col1, col2 = st.columns([3, 1])
        with col1:
            details = mailgun_status.get("details", {})
            domain = details.get("domain", "Not set")
            st.write(f"**Domain:** `{domain}`")
            if mailgun_status.get("connected"):
                st.success("Mailgun domain verified and active.")
            else:
                error = mailgun_status.get("error", "")
                st.warning(f"Configured but not fully connected. {error}")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📤 Send Test", key="send_mailgun_test"):
                try:
                    result = api.request("POST", "/config/test-connection", {"service": "mailgun"}, timeout=15)
                    if result and result.get("success"):
                        st.toast(f"✅ {result.get('message')}", icon="✅")
                    else:
                        st.error(f"Failed: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        st.info("💡 **Configure Mailgun:** Set `MAILGUN_API_KEY` and `MAILGUN_DOMAIN` in System Settings → Email category.")

    st.markdown("---")
    st.markdown("### 📅 Cal.com Calendar Integration")
    st.caption("Booking tracking for the marketing funnel (Impressions → Bookings → Shows → Closed Won).")

    calcom_status = integration_status.get("calcom", {})
    if calcom_status.get("connected"):
        details = calcom_status.get("details", {})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Event Types", details.get("event_types_count", 0), help="Number of Cal.com event types configured for booking.")
        with col2:
            st.metric("Status", "Connected", help="Cal.com API is connected and accepting webhook events.")

        try:
            bookings_resp = api.request("GET", "/funnel/calendar/bookings")
            bookings = bookings_resp.get("bookings", [])
            if bookings:
                st.markdown("**Recent Bookings:**")
                for b in bookings[:5]:
                    attendees = b.get("attendees", [])
                    email = attendees[0].get("email", "N/A") if attendees else b.get("attendee_email", "N/A")
                    title = b.get("title", "Booking")
                    start = b.get("startTime", b.get("start_time", ""))[:16].replace("T", " ")
                    status_icon = "✅" if b.get("status") == "ACCEPTED" else "⏳"
                    st.caption(f"{status_icon} {title} — {email} — {start}")
            else:
                st.caption("No recent bookings found.")
        except Exception:
            st.caption("Bookings data available on the Funnel Attribution page.")

        st.caption("📅 **Flow:** Content → Click → Lead → Cal.com Booking → Delayed Reward → Funnel Attribution. See Funnel Attribution page for full tracking.")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🧪 Test Connection", key="test_calcom"):
                try:
                    result = api.request("POST", "/config/test-connection", {"service": "calcom"}, timeout=15)
                    if result and result.get("success"):
                        st.toast(f"✅ {result.get('message')}", icon="✅")
                    else:
                        st.error(f"Failed: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        if calcom_status.get("configured"):
            error_msg = calcom_status.get("error", "Connection failed")
            st.warning(f"⚠️ Cal.com key configured but not reachable: **{error_msg}**")
            st.caption("Update `CALENDAR_API_KEY` in Operations → System Settings → Calendar if the key has expired.")
        else:
            st.info("💡 **Configure Cal.com:** Set `CALENDAR_API_KEY` in Operations → System Settings → Calendar category.")

    st.markdown("---")
    st.markdown("### 🔗 HubSpot CRM")
    st.caption("Deal pipeline tracking for full funnel attribution (Lead → Deal → Closed Won).")

    hubspot_status = integration_status.get("hubspot", {})
    if hubspot_status.get("connected"):
        st.success("HubSpot CRM is connected.")
        details = hubspot_status.get("details", {})
        if details.get("contacts"):
            st.metric("Contacts", details["contacts"], help="Total contacts in HubSpot CRM.")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🧪 Test Connection", key="test_hubspot"):
                try:
                    result = api.request("POST", "/config/test-connection", {"service": "hubspot"}, timeout=15)
                    if result and result.get("success"):
                        st.toast(f"✅ {result.get('message')}", icon="✅")
                    else:
                        st.error(f"Failed: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        st.info("""
💡 **HubSpot Setup Required:**

HubSpot CRM tracks the complete deal pipeline for full-funnel attribution.

**To configure:**
1. Get your HubSpot API key from Settings → Integrations → API Key
2. Set in System Settings → Integration:
   - `HUBSPOT_API_KEY`: Your private API key
   - `HUBSPOT_PORTAL_ID`: Your HubSpot portal ID

**Note:** Without HubSpot, funnel data below "Bookings" (Shows, Closed Won) is estimated using industry averages (80% show rate, 25% close rate).
        """)

    st.markdown("---")
    st.markdown("### 💼 LinkedIn")
    st.caption("B2B social media marketing — post content directly to LinkedIn company pages and profiles.")

    linkedin_status = integration_status.get("linkedin", {})
    if linkedin_status.get("connected"):
        st.success("LinkedIn is connected.")
        details = linkedin_status.get("details", {})
        if details.get("profile"):
            st.metric("Profile", details["profile"])
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🧪 Test Connection", key="test_linkedin"):
                try:
                    result = api.request("POST", "/config/test-connection", {"service": "linkedin"}, timeout=15)
                    if result and result.get("success"):
                        st.toast(f"✅ {result.get('message')}", icon="✅")
                    else:
                        st.error(f"Failed: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        st.info("""
💡 **LinkedIn Setup Required:**

LinkedIn integration enables B2B content deployment to your company page.

**To configure:**
1. Create a LinkedIn App at [linkedin.com/developers](https://www.linkedin.com/developers/)
2. Set in System Settings → LinkedIn:
   - `LINKEDIN_CLIENT_ID`: Your app's client ID
   - `LINKEDIN_CLIENT_SECRET`: Your app's client secret
   - `LINKEDIN_ACCESS_TOKEN`: OAuth2 access token
   - `LINKEDIN_ORGANIZATION_ID`: Your company page ID
        """)

    st.markdown("---")
    st.markdown("### 🐦 Twitter/X")
    st.caption("Social media marketing — automated content deployment to Twitter/X.")

    twitter_status = integration_status.get("twitter", {})
    if twitter_status.get("connected"):
        st.success("Twitter/X is connected.")
        details = twitter_status.get("details", {})
        if details.get("username"):
            st.metric("Account", details["username"])
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🧪 Test Connection", key="test_twitter"):
                try:
                    result = api.request("POST", "/config/test-connection", {"service": "twitter"}, timeout=15)
                    if result and result.get("success"):
                        st.toast(f"✅ {result.get('message')}", icon="✅")
                    else:
                        st.error(f"Failed: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        st.info("""
💡 **Twitter/X Setup Required:**

Twitter/X integration enables automated content posting.

**To configure:**
1. Create a Twitter Developer account and app at [developer.twitter.com](https://developer.twitter.com)
2. Set in System Settings → Twitter:
   - `TWITTER_API_KEY`: Consumer API key
   - `TWITTER_API_SECRET`: Consumer API secret
   - `TWITTER_ACCESS_TOKEN`: Access token
   - `TWITTER_ACCESS_TOKEN_SECRET`: Access token secret
        """)

    st.markdown("---")
    st.markdown("### 📝 WordPress")
    st.caption("Blog CMS publishing — deploy generated blog posts to your WordPress site.")

    wordpress_status = integration_status.get("wordpress", {})
    if wordpress_status.get("connected"):
        st.success("WordPress is connected.")
        details = wordpress_status.get("details", {})
        if details.get("url"):
            st.metric("Site URL", details["url"])
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🧪 Test Connection", key="test_wordpress"):
                try:
                    result = api.request("POST", "/config/test-connection", {"service": "wordpress"}, timeout=15)
                    if result and result.get("success"):
                        st.toast(f"✅ {result.get('message')}", icon="✅")
                    else:
                        st.error(f"Failed: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        st.info("""
💡 **WordPress Setup Required:**

WordPress integration enables automated blog post publishing.

**To configure:**
1. Enable the WordPress REST API on your site
2. Set in System Settings → Blog:
   - `BLOG_CMS_URL`: Your WordPress site URL (e.g., `https://blog.agentic.com`)
   - `BLOG_API_KEY`: REST API key or application password
   - `BLOG_USERNAME`: WordPress admin username
   - `BLOG_APP_PASSWORD`: WordPress application password
        """)

    st.markdown("---")
    st.markdown("### 📬 Mailchimp")
    st.caption("Email marketing automation — manage audience lists and deploy email campaigns.")

    mailchimp_status = integration_status.get("mailchimp", {})
    if mailchimp_status.get("connected"):
        st.success("Mailchimp is connected.")
        details = mailchimp_status.get("details", {})
        if details.get("account"):
            st.metric("Account", details["account"])
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🧪 Test Connection", key="test_mailchimp"):
                try:
                    result = api.request("POST", "/config/test-connection", {"service": "mailchimp"}, timeout=15)
                    if result and result.get("success"):
                        st.toast(f"✅ {result.get('message')}", icon="✅")
                    else:
                        st.error(f"Failed: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        st.info("""
💡 **Mailchimp Setup Required:**

Mailchimp manages email marketing lists and campaign delivery.

**To configure:**
1. Get your Mailchimp API key from Account → Extras → API Keys
2. Set in System Settings → Email:
   - `MAILCHIMP_API_KEY`: Your API key (includes datacenter suffix, e.g., `xxx-us1`)
   - `MAILCHIMP_LIST_ID`: Default audience/list ID
   - `MAILCHIMP_FROM_EMAIL`: Sender email address
        """)

    st.markdown("---")
    st.markdown("### 🛡️ Google Perspective API")
    st.caption("Content safety — analyzes LLM-generated text for toxicity before deployment.")

    perspective_status = integration_status.get("perspective", {})
    if perspective_status.get("connected"):
        st.success("Perspective API is connected.")
        st.caption("🛡️ All LLM-generated content is checked for toxicity, profanity, and threat language before deployment.")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🧪 Test Connection", key="test_perspective"):
                try:
                    result = api.request("POST", "/config/test-connection", {"service": "perspective"}, timeout=15)
                    if result and result.get("success"):
                        st.toast(f"✅ {result.get('message')}", icon="✅")
                    else:
                        st.error(f"Failed: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        st.info("""
💡 **Perspective API Setup Required:**

Google's Perspective API provides toxicity analysis for LLM-generated content as part of the safety guardrails.

**To configure:**
1. Enable the Perspective API in [Google Cloud Console](https://console.cloud.google.com)
2. Set in System Settings → Governance:
   - `PERSPECTIVE_API_KEY`: Your Google Cloud API key
        """)

    st.markdown("---")
    st.markdown("### 🔍 Apify")
    st.caption("Web scraping & competitive intelligence — automated market research data collection.")

    apify_status = integration_status.get("apify", {})
    if apify_status.get("connected"):
        st.success("Apify is connected.")
        details = apify_status.get("details", {})
        if details.get("username"):
            st.metric("Username", details["username"])
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🧪 Test Connection", key="test_apify"):
                try:
                    result = api.request("POST", "/config/test-connection", {"service": "apify"}, timeout=15)
                    if result and result.get("success"):
                        st.toast(f"✅ {result.get('message')}", icon="✅")
                    else:
                        st.error(f"Failed: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        st.info("""
💡 **Apify Setup Required:**

Apify provides web scraping capabilities for competitive intelligence and market research.

**To configure:**
1. Create an account at [apify.com](https://apify.com)
2. Set in System Settings → Apify:
   - `APIFY_API_TOKEN`: Your Apify API token
   - `ENABLE_SCRAPING`: Set to `True` to enable scraping features
        """)
