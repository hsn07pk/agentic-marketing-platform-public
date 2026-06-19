"""
Enhanced data table component with sorting and filtering
"""
import streamlit as st
import pandas as pd
from typing import List, Optional, Dict, Any

def render_data_table(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    sortable: bool = True,
    filterable: bool = False,
    hide_index: bool = True,
    height: Optional[int] = None,
    column_config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Render an enhanced data table with sorting and filtering options."""
    if data is None or data.empty:
        st.info("No data available")
        return pd.DataFrame()
    
    df = data.copy()
    
    if columns:
        available_cols = [col for col in columns if col in df.columns]
        df = df[available_cols]
    
    if filterable and not df.empty:
        with st.expander("🔍 Filter Data", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                search_term = st.text_input("Search", placeholder="Search all columns...")
                if search_term:
                    mask = df.astype(str).apply(
                        lambda x: x.str.contains(search_term, case=False, na=False)
                    ).any(axis=1)
                    df = df[mask]
            
            with col2:
                if st.checkbox("Advanced Filters"):
                    for col in df.columns:
                        if df[col].dtype in ['object', 'string']:
                            unique_vals = df[col].unique()
                            if len(unique_vals) < 20:
                                selected = st.multiselect(
                                    f"Filter {col}",
                                    options=unique_vals,
                                    key=f"filter_{col}"
                                )
                                if selected:
                                    df = df[df[col].isin(selected)]
    
    if sortable and not df.empty:
        sort_col = st.selectbox(
            "Sort by",
            options=list(df.columns),
            key="sort_column"
        )
        
        if sort_col:
            sort_order = st.radio(
                "Order",
                options=["Ascending", "Descending"],
                horizontal=True,
                key="sort_order"
            )
            ascending = sort_order == "Ascending"
            df = df.sort_values(by=sort_col, ascending=ascending)
    
    if height:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=hide_index,
            height=height,
            column_config=column_config
        )
    else:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=hide_index,
            column_config=column_config
        )
    
    if not df.empty:
        st.caption(f"Showing {len(df)} rows")
    
    return df

def render_simple_table(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    max_rows: int = 10
) -> None:
    """Render a simple table without advanced features."""
    if data is None or data.empty:
        st.info("No data available")
        return
    
    df = data.copy()
    
    if columns:
        available_cols = [col for col in columns if col in df.columns]
        df = df[available_cols]
    
    if len(df) > max_rows:
        df = df.head(max_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {max_rows} of {len(data)} rows")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"{len(df)} rows")

def export_to_csv(data: pd.DataFrame, filename: str = "data.csv") -> None:
    """Add a download button for CSV export."""
    if data is None or data.empty:
        return
    
    csv = data.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=f"download_{filename}"
    )
