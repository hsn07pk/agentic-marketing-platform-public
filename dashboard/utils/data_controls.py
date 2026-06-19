"""
Reusable search, filter, and sort components for dashboard pages.
"""
import streamlit as st
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

def render_data_controls(
    data: List[Dict[str, Any]],
    search_fields: List[str],
    filter_configs: Optional[List[Dict[str, Any]]] = None,
    sort_options: Optional[List[str]] = None,
    key_prefix: str = "data",
    show_search: bool = True,
    show_filters: bool = True,
    show_sort: bool = True
) -> List[Dict[str, Any]]:
    """Render search, filter, and sort controls and return filtered/sorted data."""
    if not data:
        return data
    
    filtered_data = data.copy()
    
    col_search, col_sort = st.columns([3, 1])
    
    if show_search and search_fields:
        with col_search:
            search_term = st.text_input(
                "🔍 Search",
                placeholder=f"Search in {', '.join(search_fields[:3])}{'...' if len(search_fields) > 3 else ''}",
                key=f"{key_prefix}_search"
            )
            if search_term:
                search_lower = search_term.lower()
                filtered_data = [
                    item for item in filtered_data
                    if any(
                        search_lower in str(item.get(field, '')).lower()
                        for field in search_fields
                    )
                ]
    
    if show_sort and sort_options:
        with col_sort:
            sort_col1, sort_col2 = st.columns([2, 1])
            with sort_col1:
                sort_by = st.selectbox(
                    "Sort by",
                    options=sort_options,
                    key=f"{key_prefix}_sort_by",
                    label_visibility="collapsed"
                )
            with sort_col2:
                sort_order = st.selectbox(
                    "Order",
                    options=["↓ Desc", "↑ Asc"],
                    key=f"{key_prefix}_sort_order",
                    label_visibility="collapsed"
                )
            
            if sort_by:
                reverse = sort_order == "↓ Desc"
                filtered_data = sorted(
                    filtered_data,
                    key=lambda x: _get_sort_value(x.get(sort_by)),
                    reverse=reverse
                )
    
    if show_filters and filter_configs:
        with st.expander("🎛️ Filters", expanded=False):
            filter_cols = st.columns(min(len(filter_configs), 4))
            
            for idx, config in enumerate(filter_configs):
                col_idx = idx % len(filter_cols)
                with filter_cols[col_idx]:
                    filtered_data = _apply_filter(
                        filtered_data,
                        config,
                        key_prefix,
                        idx
                    )
    
    if len(filtered_data) != len(data):
        st.caption(f"Showing {len(filtered_data)} of {len(data)} items")
    
    return filtered_data

def _get_sort_value(value: Any) -> Any:

    if value is None:
        return ""
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return value.lower()
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, datetime):
        return value
    return str(value).lower()

def _apply_filter(
    data: List[Dict[str, Any]],
    config: Dict[str, Any],
    key_prefix: str,
    idx: int
) -> List[Dict[str, Any]]:

    field = config['field']
    label = config.get('label', field.replace('_', ' ').title())
    filter_type = config.get('type', 'select')
    options = config.get('options', 'auto')
    
    if options == 'auto':
        unique_values = set()
        for item in data:
            val = item.get(field)
            if val is not None:
                if isinstance(val, list):
                    unique_values.update(val)
                else:
                    unique_values.add(str(val))
        options = sorted(list(unique_values))
    
    if filter_type == 'select':
        selected = st.selectbox(
            label,
            options=["All"] + options,
            key=f"{key_prefix}_filter_{idx}"
        )
        if selected and selected != "All":
            data = [
                item for item in data
                if str(item.get(field, '')) == selected
            ]
    
    elif filter_type == 'multiselect':
        selected = st.multiselect(
            label,
            options=options,
            key=f"{key_prefix}_filter_{idx}"
        )
        if selected:
            data = [
                item for item in data
                if str(item.get(field, '')) in selected or
                (isinstance(item.get(field), list) and 
                 any(str(v) in selected for v in item.get(field, [])))
            ]
    
    elif filter_type == 'date_range':
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                f"{label} From",
                value=None,
                key=f"{key_prefix}_filter_{idx}_start"
            )
        with col2:
            end_date = st.date_input(
                f"{label} To",
                value=None,
                key=f"{key_prefix}_filter_{idx}_end"
            )
        
        if start_date or end_date:
            def in_range(item):
                val = item.get(field)
                if not val:
                    return False
                try:
                    if isinstance(val, str):
                        item_date = datetime.fromisoformat(val.replace('Z', '+00:00')).date()
                    elif isinstance(val, datetime):
                        item_date = val.date()
                    else:
                        return False
                    
                    if start_date and item_date < start_date:
                        return False
                    if end_date and item_date > end_date:
                        return False
                    return True
                except:
                    return False
            
            data = [item for item in data if in_range(item)]
    
    elif filter_type == 'number_range':
        min_val = config.get('min', 0)
        max_val = config.get('max', 100)
        
        range_vals = st.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            key=f"{key_prefix}_filter_{idx}"
        )
        
        if range_vals != (min_val, max_val):
            data = [
                item for item in data
                if range_vals[0] <= (item.get(field) or 0) <= range_vals[1]
            ]
    
    return data

def render_table_with_controls(
    data: List[Dict[str, Any]],
    columns: List[Dict[str, Any]],
    search_fields: Optional[List[str]] = None,
    filter_configs: Optional[List[Dict[str, Any]]] = None,
    sort_options: Optional[List[str]] = None,
    key_prefix: str = "table",
    page_size: int = 10
) -> None:
    """Render a data table with search, filter, sort, and pagination."""
    if not data:
        st.info("No data available")
        return
    
    if search_fields is None:
        search_fields = [col['field'] for col in columns[:3]]
    
    if sort_options is None:
        sort_options = [col['field'] for col in columns]
    
    filtered_data = render_data_controls(
        data=data,
        search_fields=search_fields,
        filter_configs=filter_configs,
        sort_options=sort_options,
        key_prefix=key_prefix
    )
    
    if not filtered_data:
        st.warning("No items match your search/filter criteria")
        return
    
    total_pages = max(1, (len(filtered_data) + page_size - 1) // page_size)
    
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            page = st.number_input(
                f"Page (1-{total_pages})",
                min_value=1,
                max_value=total_pages,
                value=1,
                key=f"{key_prefix}_page"
            )
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_data = filtered_data[start_idx:end_idx]
    else:
        page_data = filtered_data
    
    import pandas as pd
    
    table_data = []
    for item in page_data:
        row = {}
        for col in columns:
            field = col['field']
            label = col.get('label', field.replace('_', ' ').title())
            value = item.get(field, '')
            
            formatter = col.get('format')
            if formatter:
                try:
                    value = formatter(value)
                except:
                    pass
            
            row[label] = value
        table_data.append(row)
    
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

def render_searchable_select(
    items: List[Dict[str, Any]],
    display_field: str = 'name',
    id_field: str = 'id',
    label: str = "Select Item",
    search_fields: Optional[List[str]] = None,
    format_option: Optional[Callable[[Dict], str]] = None,
    show_recent: int = 5,
    key_prefix: str = "select",
    placeholder: str = "Type to search..."
) -> Optional[str]:
    """Render a searchable select dropdown with recent items and search."""
    if not items:
        st.info("No items available")
        return None
    
    if search_fields is None:
        search_fields = [display_field, id_field]
    
    search_term = st.text_input(
        f"🔍 {label}",
        placeholder=placeholder,
        key=f"{key_prefix}_search"
    )
    
    if search_term:
        search_lower = search_term.lower()
        filtered_items = [
            item for item in items
            if any(search_lower in str(item.get(field, '')).lower() for field in search_fields)
        ]
    else:
        filtered_items = items
    
    if not filtered_items:
        st.warning("No items match your search")
        return None
    
    try:
        filtered_items = sorted(
            filtered_items,
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )
    except:
        pass
    
    if format_option:
        options = {format_option(item): item.get(id_field) for item in filtered_items}
    else:
        options = {}
        for item in filtered_items:
            name = item.get(display_field, 'Unnamed')
            item_id = item.get(id_field, '')
            status = item.get('status', '')
            platform = item.get('platform', '')
            created = item.get('created_at', '')[:10] if item.get('created_at') else ''
            
            display_parts = [name]
            if platform:
                display_parts.append(f"[{platform}]")
            if status:
                display_parts.append(f"({status})")
            if created:
                display_parts.append(f"• {created}")
            
            display = " ".join(display_parts)
            options[display] = item_id
    
    if search_term:
        st.caption(f"Found {len(filtered_items)} items matching '{search_term}'")
    else:
        st.caption(f"Showing {min(len(filtered_items), 20)} of {len(items)} items (type to search)")
    
    display_options = list(options.keys())[:50]
    
    if not display_options:
        return None
    
    selected_display = st.selectbox(
        "Select",
        options=display_options,
        key=f"{key_prefix}_select",
        label_visibility="collapsed"
    )
    
    return options.get(selected_display)

def render_campaign_selector(
    api_client,
    label: str = "Select Campaign",
    key_prefix: str = "campaign",
    include_status: Optional[List[str]] = None
) -> Optional[str]:
    """Render a searchable campaign selector."""
    try:
        campaigns = api_client.get_campaigns(limit=500)
        
        if include_status:
            campaigns = [c for c in campaigns if c.get('status') in include_status]
        
        if not campaigns:
            st.info("No campaigns found")
            return None
        
        return render_searchable_select(
            items=campaigns,
            display_field='name',
            id_field='id',
            label=label,
            search_fields=['name', 'id', 'platform', 'target_persona', 'goal'],
            key_prefix=key_prefix,
            placeholder="Search by name, ID, platform, or persona..."
        )
    except Exception as e:
        st.error(f"Failed to load campaigns: {e}")
        return None
