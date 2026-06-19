"""
Confirmation dialog component for destructive actions
"""
import streamlit as st
from typing import Callable, Optional


def confirm_action(
    action_name: str,
    confirm_message: str,
    on_confirm: Callable,
    danger: bool = False,
    button_text: str = "Confirm",
    key: Optional[str] = None
) -> bool:
    """
    Show a confirmation dialog before executing an action
    
    Args:
        action_name: Name of the action (e.g., "Delete Campaign")
        confirm_message: Message to display
        on_confirm: Function to call when confirmed
        danger: Whether this is a dangerous action (red button)
        button_text: Text for confirm button
        key: Unique key for the dialog
        
    Returns:
        True if confirmed and executed, False otherwise
    """
    unique_key = key or f"confirm_{action_name.lower().replace(' ', '_')}"
    
    if st.button(action_name, key=f"trigger_{unique_key}", type="primary" if not danger else "secondary"):
        st.session_state[f"{unique_key}_show_confirm"] = True
    
    if st.session_state.get(f"{unique_key}_show_confirm", False):
        with st.container():
            st.warning(confirm_message)
            
            col1, col2, col3 = st.columns([1, 1, 3])
            
            with col1:
                if st.button(button_text, key=f"yes_{unique_key}", type="primary"):
                    st.session_state[f"{unique_key}_show_confirm"] = False
                    try:
                        on_confirm()
                        st.success(f"{action_name} completed successfully!")
                        return True
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        return False
            
            with col2:
                if st.button("Cancel", key=f"no_{unique_key}"):
                    st.session_state[f"{unique_key}_show_confirm"] = False
                    st.info("Action cancelled")
                    return False
    
    return False


def show_warning_dialog(
    title: str,
    message: str,
    severity: str = "warning"
) -> None:
    """
    Show a warning/info dialog
    
    Args:
        title: Dialog title
        message: Dialog message
        severity: "warning", "error", "info", or "success"
    """
    severity_functions = {
        "warning": st.warning,
        "error": st.error,
        "info": st.info,
        "success": st.success
    }
    
    func = severity_functions.get(severity, st.warning)
    func(f"**{title}**\n\n{message}")


def show_confirmation_prompt(
    question: str,
    key: str
) -> Optional[bool]:
    """
    Show a simple yes/no confirmation prompt
    
    Args:
        question: Question to ask
        key: Unique key for the prompt
        
    Returns:
        True if yes, False if no, None if not answered yet
    """
    st.write(question)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Yes", key=f"yes_{key}", type="primary"):
            return True
    
    with col2:
        if st.button("No", key=f"no_{key}"):
            return False
    
    return None
