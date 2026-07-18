import os
import streamlit.components.v1 as components

# Declare the custom component
_component_func = components.declare_component(
    "ambient_speech",
    path=os.path.dirname(os.path.abspath(__file__))
)

def live_speech_component(key=None):
    """
    Renders the live speech recognition component.
    Returns the accumulated transcript string.
    """
    return _component_func(key=key, default="")
