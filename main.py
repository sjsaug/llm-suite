import streamlit as st
import json
import pandas as pd
import ollama
from typing import Optional
import html

# st page config
st.set_page_config(
    page_title="Inference Comparator",
    page_icon="🦙",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

if 'results' not in st.session_state:
    st.session_state.results = {}
    
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = []

if 'current_streaming_text' not in st.session_state:
    st.session_state.current_streaming_text = ""

# custom CSS
st.markdown("""
<style>
    .model-response {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .stTextArea textarea {
        height: 150px;
    }
    .title {
        text-align: center;
        margin-bottom: 30px;
    }
    /* Tooltip container */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        margin-top: 0px;
        color: #0068c9;
        font-size: 0.9rem;
    }
    /* Tooltip text */
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 90%;
        background-color: #f0f2f6;
        color: #333;
        text-align: left;
        padding: 10px;
        border-radius: 6px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        
        /* Position the tooltip */
        position: absolute;
        z-index: 1;
        top: 125%;
        left: 0;
    }
    /* Show the tooltip text when you mouse over the tooltip container */
    .tooltip:hover .tooltiptext {
        visibility: visible;
    }
    /* Fix checkbox alignment */
    .model-label {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .model-label input {
        margin: 0;
    }
    /* Bold select all text */
    .select-all-bold {
        font-weight: bold;
    }
    /* Blue button styling */
    .stButton>button {
        border-color: #0068c9;
        color: #0068c9;
    }
    .stButton>button[data-baseweb="button"] {
        border-color: #0068c9;
    }
    /* Hide Streamlit menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Inference Comparator</h1>", unsafe_allow_html=True)

# get list of available models with details (https://github.com/ollama/ollama-python/blob/main/examples/list.py)
@st.cache_data(ttl=300)  # cache for 5 mins
def get_available_models():
    try:
        response = ollama.list()
        models_info = []
        model_names = set()  # track unique base model names
        
        for model in response.models:
            # parse model name and version if present (format: model:version)
            full_name = model.model
            if ":" in full_name:
                base_name, version = full_name.split(":", 1)
            else:
                base_name, version = full_name, "latest"
            
            # Add base name to unique set
            model_names.add(base_name)
            
            model_info = {
                "name": full_name,  # full name with version
                "base_name": base_name,  # base model name
                "version": version,  # version tag
                "size_mb": round(model.size.real / 1024 / 1024, 2)
            }

            if model.details:
                model_info["format"] = model.details.format
                model_info["family"] = model.details.family
                model_info["parameter_size"] = model.details.parameter_size
                model_info["quantization_level"] = model.details.quantization_level
            models_info.append(model_info)
        
        # sort models by base name and then by version
        models_info.sort(key=lambda x: (x["base_name"], x["version"]))
            
        return list(model_names), models_info
    except Exception as e:
        st.error(f"Error connecting to Ollama: {e}")
        return [], []

# Function to create HTML tooltip for model details
def create_model_tooltip(model_info):
    tooltip_content = f"""
    <b>Base Model:</b> {model_info['base_name']}<br>
    <b>Version:</b> {model_info['version']}<br>
    <b>Size:</b> {model_info['size_mb']} MB<br>
    """
    
    # Add additional details if available
    if "family" in model_info:
        tooltip_content += f"<b>Family:</b> {model_info['family']}<br>"
    if "parameter_size" in model_info:
        tooltip_content += f"<b>Parameter Size:</b> {model_info['parameter_size']}<br>"
    if "quantization_level" in model_info:
        tooltip_content += f"<b>Quantization:</b> {model_info['quantization_level']}<br>"
    if "format" in model_info:
        tooltip_content += f"<b>Format:</b> {model_info['format']}<br>"
    
    return tooltip_content

# non-streaming inference function
def query_model(model_name: str, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> str:
    try:
        # debugging info (can remove later)
        st.session_state.debug_info.append(f"Querying model: {model_name}")
        
        # prepare parameters - use options dict for temperature
        params = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }
        
        # add system prompt if provided
        if system_prompt and system_prompt.strip():
            params["system"] = system_prompt
        
        st.session_state.debug_info.append(f"Sending request to {model_name}...")
        st.session_state.debug_info.append(f"Parameters: {params}")
        response = ollama.generate(**params)
        
        # check response
        st.session_state.debug_info.append(f"Got response from {model_name}, length: {len(response.response)}")
        return response.response
    except Exception as e:
        error_msg = f"Exception with {model_name}: {str(e)}"
        st.session_state.debug_info.append(error_msg)
        # Try alternative approach if the API has changed
        try:
            st.session_state.debug_info.append("Trying alternative API approach...")
            # Simpler approach without temperature
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                system=system_prompt if system_prompt and system_prompt.strip() else None
            )
            st.session_state.debug_info.append(f"Success with alternative approach! Response length: {len(response.response)}")
            return response.response
        except Exception as e2:
            error_msg = f"Alternative approach also failed: {str(e2)}"
            st.session_state.debug_info.append(error_msg)
            return f"Error: Unable to query model. Original error: {str(e)}, Alternative error: {str(e2)}"

# streaming inference function
def query_model_streaming(model_name: str, prompt: str, system_prompt: Optional[str] = None, 
                         temperature: float = 0.7, progress_text=None, response_length=None, 
                         streaming_display=None):
    try:
        # debugging info
        st.session_state.debug_info.append(f"Streaming from model: {model_name}")
        
        # prepare parameters
        params = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature}
        }
        
        # add system prompt if provided
        if system_prompt and system_prompt.strip():
            params["system"] = system_prompt
        
        st.session_state.debug_info.append(f"Sending streaming request to {model_name}...")
        
        # Initialize response text
        full_response = ""
        
        # Stream the response
        for chunk in ollama.generate(**params):
            if chunk and 'response' in chunk:
                text_chunk = chunk['response']
                full_response += text_chunk
                
                # Update the streaming display with current text
                if streaming_display:
                    st.session_state.current_streaming_text = full_response
                    streaming_display.markdown(f"<div class='model-response'>{full_response}</div>", 
                                               unsafe_allow_html=True)
                
                # Update the combined progress text and length counter
                if progress_text:
                    progress_text.markdown(f"**Model:** {model_name} | Response length: {len(full_response)} characters")
        
        st.session_state.debug_info.append(f"Completed streaming from {model_name}, final length: {len(full_response)}")
        return full_response
    except Exception as e:
        error_msg = f"Streaming exception with {model_name}: {str(e)}"
        st.session_state.debug_info.append(error_msg)
        return f"Error: Unable to stream from model. Error: {str(e)}"

# sidebar
with st.sidebar:
    st.header("Settings")
    
    # model selection
    st.subheader("Select Models to Compare")
    selected_models = []
    
    # get models
    base_model_names, models_info = get_available_models()
    
    if not models_info:
        st.warning("No models found. Make sure Ollama is running.")
    
    # group models by base model name
    models_by_family = {}
    model_info_by_name = {}  # For quick lookups by model name
    
    for model in models_info:
        base_name = model["base_name"]
        model_info_by_name[model["name"]] = model
        if base_name not in models_by_family:
            models_by_family[base_name] = []
        models_by_family[base_name].append(model["name"])
    
    # select all - with bold text
    if st.checkbox("", key="select_all"):
        st.markdown('<span class="select-all-bold">Select All Models</span>', unsafe_allow_html=True)
        selected_models = [model["name"] for model in models_info]
    else:
        st.markdown('<span class="select-all-bold">Select All Models</span>', unsafe_allow_html=True)
        
        # Handle each model family - always use expanders
        for base_name, versions in models_by_family.items():
            if len(versions) == 1:
                # Even for single model, use an expander
                with st.expander(f"{base_name} (1 version)", expanded=True):
                    model_name = versions[0]
                    model_info = model_info_by_name[model_name]
                    tooltip_content = create_model_tooltip(model_info)
                    
                    # Use columns for displaying model and details
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if st.checkbox(model_name, key=f"model_{model_name}"):
                            selected_models.append(model_name)
                    with col2:
                        # Add tooltip that attaches to model name
                        st.markdown(f"""<div class="tooltip">ℹ️
                        <span class="tooltiptext">{tooltip_content}</span>
                        </div>""", unsafe_allow_html=True)
            else:
                # For multiple versions, use expander
                with st.expander(f"{base_name} ({len(versions)} versions)", expanded=True):
                    # select all versions of this model
                    if st.checkbox(f"Select all {base_name} versions", key=f"select_all_{base_name}"):
                        for version in versions:
                            selected_models.append(version)
                    
                    # create columns for checkboxes to make better use of space
                    cols = st.columns(2)
                    for i, version in enumerate(versions):
                        col_idx = i % 2
                        model_info = model_info_by_name[version]
                        tooltip_content = create_model_tooltip(model_info)
                        
                        with cols[col_idx]:
                            # Create a checkbox with label
                            if st.checkbox(version, key=f"model_{version}"):
                                selected_models.append(version)
                            
                            # Add a small tooltip icon with details
                            st.markdown(f"""<div class="tooltip">ℹ️
                            <span class="tooltiptext">{tooltip_content}</span>
                            </div>""", unsafe_allow_html=True)
    
    # selected models count
    if selected_models:
        st.info(f"Selected {len(selected_models)} models: {', '.join(selected_models)}")
    else:
        st.info("No models selected")
    
    # extra params
    st.subheader("Parameters")
    
    # Add streaming option (checked by default)
    enable_streaming = st.checkbox("Enable streaming", value=True, 
                                  help="Show responses as they are generated. You'll see the text being generated in real-time.")
    
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    
    # optional system prompt
    st.subheader("System Prompt (Optional)")
    system_prompt = st.text_area("Enter a system prompt", 
                               "")
    
    # Move refresh models button and model count to bottom
    st.subheader("Model Management")
    
    if st.button("Refresh Available Models"):
        st.cache_data.clear()
        st.rerun()
        
    if models_info:
        st.success(f"Found {len(models_info)} models across {len(base_model_names)} model families")

st.header("Enter Your Prompt")
user_prompt = st.text_area("The same prompt will be sent to all selected models", 
                          "")

col1, col2 = st.columns([1, 1])
with col1:
    compare_button = st.button("Compare Models", use_container_width=True)
with col2:
    clear_button = st.button("Clear Results", type="secondary", use_container_width=True)

# store results in session state to persist between reruns
if 'results' not in st.session_state:
    st.session_state.results = {}

if clear_button:
    st.session_state.results = {}
    st.session_state.current_streaming_text = ""
    st.rerun()

# Section for live streaming display
streaming_section = st.empty()

if compare_button:
    if not selected_models:
        st.warning("Please select at least one model to compare.")
    else:
        # clear prev results
        st.session_state.debug_info = []
        st.session_state.debug_info.append(f"Starting comparison with {len(selected_models)} models")
        
        with st.spinner("Generating responses..."):
            # progress bar and status text container (combined)
            progress_container = st.empty()
            progress_bar = st.progress(0)
            
            # Set up streaming if enabled
            streaming_display = None
            if enable_streaming:
                # Create container for streaming display
                streaming_display = streaming_section.empty()
            
            # process each model one at a time (IMPORTANT)
            for i, model in enumerate(selected_models):
                try:
                    if enable_streaming:
                        # Combined progress text and length counter on same line
                        progress_container.markdown(f"**Model {i+1}/{len(selected_models)}:** {model} | Response length: 0 characters")
                        
                        # Reset streaming display for next model
                        st.session_state.current_streaming_text = ""
                        
                        response = query_model_streaming(
                            model, 
                            user_prompt, 
                            system_prompt, 
                            temperature, 
                            progress_container,  # Pass the container for updates
                            progress_container,  # Use same container for length updates
                            streaming_display
                        )
                    else:
                        # Only show "Processing" message when not streaming
                        progress_container.markdown(f"**Generating {i+1}/{len(selected_models)}:** {model}")
                        st.session_state.debug_info.append(f"Processing model {i+1}/{len(selected_models)}: {model}")
                        response = query_model(model, user_prompt, system_prompt, temperature)
                    
                    st.session_state.results[model] = response
                    progress_bar.progress((i + 1) / len(selected_models))
                    
                except Exception as e:
                    st.session_state.debug_info.append(f"Unhandled exception for {model}: {str(e)}")
                    st.session_state.results[model] = f"Unhandled error: {str(e)}"
            
            # Clear progress indicators after completion
            progress_bar.empty()
            progress_container.empty()
            if streaming_display:
                streaming_display.empty()
            
            with st.expander("Debug Information", expanded=False):
                for line in st.session_state.debug_info:
                    st.text(line)

# display results
if st.session_state.results:
    st.header("Model Responses")
    
    # Clear streaming section once we have results
    streaming_section.empty()
    
    # show response status summary only if there are errors
    success_count = sum(1 for r in st.session_state.results.values() if not r.startswith("Error"))
    error_count = len(st.session_state.results) - success_count
    
    if error_count > 0:
        st.warning(f"{success_count} successful responses, {error_count} errors")
    
    # tabs for different view modes
    tab1, tab2 = st.tabs(["Side by Side", "Stacked"])
    
    with tab1:
        # side by side view
        models_with_results = [m for m in selected_models if m in st.session_state.results]
        
        if not models_with_results:
            models_with_results = list(st.session_state.results.keys())
        
        if len(models_with_results) == 1:
            # edge case: one column for single model
            model_name = models_with_results[0]
            response = st.session_state.results[model_name]
            st.subheader(f"{model_name} ({len(response)} chars)")
            st.markdown(f"<div class='model-response'>{response}</div>", 
                       unsafe_allow_html=True)
        elif len(models_with_results) == 2:
            # Two models - show side by side in two columns
            col1, col2 = st.columns(2)
            with col1:
                model_name = models_with_results[0]
                response = st.session_state.results[model_name]
                st.subheader(f"{model_name} ({len(response)} chars)")
                st.markdown(f"<div class='model-response'>{response}</div>", 
                           unsafe_allow_html=True)
            with col2:
                model_name = models_with_results[1]
                response = st.session_state.results[model_name]
                st.subheader(f"{model_name} ({len(response)} chars)")
                st.markdown(f"<div class='model-response'>{response}</div>", 
                           unsafe_allow_html=True)
        else:
            # More than two models - show two per row
            for i in range(0, len(models_with_results), 2):
                # Create a new row with two columns
                row_cols = st.columns(2)
                
                # First column in the row
                with row_cols[0]:
                    model_name = models_with_results[i]
                    response = st.session_state.results[model_name]
                    st.subheader(f"{model_name} ({len(response)} chars)")
                    st.markdown(f"<div class='model-response'>{response}</div>", 
                               unsafe_allow_html=True)
                
                # Second column in the row (if available)
                if i + 1 < len(models_with_results):
                    with row_cols[1]:
                        model_name = models_with_results[i + 1]
                        response = st.session_state.results[model_name]
                        st.subheader(f"{model_name} ({len(response)} chars)")
                        st.markdown(f"<div class='model-response'>{response}</div>", 
                                   unsafe_allow_html=True)
    
    with tab2:
        # stacked view
        for model, response in st.session_state.results.items():
            with st.expander(f"{model} ({len(response)} chars)", expanded=True):
                st.markdown(f"<div class='model-response'>{response}</div>", unsafe_allow_html=True)
                
    # export options
    st.header("Export Results")
    
    # convert results to DataFrame for export
    results_df = pd.DataFrame({
        "Model": list(st.session_state.results.keys()),
        "Response": list(st.session_state.results.values()),
        "Length": [len(response) for response in st.session_state.results.values()]
    })
    
    # Place download buttons side by side
    col1, col2 = st.columns(2)
    
    # export as CSV
    csv = results_df.to_csv(index=False)
    with col1:
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="ollama_model_comparison.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # export as JSON
    json_results = json.dumps({
        "prompt": user_prompt,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "results": st.session_state.results,
        "response_lengths": {model: len(response) for model, response in st.session_state.results.items()}
    }, indent=2)
    
    with col2:
        st.download_button(
            label="Download as JSON",
            data=json_results,
            file_name="ollama_model_comparison.json",
            mime="application/json",
            use_container_width=True
        )