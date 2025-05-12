import streamlit as st
import json
import pandas as pd
import ollama
from typing import Optional
import html

# st page config
st.set_page_config(
    page_title="LLM Suite",
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
    
    /* Hide Streamlit menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>LLM Suite</h1>", unsafe_allow_html=True)

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

# --- Sidebar Tabs Navigation ---
with st.sidebar:
    st.markdown("## LLM Suite")
    sidebar_tabs = st.tabs(["Models", "Settings"])

    # --- Models Tab ---
    with sidebar_tabs[0]:
        selected_models = []
        st.subheader("Select Models to Compare")
        base_model_names, models_info = get_available_models()
        if not models_info:
            st.warning("No models found. Make sure Ollama is running.")
        models_by_family = {}
        model_info_by_name = {}
        for model in models_info:
            base_name = model["base_name"]
            model_info_by_name[model["name"]] = model
            if base_name not in models_by_family:
                models_by_family[base_name] = []
            models_by_family[base_name].append(model["name"])
        select_all = st.checkbox("Select All Models", key="select_all", value=False)
        if select_all:
            selected_models = [model["name"] for model in models_info]
        else:
            for base_name, versions in models_by_family.items():
                if len(versions) == 1:
                    with st.expander(f"{base_name} (1 version)", expanded=True):
                        model_name = versions[0]
                        model_info = model_info_by_name[model_name]
                        help_text = f"""
                        Base Model: {model_info['base_name']}
                        Version: {model_info['version']} 
                        Size: {model_info['size_mb']} MB
                        """
                        if "family" in model_info:
                            help_text += f"\nFamily: {model_info['family']}"
                        if "parameter_size" in model_info:
                            help_text += f"\nParameter Size: {model_info['parameter_size']}"
                        if "quantization_level" in model_info:
                            help_text += f"\nQuantization: {model_info['quantization_level']}"
                        if "format" in model_info:
                            help_text += f"\nFormat: {model_info['format']}"
                        if st.checkbox(f"{model_name}", key=f"model_{model_name}", help=help_text):
                            selected_models.append(model_name)
                else:
                    with st.expander(f"{base_name} ({len(versions)} versions)", expanded=True):
                        select_all_family = st.checkbox(
                            f"Select all {base_name} versions",
                            key=f"select_all_{base_name}",
                            value=False
                        )
                        if select_all_family:
                            # If select all family is checked, add all versions and hide individual checkboxes
                            selected_models.extend([version for version in versions if version not in selected_models])
                        else:
                            # Show individual checkboxes for each version
                            for version in versions:
                                model_info = model_info_by_name[version]
                                help_text = f"""
                                Base Model: {model_info['base_name']}
                                Version: {model_info['version']} 
                                Size: {model_info['size_mb']} MB
                                """
                                if "family" in model_info:
                                    help_text += f"\nFamily: {model_info['family']}"
                                if "parameter_size" in model_info:
                                    help_text += f"\nParameter Size: {model_info['parameter_size']}"
                                if "quantization_level" in model_info:
                                    help_text += f"\nQuantization: {model_info['quantization_level']}"
                                if "format" in model_info:
                                    help_text += f"\nFormat: {model_info['format']}"
                                if st.checkbox(f"{version}", key=f"model_{version}", help=help_text):
                                    selected_models.append(version)
                                elif version in selected_models:
                                    selected_models.remove(version)
        if selected_models:
            if len(selected_models) == 1:
                st.info(f"Selected 1 model: {selected_models[0]}")
            else:
                st.info(f"Selected {len(selected_models)} models: {', '.join(selected_models)}")
        else:
            st.info("No models selected")

    # --- Settings Tab ---
    with sidebar_tabs[1]:
        st.subheader("Parameters")
        enable_streaming = st.checkbox("Enable streaming", value=True, 
                                      help="Show responses as they are generated. You'll see the text being generated in real-time.",
                                      key="enable_streaming")
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1, key="temperature")
        st.subheader("System Prompt (Optional)")
        system_prompt = st.text_area("Enter a system prompt", "", key="system_prompt")
        st.subheader("Model Management")
        if st.button("Refresh Available Models", key="refresh_models"):
            st.cache_data.clear()
            st.rerun()
        base_model_names, models_info = get_available_models()
        if models_info:
            st.success(f"Found {len(models_info)} models across {len(base_model_names)} model families")

# --- Ensure settings variables are available regardless of sidebar tab ---
if 'enable_streaming' not in st.session_state:
    st.session_state.enable_streaming = True
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = ""

enable_streaming = st.session_state.enable_streaming
temperature = st.session_state.temperature
system_prompt = st.session_state.system_prompt

st.header("Enter Your Prompt")
user_prompt = st.text_area("The same prompt will be sent to all selected models", 
                          "")

# Replace the two buttons with a single Compare Models button
compare_button = st.button("Compare Models", use_container_width=True)

# Add a stop button that only shows during inference
stop_button_container = st.empty()

# Move download buttons to this location
download_buttons_container = st.container()

# store results in session state to persist between reruns
if 'results' not in st.session_state:
    st.session_state.results = {}

# Add a "running" flag to track inference status
if 'inference_running' not in st.session_state:
    st.session_state.inference_running = False

# MOVED: Progress container and bar ABOVE streaming section
progress_container = st.empty()
progress_bar = st.empty()

# Section for live streaming display
streaming_section = st.empty()

# Add a stop inference handler
if 'stop_inference' not in st.session_state:
    st.session_state.stop_inference = False

# Define the model processing function before using it
def process_models():
    # Set up streaming if enabled
    streaming_display = None
    if enable_streaming:
        # Create container for streaming display
        streaming_display = streaming_section.empty()
    
    # process each model one at a time (IMPORTANT)
    for i, model in enumerate(selected_models):
        # Check if inference was stopped
        if st.session_state.stop_inference:
            st.session_state.debug_info.append("Inference stopped by user")
            break
            
        try:
            if enable_streaming:
                # Combined progress text and length counter on same line
                progress_container.markdown(f"**Model {i+1}/{len(selected_models)}:** {model} | Response length: 0 characters")
                progress_bar.progress((i) / len(selected_models))
                
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
                # Show processing message when not streaming
                progress_container.markdown(f"**Generating {i+1}/{len(selected_models)}:** {model}")
                progress_bar.progress((i) / len(selected_models))
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
    
    # Fixed debug information expander
    with st.expander("Debug Information"):
        st.code("\n".join(st.session_state.debug_info))
    
    # Clear stop button when done
    stop_button_container.empty()
    
    # Set inference running flag to false
    st.session_state.inference_running = False

if compare_button:
    if not selected_models:
        st.warning("Please select at least one model to compare.")
    else:
        # Clear previous results first (was previously done by the Clear Results button)
        st.session_state.results = {}
        st.session_state.current_streaming_text = ""
        
        # Reset stop flag
        st.session_state.stop_inference = False
        # Set inference running flag
        st.session_state.inference_running = True
        
        # Show stop button
        with stop_button_container:
            if st.button("Stop Inference", type="primary", use_container_width=True):
                st.session_state.stop_inference = True
                st.info("Stopping inference... please wait.")
                st.rerun()
                
        # clear prev results
        st.session_state.debug_info = []
        st.session_state.debug_info.append(f"Starting comparison with {len(selected_models)} models")
        
        # Only use spinner when NOT streaming to avoid the "In Progress..." text
        if not enable_streaming:
            with st.spinner("Generating responses..."):
                process_models()
        else:
            # When streaming, don't use the spinner so no "In Progress..." appears
            process_models()

# Only show download buttons when we have results
with download_buttons_container:
    if st.session_state.results:
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
            st.markdown(f"<div class='model-response'>{html.escape(response)}</div>", 
                       unsafe_allow_html=True)
        elif len(models_with_results) == 2:
            # Two models - show side by side in two columns
            col1, col2 = st.columns(2)
            with col1:
                model_name = models_with_results[0]
                response = st.session_state.results[model_name]
                st.subheader(f"{model_name} ({len(response)} chars)")
                st.markdown(f"<div class='model-response'>{html.escape(response)}</div>", 
                           unsafe_allow_html=True)
            with col2:
                model_name = models_with_results[1]
                response = st.session_state.results[model_name]
                st.subheader(f"{model_name} ({len(response)} chars)")
                st.markdown(f"<div class='model-response'>{html.escape(response)}</div>", 
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
                    st.markdown(f"<div class='model-response'>{html.escape(response)}</div>", 
                               unsafe_allow_html=True)
                
                # Second column in the row (if available)
                if i + 1 < len(models_with_results):
                    with row_cols[1]:
                        model_name = models_with_results[i + 1]
                        response = st.session_state.results[model_name]
                        st.subheader(f"{model_name} ({len(response)} chars)")
                        st.markdown(f"<div class='model-response'>{html.escape(response)}</div>", 
                                   unsafe_allow_html=True)
    
    with tab2:
        # stacked view
        for model, response in st.session_state.results.items():
            with st.expander(f"{model} ({len(response)} chars)", expanded=True):
                st.markdown(f"<div class='model-response'>{html.escape(response)}</div>", unsafe_allow_html=True)