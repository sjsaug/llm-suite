import streamlit as st
import json
import pandas as pd
import ollama
from typing import Optional

# st page config
st.set_page_config(
    page_title="Inference Comparator",
    page_icon="🦙",
    layout="wide"
)

if 'results' not in st.session_state:
    st.session_state.results = {}
    
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = []

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

# inference functoin
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

# sidebar
with st.sidebar:
    st.header("Settings")
    
    if st.button("Refresh Available Models"):
        st.cache_data.clear()
        st.rerun()
    
    base_model_names, models_info = get_available_models()
    
    if not models_info:
        st.warning("No models found. Make sure Ollama is running.")
    else:
        st.success(f"Found {len(models_info)} models across {len(base_model_names)} model families")
        
        # expandable model details section
        with st.expander("View Model Details"):
            for model in models_info:
                st.markdown(f"### {model['name']}")
                st.markdown(f"**Base Model:** {model['base_name']}")
                st.markdown(f"**Version:** {model['version']}")
                st.markdown(f"**Size:** {model['size_mb']} MB")
                
                # additional details if available
                if "family" in model:
                    st.markdown(f"**Family:** {model['family']}")
                if "parameter_size" in model:
                    st.markdown(f"**Parameter Size:** {model['parameter_size']}")
                if "quantization_level" in model:
                    st.markdown(f"**Quantization:** {model['quantization_level']}")
                if "format" in model:
                    st.markdown(f"**Format:** {model['format']}")
                    
                st.markdown("---")
    
    # model selection
    st.subheader("Select Models to Compare")
    selected_models = []
    
    # group models by base model name
    models_by_family = {}
    for model in models_info:
        base_name = model["base_name"]
        if base_name not in models_by_family:
            models_by_family[base_name] = []
        models_by_family[base_name].append(model["name"])
    
    # select sll
    if st.checkbox("Select All Models", key="select_all"):
        selected_models = [model["name"] for model in models_info]
    else:
        # expandable sections for each model family
        for base_name, versions in models_by_family.items():
            with st.expander(f"{base_name} ({len(versions)} versions)"):
                # select all versions of this model
                if st.checkbox(f"Select all {base_name} versions", key=f"select_all_{base_name}"):
                    for version in versions:
                        selected_models.append(version)
                else:
                    # create columns for checkboxes to make better use of space
                    cols = st.columns(2)
                    for i, version in enumerate(versions):
                        col_idx = i % 2
                        if cols[col_idx].checkbox(version, key=f"model_{version}"):
                            selected_models.append(version)
    
    # selected models count
    if selected_models:
        st.info(f"Selected {len(selected_models)} models: {', '.join(selected_models)}")
    else:
        st.info("No models selected")
    
    # extra params
    st.subheader("Parameters")
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    
    # optional system prompt
    st.subheader("System Prompt (Optional)")
    system_prompt = st.text_area("Enter a system prompt", 
                               "")

st.header("Enter Your Prompt")
user_prompt = st.text_area("The same prompt will be sent to all selected models", 
                          "")

col1, col2 = st.columns([1, 1])
with col1:
    compare_button = st.button("Compare Models", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("Clear Results", type="secondary", use_container_width=True)

# store results in session state to persist between reruns
if 'results' not in st.session_state:
    st.session_state.results = {}

if clear_button:
    st.session_state.results = {}
    st.rerun()

if compare_button:
    if not selected_models:
        st.warning("Please select at least one model to compare.")
    else:
        # clear prev results
        st.session_state.debug_info = []
        st.session_state.debug_info.append(f"Starting comparison with {len(selected_models)} models")
        
        with st.spinner("Generating responses..."):
            # progress bar and status text
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # function to query a model and update progress
            def query_and_track(i, total, model):
                try:
                    progress_text.text(f"Processing {i+1}/{total}: {model}")
                    st.session_state.debug_info.append(f"Processing model {i+1}/{total}: {model}")
                    response = query_model(model, user_prompt, system_prompt, temperature)
                    st.session_state.results[model] = response
                    progress_bar.progress((i + 1) / total)
                    return model, response
                except Exception as e:
                    st.session_state.debug_info.append(f"Error in query_and_track for {model}: {str(e)}")
                    st.session_state.results[model] = f"Error: {str(e)}"
                    return model, f"Error: {str(e)}"
            
            # process each model one at a time (IMPORTANT)
            for i, model in enumerate(selected_models):
                try:
                    query_and_track(i, len(selected_models), model)
                except Exception as e:
                    st.session_state.debug_info.append(f"Unhandled exception for {model}: {str(e)}")
                    st.session_state.results[model] = f"Unhandled error: {str(e)}"
            
            progress_bar.empty()
            progress_text.empty()
            st.success(f"Completed {len(selected_models)} model queries")
            
            with st.expander("Debug Information", expanded=False):
                for line in st.session_state.debug_info:
                    st.text(line)

# display results
if st.session_state.results:
    st.header("Model Responses")
    
    # show response status summary
    success_count = sum(1 for r in st.session_state.results.values() if not r.startswith("Error"))
    error_count = len(st.session_state.results) - success_count
    
    if error_count > 0:
        st.warning(f"{success_count} successful responses, {error_count} errors")
    else:
        st.success(f"All {len(st.session_state.results)} responses generated successfully")
    
    # tabs for different view modes
    tab1, tab2 = st.tabs(["Side by Side", "Stacked"])
    
    with tab1:
        # side by side view
        models_with_results = [m for m in selected_models if m in st.session_state.results]
        
        if not models_with_results:
            models_with_results = list(st.session_state.results.keys())
        
        if len(models_with_results) == 1:
            # edge case: one column for single model
            st.subheader(models_with_results[0])
            st.markdown(f"<div class='model-response'>{st.session_state.results[models_with_results[0]]}</div>", 
                       unsafe_allow_html=True)
        elif len(models_with_results) == 2:
            # edge case: two columns for two models
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(models_with_results[0])
                st.markdown(f"<div class='model-response'>{st.session_state.results[models_with_results[0]]}</div>", 
                           unsafe_allow_html=True)
            with col2:
                st.subheader(models_with_results[1])
                st.markdown(f"<div class='model-response'>{st.session_state.results[models_with_results[1]]}</div>", 
                           unsafe_allow_html=True)
        else:
            # dynamic case: grid for more than two models
            cols = st.columns(min(3, len(models_with_results)))
            for i, model in enumerate(models_with_results):
                with cols[i % len(cols)]:
                    st.subheader(model)
                    st.markdown(f"<div class='model-response'>{st.session_state.results[model]}</div>", 
                               unsafe_allow_html=True)
    
    with tab2:
        # stacked view
        for model, response in st.session_state.results.items():
            with st.expander(model, expanded=True):
                st.markdown(f"<div class='model-response'>{response}</div>", unsafe_allow_html=True)
                
    # export options
    st.header("Export Results")
    
    # convert results to DataFrame for export
    results_df = pd.DataFrame({
        "Model": list(st.session_state.results.keys()),
        "Response": list(st.session_state.results.values())
    })
    
    # export as CSV
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="ollama_model_comparison.csv",
        mime="text/csv",
    )
    
    # export as JSON
    json_results = json.dumps({
        "prompt": user_prompt,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "results": st.session_state.results
    }, indent=2)
    
    st.download_button(
        label="Download as JSON",
        data=json_results,
        file_name="ollama_model_comparison.json",
        mime="application/json",
    )