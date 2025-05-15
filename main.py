import streamlit as st
import pandas as pd
import ollama
from typing import Optional
import html
import requests
import subprocess

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

# Initialize session state variables if they don't exist
if 'results' not in st.session_state:
    st.session_state.results = {}
    
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = []

if 'current_streaming_text' not in st.session_state:
    st.session_state.current_streaming_text = ""

if 'inference_running' not in st.session_state:
    st.session_state.inference_running = False

if 'stop_inference' not in st.session_state:
    st.session_state.stop_inference = False

if 'evaluation_result' not in st.session_state:
    st.session_state.evaluation_result = None

if 'download_status' not in st.session_state:
    st.session_state.download_status = None

if 'remove_status' not in st.session_state:
    st.session_state.remove_status = None

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

# get list of available models with details
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

# Function to get available models from Ollama repository
@st.cache_data(ttl=600)  # cache for 10 mins
def get_ollama_available_models():
    try:
        response = requests.get("https://ollama.com/search", timeout=10)
        if response.status_code == 200:
            st.error("Unable to fetch remote model list: Ollama loads models dynamically and does not provide a public API for this. Please check https://ollama.com/library manually.")
            return []
        else:
            st.error(f"Failed to fetch models: HTTP {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching available models: {e}")
        return []

# Function to download model from Ollama
def download_model(model_name):
    try:
        with st.spinner(f"Downloading {model_name}... This may take a while depending on the model size."):
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                check=True
            )
            return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.stderr}"
    except Exception as e:
        return False, f"Error: {str(e)}"

# Function to remove model from Ollama
def remove_model(model_name):
    try:
        with st.spinner(f"Removing {model_name}..."):
            result = subprocess.run(
                ["ollama", "rm", model_name],
                capture_output=True,
                text=True,
                check=True
            )
            return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.stderr}"
    except Exception as e:
        return False, f"Error: {str(e)}"

# non-streaming inference function
def query_model(model_name: str, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> str:
    try:
        # prepare parameters
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
        response = ollama.generate(**params)
        
        return response.response
    except Exception as e:
        error_msg = f"Exception with {model_name}: {str(e)}"
        st.session_state.debug_info.append(error_msg)
        return f"Error: Unable to query model. {str(e)}"

# streaming inference function
def query_model_streaming(model_name: str, prompt: str, system_prompt: Optional[str] = None, 
                         temperature: float = 0.7, progress_container=None, 
                         streaming_display=None):
    try:
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
        
        # Initialize response text
        full_response = ""
        
        # Stream the response
        for chunk in ollama.generate(**params):
            # Check if inference was stopped
            if st.session_state.stop_inference:
                break
                
            if chunk and 'response' in chunk:
                text_chunk = chunk['response']
                full_response += text_chunk
                
                # Update the streaming display with current text
                if streaming_display:
                    st.session_state.current_streaming_text = full_response
                    streaming_display.markdown(f"<div class='model-response'>{full_response}</div>", 
                                               unsafe_allow_html=True)
                
                # Update the progress text and length counter
                if progress_container:
                    progress_container.markdown(f"**Model:** {model_name} | Response length: {len(full_response)} characters")
        
        return full_response
    except Exception as e:
        error_msg = f"Streaming exception with {model_name}: {str(e)}"
        st.session_state.debug_info.append(error_msg)
        return f"Error: Unable to stream from model. {str(e)}"

# --- Function to evaluate model responses ---
def evaluate_responses(evaluation_model: str, responses: dict, user_prompt: str, evaluation_prompt: str, temperature: float = 0.7) -> str:
    try:
        # Prepare the prompt for evaluation
        formatted_responses = ""
        for i, (model_name, response) in enumerate(responses.items(), 1):
            formatted_responses += f"\n\n--- MODEL {i}: {model_name} ---\n{response}"
        
        full_prompt = f"""Original User Prompt: {user_prompt}

The following are responses from different LLM models to this prompt:
{formatted_responses}

Based on these responses, please provide your evaluation.
"""
        
        # Call the evaluation model
        st.session_state.debug_info.append(f"Sending evaluation request to {evaluation_model}...")
        return query_model(evaluation_model, full_prompt, evaluation_prompt, temperature)
    except Exception as e:
        error_msg = f"Evaluation exception with {evaluation_model}: {str(e)}"
        st.session_state.debug_info.append(error_msg)
        return f"Error: Unable to perform evaluation. {str(e)}"

# --- Sidebar Tabs Navigation ---
with st.sidebar:
    st.markdown("## LLM Suite")
    sidebar_tabs = st.tabs(["Models", "Model Management", "Settings"])

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
                        Family: {model_info['family']}
                        Parameter Size: {model_info['parameter_size']}
                        Quantization: {model_info['quantization_level']}
                        Format: {model_info['format']}
                        """
                        if "format" in model_info:
                            help_text += f"\n{model_info['base_name']}:{model_info['version']}"
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
                            # If select all family is checked, add all versions
                            selected_models.extend([version for version in versions if version not in selected_models])
                        else:
                            # Show individual checkboxes for each version
                            for version in versions:
                                model_info = model_info_by_name[version]
                                help_text = f"""
                                Base Model: {model_info['base_name']}
                            Version: {model_info['version']} 
                            Size: {model_info['size_mb']} MB
                            Family: {model_info['family']}
                            Parameter Size: {model_info['parameter_size']}
                            Quantization: {model_info['quantization_level']}
                            Format: {model_info['format']}
                                """
                                if "format" in model_info:
                                    help_text += f"\n{model_info['base_name']}:{model_info['version']}"
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

    # --- Model Management Tab ---
    with sidebar_tabs[1]:
        st.subheader("Model Management")

        if st.button("Refresh Available Models", key="refresh_models"):
            st.cache_data.clear()
            st.session_state.download_status = None
            st.session_state.remove_status = None
            st.rerun()

        
        # Display current installed models
        st.markdown("#### Installed Models")
        _, installed_models = get_available_models()
        
        if installed_models:
            # Show installed models with info tooltips and checkboxes for removal
            remove_checks = {}
            for model in installed_models:
                # Create the same help_text format as in the Models tab
                help_text = f"""
                        Base Model: {model.get('base_name','')}
                        Version: {model.get('version','')}
                        Size: {model.get('size_mb','')} MB
                        Family: {model.get('family','')}
                        Parameter Size: {model.get('parameter_size','')}
                        Quantization: {model.get('quantization_level','')}
                        Format: {model.get('format','')}
                        """
                if "format" in model:
                    help_text += f"\n{model['base_name']}:{model['version']}"

                # Use a single checkbox with help tooltip
                remove_checks[model["name"]] = st.checkbox(
                    f"{model['name']}", 
                    key=f"remove_{model['name']}",
                    help=help_text
                )
            # Remove button for selected models
            selected_to_remove = [name for name, checked in remove_checks.items() if checked]
            if selected_to_remove:
                if st.button("Remove Selected Models", key="remove_selected_models"):
                    errors = []
                    for model_name in selected_to_remove:
                        success, message = remove_model(model_name)
                        if not success:
                            errors.append(f"{model_name}: {message}")
                    st.cache_data.clear()
                    if errors:
                        st.session_state.remove_status = {"success": False, "message": "; ".join(errors)}
                    else:
                        st.session_state.remove_status = {"success": True, "message": f"Successfully removed: {', '.join(selected_to_remove)}"}
                    st.rerun()
            # Display removal status if available
            if st.session_state.remove_status:
                if st.session_state.remove_status["success"]:
                    st.success(st.session_state.remove_status["message"])
                else:
                    st.error(st.session_state.remove_status["message"])
        else:
            st.info("No installed models found")
        
        # Download new models section
        st.markdown("#### Download New Models")
        st.info("Remote model list cannot be fetched. Please visit the [Ollama Library](https://ollama.com/library) to browse available models. You can manually enter the model name to download it below.")
        model_to_download = st.text_input(
            "Enter model name to download (e.g., llama3, phi3, etc.)",
            key="manual_model_download"
        )
        col1, col2 = st.columns(2)
        with col1:
            download_option = st.radio(
                "Version",
                options=["latest", "custom"],
                key="download_option"
            )
        version_to_download = "latest"
        if download_option == "custom":
            with col2:
                version_to_download = st.text_input(
                    "Enter version",
                    value="latest",
                    key="version_input"
                )
        full_model_name = f"{model_to_download}:{version_to_download}" if model_to_download else ""
        if model_to_download and st.button(f"Download {full_model_name}", key="download_model_button"):
            success, message = download_model(full_model_name)
            if success:
                st.session_state.download_status = {"success": True, "message": f"Successfully downloaded {full_model_name}"}
                st.cache_data.clear()
                st.rerun()
            else:
                st.session_state.download_status = {"success": False, "message": message}
        if st.session_state.download_status:
            if st.session_state.download_status["success"]:
                st.success(st.session_state.download_status["message"])
            else:
                st.error(st.session_state.download_status["message"])

    # --- Settings Tab ---
    with sidebar_tabs[2]:
        st.subheader("Parameters")
        enable_streaming = st.checkbox("Enable streaming", value=True, 
                                      help="Show responses as they are generated. You'll see the text being generated in real-time.",
                                      key="enable_streaming")
        remove_think_blocks_setting = st.checkbox(
            "Remove think blocks",
            value=False,
            help="Remove any model thought processes from the final response",
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1, key="temperature")
        
        st.subheader("System Prompt (Optional)")
        system_prompt = st.text_area("Enter a system prompt", "", key="system_prompt")
        
        # --- Evaluation Settings ---
        st.subheader("Evaluation")
        base_model_names, models_info = get_available_models()
        model_names = [model["name"] for model in models_info]
        
        if model_names:
            # Default to the first model if available
            default_model = model_names[0] if model_names else ""
            evaluation_model = st.selectbox(
                "Evaluation Model", 
                options=model_names,
                index=0,
                help="Select a model to evaluate the responses",
                key="evaluation_model"
            )
        else:
            st.warning("No models available for evaluation")
            evaluation_model = ""

        evaluation_prompt = st.text_area(
            "Evaluation Prompt", 
            "Please evaluate the following responses from various LLMs and determine which is the most accurate.",
            key="evaluation_prompt"
        )