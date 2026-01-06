import streamlit as st
import pandas as pd
import ollama
from typing import Optional
import html
import re
import requests
import subprocess
import json
import configparser
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any

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

if 'performance_stats' not in st.session_state:
    st.session_state.performance_stats = {}

if 'download_status' not in st.session_state:
    st.session_state.download_status = None

if 'remove_status' not in st.session_state:
    st.session_state.remove_status = None

# --- Performance Stats Data Class ---
@dataclass
class PerformanceStats:
    """Holds performance metrics for a model inference run."""
    model_name: str = ""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    time_to_first_token: float = 0.0  # seconds
    time_to_completion: float = 0.0   # seconds from first token to last
    total_time: float = 0.0           # total inference time
    tokens_per_second: float = 0.0    # completion tokens / time_to_completion
    load_duration_ms: float = 0.0     # model load time (if available)
    eval_duration_ms: float = 0.0     # evaluation duration (if available)
    
    def calculate_tokens_per_second(self):
        """Calculate tokens per second based on completion time."""
        if self.time_to_completion > 0 and self.completion_tokens > 0:
            self.tokens_per_second = self.completion_tokens / self.time_to_completion
        elif self.total_time > 0 and self.completion_tokens > 0:
            self.tokens_per_second = self.completion_tokens / self.total_time
        return self.tokens_per_second
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for display/export."""
        return asdict(self)


def format_performance_stats(stats: PerformanceStats) -> str:
    """Format performance stats for display."""
    lines = []
    if stats.total_time > 0:
        lines.append(f"Total Time: {stats.total_time:.2f}s")
    if stats.time_to_first_token > 0:
        lines.append(f"Time to First Token: {stats.time_to_first_token:.3f}s")
    if stats.time_to_completion > 0:
        lines.append(f"Generation Time: {stats.time_to_completion:.2f}s")
    if stats.completion_tokens > 0:
        lines.append(f"Tokens Generated: {stats.completion_tokens}")
    if stats.tokens_per_second > 0:
        lines.append(f"Speed: {stats.tokens_per_second:.1f} tokens/sec")
    if stats.load_duration_ms > 0:
        lines.append(f"Model Loading Time: {stats.load_duration_ms:.0f}ms")
    if stats.eval_duration_ms > 0:
        lines.append(f"Evaluation Duration: {stats.eval_duration_ms:.0f}ms")
    return " | ".join(lines) if lines else "No stats available"


def create_stats_dataframe(all_stats: Dict[str, PerformanceStats]) -> pd.DataFrame:
    """Create a pandas DataFrame from performance stats for comparison."""
    if not all_stats:
        return pd.DataFrame()
    
    data = []
    for model_name, stats in all_stats.items():
        data.append({
            "Model": model_name,
            "Total Time (seconds)": round(stats.total_time, 2),
            "Time to First Token (seconds)": round(stats.time_to_first_token, 3),
            "Generation Time (seconds)": round(stats.time_to_completion, 2),
            "Tokens Generated": stats.completion_tokens,
            "Tokens per Second": round(stats.tokens_per_second, 1),
            "Model Loading Time (ms)": round(stats.load_duration_ms, 0),
        })
    
    return pd.DataFrame(data)


# Define helper functions first before using them 
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".llm_suite_profiles.ini")

def load_profiles():
    config = configparser.ConfigParser()
    if os.path.exists(CONFIG_PATH):
        config.read(CONFIG_PATH)
    return config

def save_profiles(config):
    with open(CONFIG_PATH, "w") as f:
        config.write(f)

# --- Auto-load default profile on startup - MOVED TO TOP ---
# This needs to be before any widgets are created
if not st.session_state.get("current_session_loaded", False):
    config = load_profiles()
    default_profile = config["DEFAULT"].get("default_profile", "") if "DEFAULT" in config else ""
    
    if default_profile and default_profile in config:
        profile = config[default_profile]
        loaded_models = profile.get("selected_models", "")
        loaded_models = [m for m in loaded_models.split(",") if m]
        
        # Store profile settings in session state
        st.session_state["enable_streaming_value"] = profile.getboolean("enable_streaming", True)
        st.session_state["temperature_value"] = float(profile.get("temperature", 0.7))
        st.session_state["system_prompt_value"] = profile.get("system_prompt", "")
        st.session_state["evaluation_model_value"] = profile.get("evaluation_model", "")
        st.session_state["evaluation_prompt_value"] = profile.get("evaluation_prompt", "")
        st.session_state["remove_think_blocks_value"] = profile.getboolean("remove_think_blocks", False)
        st.session_state["profile_selected_models"] = loaded_models
        
        # Set flag that profile was auto-loaded
        st.session_state["default_profile_autoloaded"] = default_profile
    
    # Mark as loaded to prevent reloading
    st.session_state["current_session_loaded"] = True

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
def query_model(model_name: str, prompt: str, system_prompt: Optional[str] = None, 
                temperature: float = 0.7) -> tuple[str, PerformanceStats]:
    """
    Query a model without streaming and return response with performance stats.
    Returns: (response_text, PerformanceStats)
    """
    stats = PerformanceStats(model_name=model_name)
    start_time = time.time()
    
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
        
        # Calculate timing
        end_time = time.time()
        stats.total_time = end_time - start_time
        
        # Extract token counts and timing from response if available
        stats = extract_ollama_stats(response, stats)
        stats.calculate_tokens_per_second()
        
        return response.response, stats
    except Exception as e:
        error_msg = f"Exception with {model_name}: {str(e)}"
        st.session_state.debug_info.append(error_msg)
        stats.total_time = time.time() - start_time
        return f"Error: Unable to query model. {str(e)}", stats


def extract_ollama_stats(response, stats: PerformanceStats) -> PerformanceStats:
    """Extract performance statistics from Ollama response object."""
    try:
        # Ollama responses include these timing fields (in nanoseconds)
        if hasattr(response, 'total_duration'):
            stats.total_time = response.total_duration / 1e9  # Convert ns to seconds
        if hasattr(response, 'load_duration'):
            stats.load_duration_ms = response.load_duration / 1e6  # Convert ns to ms
        if hasattr(response, 'eval_duration'):
            stats.eval_duration_ms = response.eval_duration / 1e6  # Convert ns to ms
            # Time to completion is essentially eval_duration for non-streaming
            stats.time_to_completion = response.eval_duration / 1e9
        if hasattr(response, 'prompt_eval_count'):
            stats.prompt_tokens = response.prompt_eval_count or 0
        if hasattr(response, 'eval_count'):
            stats.completion_tokens = response.eval_count or 0
            stats.total_tokens = stats.prompt_tokens + stats.completion_tokens
    except Exception as e:
        st.session_state.debug_info.append(f"Error extracting stats: {str(e)}")
    return stats

# streaming inference function
def query_model_streaming(model_name: str, prompt: str, system_prompt: Optional[str] = None, 
                         temperature: float = 0.7, progress_container=None, 
                         streaming_display=None) -> tuple[str, PerformanceStats]:
    """
    Query a model with streaming and return response with performance stats.
    Returns: (response_text, PerformanceStats)
    """
    stats = PerformanceStats(model_name=model_name)
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
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
        last_chunk_data = None
        
        # Stream the response
        for chunk in ollama.generate(**params):
            # Check if inference was stopped
            if st.session_state.stop_inference:
                break
                
            if chunk and 'response' in chunk:
                text_chunk = chunk['response']
                
                # Track first token time
                if first_token_time is None and text_chunk:
                    first_token_time = time.time()
                    stats.time_to_first_token = first_token_time - start_time
                
                full_response += text_chunk
                token_count += 1  # Approximate: count chunks as tokens
                
                # Update the streaming display with current text
                if streaming_display:
                    st.session_state.current_streaming_text = full_response
                    streaming_display.markdown(f"<div class='model-response'>{full_response}</div>", 
                                               unsafe_allow_html=True)
                
                # Update the progress text and length counter
                if progress_container:
                    elapsed = time.time() - start_time
                    current_tps = token_count / elapsed if elapsed > 0 else 0
                    progress_container.markdown(
                        f"**Model:** {model_name} | "
                        f"Length: {len(full_response)} chars | "
                        f"~{current_tps:.1f} tokens/sec"
                    )
            
            # Keep track of the last chunk for final stats
            last_chunk_data = chunk
        
        # Calculate final timing
        end_time = time.time()
        stats.total_time = end_time - start_time
        
        if first_token_time:
            stats.time_to_completion = end_time - first_token_time
        
        # Extract final stats from the last chunk (Ollama sends stats in final chunk)
        if last_chunk_data:
            stats = extract_streaming_final_stats(last_chunk_data, stats)
        
        # If we couldn't get token count from response, use our approximation
        if stats.completion_tokens == 0:
            stats.completion_tokens = token_count
        
        stats.calculate_tokens_per_second()
        
        return full_response, stats
    except Exception as e:
        error_msg = f"Streaming exception with {model_name}: {str(e)}"
        st.session_state.debug_info.append(error_msg)
        stats.total_time = time.time() - start_time
        return f"Error: Unable to stream from model. {str(e)}", stats


def extract_streaming_final_stats(chunk_data: dict, stats: PerformanceStats) -> PerformanceStats:
    """Extract performance statistics from the final streaming chunk."""
    try:
        # The final chunk in streaming contains the full stats
        if 'total_duration' in chunk_data:
            stats.total_time = chunk_data['total_duration'] / 1e9
        if 'load_duration' in chunk_data:
            stats.load_duration_ms = chunk_data['load_duration'] / 1e6
        if 'eval_duration' in chunk_data:
            stats.eval_duration_ms = chunk_data['eval_duration'] / 1e6
        if 'prompt_eval_count' in chunk_data:
            stats.prompt_tokens = chunk_data['prompt_eval_count'] or 0
        if 'eval_count' in chunk_data:
            stats.completion_tokens = chunk_data['eval_count'] or 0
            stats.total_tokens = stats.prompt_tokens + stats.completion_tokens
    except Exception as e:
        st.session_state.debug_info.append(f"Error extracting streaming stats: {str(e)}")
    return stats

# --- Function to evaluate model responses ---
def evaluate_responses(evaluation_model: str, responses: dict, user_prompt: str, evaluation_prompt: str, temperature: float = 0.7) -> str:
    """Evaluate model responses using a specified evaluation model."""
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
        
        # Call the evaluation model (query_model now returns tuple)
        st.session_state.debug_info.append(f"Sending evaluation request to {evaluation_model}...")
        result, _ = query_model(evaluation_model, full_prompt, evaluation_prompt, temperature)
        return result
    except Exception as e:
        error_msg = f"Evaluation exception with {evaluation_model}: {str(e)}"
        st.session_state.debug_info.append(error_msg)
        return f"Error: Unable to perform evaluation. {str(e)}"

def get_installed_model_names():
    _, models_info = get_available_models()
    return set(model["name"] for model in models_info)

def prompt_missing_models(missing_models):
    st.warning(
        f"The following models in the loaded profile are not installed: {', '.join(missing_models)}. "
        "Please download them or remove them from the profile."
    )

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
        
        # Models to pre-select from profile (if any)
        models_to_preselect = st.session_state.get("profile_selected_models", [])
        
        # Fixed key name to avoid duplicates
        select_all = st.checkbox("Select All Models", key="select_all_models", value=False)
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
                        
                        # If model should be pre-selected based on profile
                        preselect = model_name in models_to_preselect
                        
                        if st.checkbox(f"{model_name}", key=f"model_{model_name}", 
                                      help=help_text, value=preselect):
                            selected_models.append(model_name)
                else:
                    with st.expander(f"{base_name} ({len(versions)} versions)", expanded=True):
                        # Check if all versions in this family should be selected
                        family_models_in_profile = [m for m in models_to_preselect if m in versions]
                        all_family_selected = len(family_models_in_profile) == len(versions)
                        
                        select_all_family = st.checkbox(
                            f"Select all {base_name} versions",
                            key=f"select_all_{base_name}",
                            value=all_family_selected
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
                                
                                # Pre-select models from loaded profile
                                preselect = version in models_to_preselect
                                
                                if st.checkbox(f"{version}", key=f"model_{version}", 
                                              help=help_text, value=preselect):
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
        # Use profile values if available for widget defaults
        enable_streaming = st.checkbox(
            "Enable streaming", 
            value=st.session_state.get("enable_streaming_value", True), 
            help="Show responses as they are generated. You'll see the text being generated in real-time.",
            key="enable_streaming"
        )
        remove_think_blocks_setting = st.checkbox(
            "Remove think blocks",
            value=st.session_state.get("remove_think_blocks_value", False),
            help="Remove any model thought processes from the final response",
            key="remove_think_blocks"
        )
        temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=2.0, 
            value=st.session_state.get("temperature_value", 0.7), 
            step=0.1, 
            key="temperature"
        )
        
        st.subheader("System Prompt (Optional)")
        system_prompt = st.text_area(
            "Enter a system prompt", 
            value=st.session_state.get("system_prompt_value", ""), 
            key="system_prompt"
        )
        
        # --- Evaluation Settings ---
        st.subheader("Evaluation")
        base_model_names, models_info = get_available_models()
        model_names = [model["name"] for model in models_info]
        
        if model_names:
            # Default to the first model if available, but use profile value if present
            default_index = 0
            if "evaluation_model_value" in st.session_state:
                try:
                    default_index = model_names.index(st.session_state["evaluation_model_value"])
                except ValueError:
                    default_index = 0
                    
            evaluation_model = st.selectbox(
                "Evaluation Model", 
                options=model_names,
                index=default_index,
                help="Select a model to evaluate the responses",
                key="evaluation_model"
            )
        else:
            st.warning("No models available for evaluation")
            evaluation_model = ""

        evaluation_prompt = st.text_area(
            "Evaluation Prompt", 
            value=st.session_state.get("evaluation_prompt_value", "Several LLMs have been queried with the same prompt. Following are their individual responses to the prompt. Please look over the responses as a whole, and determine which response(s) are the most recurring. DO NOT evaluate the prompt on your own, only find which the most common model response."),
            key="evaluation_prompt"
        )

        # --- Profile/Config Management ---
        st.markdown("### Config / Profile Management")
        config = load_profiles()
        profile_names = [s for s in config.sections() if s != "DEFAULT"]
        default_profile = config["DEFAULT"].get("default_profile", "") if "DEFAULT" in config else ""

        # Select profile to load
        selected_profile = st.selectbox(
            "Select profile to load",
            options=[""] + profile_names,
            index=profile_names.index(default_profile) + 1 if default_profile in profile_names else 0,
            key="profile_select"
        )

        # Profile action buttons (Load, Set Default, Delete) grouped together
        if selected_profile:
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Load Profile", key="load_profile_button"):
                    profile = config[selected_profile]
                    loaded_models = profile.get("selected_models", "")
                    loaded_models = [m for m in loaded_models.split(",") if m]
                    installed_models = get_installed_model_names()
                    missing_models = [m for m in loaded_models if m not in installed_models]
                    if missing_models:
                        prompt_missing_models(missing_models)
                    
                    # Store in session state with _value suffix
                    st.session_state["enable_streaming_value"] = profile.getboolean("enable_streaming", True)
                    st.session_state["temperature_value"] = float(profile.get("temperature", 0.7))
                    st.session_state["system_prompt_value"] = profile.get("system_prompt", "")
                    st.session_state["evaluation_model_value"] = profile.get("evaluation_model", "")
                    st.session_state["evaluation_prompt_value"] = profile.get("evaluation_prompt", "")
                    st.session_state["remove_think_blocks_value"] = profile.getboolean("remove_think_blocks", False)
                    
                    # Just store the model names to select, don't try to set checkbox states directly
                    st.session_state["profile_selected_models"] = loaded_models
                    st.rerun()
            
            with col2:
                if st.button("Set as Default", key="set_default_profile"):
                    if "DEFAULT" not in config:
                        config["DEFAULT"] = {}
                    config["DEFAULT"]["default_profile"] = selected_profile
                    save_profiles(config)
                    st.success(f"Profile '{selected_profile}' set as default.")
            
            with col3:
                if st.button("Delete Profile", key="delete_profile_button"):
                    config.remove_section(selected_profile)
                    save_profiles(config)
                    st.success(f"Profile '{selected_profile}' deleted.")
                    st.rerun()

        # Show current default and auto-load message
        if default_profile:
            st.info(f"Default profile: {default_profile}")
            
        # Save current settings as profile
        st.markdown("#### Create New Profile")
        new_profile_name = st.text_input("Profile name", key="profile_name_input")
        if st.button("Save Profile", key="save_profile_button") and new_profile_name:
            # Get current selected models
            current_selected_models = selected_models[:]  # Make a copy
            
            config[new_profile_name] = {
                "selected_models": ",".join(current_selected_models),
                "enable_streaming": str(st.session_state.get("enable_streaming", True)),
                "temperature": str(st.session_state.get("temperature", 0.7)),
                "system_prompt": st.session_state.get("system_prompt", ""),
                "evaluation_model": st.session_state.get("evaluation_model", ""),
                "evaluation_prompt": st.session_state.get("evaluation_prompt", ""),
                "remove_think_blocks": str(st.session_state.get("remove_think_blocks", False)),
            }
            save_profiles(config)
            st.success(f"Profile '{new_profile_name}' saved.")
            st.rerun()

# --- Main content area ---
st.header("Enter Your Prompt")
user_prompt = st.text_area("The same prompt will be sent to all selected models", "")

# Compare Models button
compare_button = st.button("Compare Models", use_container_width=True)

# Add a stop button that only shows during inference
stop_button_container = st.empty()

# Progress container and bar
progress_container = st.empty()
progress_bar = st.empty()

# Section for live streaming display
streaming_section = st.empty()

# Define the model processing function
def process_models(selected_models):
    """Process all selected models and collect responses with performance stats."""
    # Reset debug info for this run
    st.session_state.debug_info = []
    st.session_state.debug_info.append(f"Starting comparison with {len(selected_models)} models")
    
    # Reset performance stats
    st.session_state.performance_stats = {}
    
    # Set up streaming if enabled
    streaming_display = streaming_section.empty() if enable_streaming else None
    
    # Process each model one at a time
    for i, model in enumerate(selected_models):
        # Check if inference was stopped
        if st.session_state.stop_inference:
            st.session_state.debug_info.append("Inference stopped by user")
            break
            
        try:
            if enable_streaming:
                # Display progress
                progress_container.markdown(f"**Model {i+1}/{len(selected_models)}:** {model} | Starting...")
                progress_bar.progress((i) / len(selected_models))
                
                # Reset streaming display for next model
                st.session_state.current_streaming_text = ""
                
                response, stats = query_model_streaming(
                    model, 
                    user_prompt, 
                    system_prompt, 
                    temperature, 
                    progress_container,
                    streaming_display
                )
            else:
                # Show processing message when not streaming
                progress_container.markdown(f"**Generating {i+1}/{len(selected_models)}:** {model}")
                progress_bar.progress((i) / len(selected_models))
                response, stats = query_model(model, user_prompt, system_prompt, temperature)
            
            st.session_state.results[model] = response
            st.session_state.performance_stats[model] = stats
            progress_bar.progress((i + 1) / len(selected_models))
            
            # Log performance stats
            st.session_state.debug_info.append(
                f"{model}: {stats.completion_tokens} tokens in {stats.total_time:.2f}s "
                f"({stats.tokens_per_second:.1f} tok/s)"
            )
            
        except Exception as e:
            st.session_state.debug_info.append(f"Unhandled exception for {model}: {str(e)}")
            st.session_state.results[model] = f"Unhandled error: {str(e)}"
            # Create empty stats for failed model
            st.session_state.performance_stats[model] = PerformanceStats(model_name=model)
    
    # Clear progress indicators after completion
    progress_bar.empty()
    progress_container.empty()
    if streaming_display:
        streaming_display.empty()
    
    # Show debug information in an expander
    with st.expander("Debug Information"):
        st.code("\n".join(st.session_state.debug_info))
    
    # Clear stop button when done
    stop_button_container.empty()
    
    # Set inference running flag to false
    st.session_state.inference_running = False

def remove_think_blocks(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

if compare_button:
    if not selected_models:
        st.warning("Please select at least one model to compare.")
    else:
        # Clear previous results
        st.session_state.results = {}
        st.session_state.current_streaming_text = ""
        st.session_state.performance_stats = {}
        # Clear previous evaluation result
        st.session_state.evaluation_result = None
        
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
                
        # Process models with or without spinner based on streaming setting
        if not enable_streaming:
            with st.spinner("Generating responses..."):
                process_models(selected_models)
        else:
            process_models(selected_models)

# Display results
if st.session_state.results:
    st.header("Model Responses")
    
    # Clear streaming section once we have results
    streaming_section.empty()
    
    # Show response status summary only if there are errors
    success_count = sum(1 for r in st.session_state.results.values() if not r.startswith("Error"))
    error_count = len(st.session_state.results) - success_count
    
    if error_count > 0:
        st.warning(f"{success_count} successful responses, {error_count} errors")
    
    # --- Performance Stats Section ---
    if st.session_state.performance_stats:
        with st.expander("Performance Statistics", expanded=True):
            stats_df = create_stats_dataframe(st.session_state.performance_stats)
            if not stats_df.empty:
                # Display the stats table
                st.dataframe(
                    stats_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Model": st.column_config.TextColumn("Model", width="medium"),
                        "Total Time (seconds)": st.column_config.NumberColumn("Total Time (seconds)", format="%.2f"),
                        "Time to First Token (seconds)": st.column_config.NumberColumn("Time to First Token (seconds)", format="%.3f"),
                        "Generation Time (seconds)": st.column_config.NumberColumn("Generation Time (seconds)", format="%.2f"),
                        "Tokens Generated": st.column_config.NumberColumn("Tokens Generated", format="%d"),
                        "Tokens per Second": st.column_config.NumberColumn("Tokens per Second", format="%.1f"),
                        "Model Loading Time (ms)": st.column_config.NumberColumn("Model Loading Time (ms)", format="%.0f"),
                    }
                )
                
                # Export stats table as CSV
                st.download_button(
                    label="Export Statistics as CSV",
                    data=stats_df.to_csv(index=False),
                    file_name="performance_statistics.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Show performance comparison chart if multiple models
                if len(st.session_state.performance_stats) > 1:
                    st.markdown("#### Speed Comparison (Tokens per Second)")
                    
                    # Create chart using matplotlib for export capability
                    import matplotlib.pyplot as plt
                    import io
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    models = stats_df["Model"].tolist()
                    speeds = stats_df["Tokens per Second"].tolist()
                    
                    bars = ax.bar(models, speeds, color='#4CAF50')
                    ax.set_xlabel('Model')
                    ax.set_ylabel('Tokens per Second')
                    ax.set_title('Model Speed Comparison')
                    
                    # Rotate x-axis labels if many models
                    if len(models) > 3:
                        plt.xticks(rotation=45, ha='right')
                    
                    # Add value labels on bars
                    for bar, speed in zip(bars, speeds):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               f'{speed:.1f}', ha='center', va='bottom', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Export chart as PNG
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        label="Export Speed Chart as PNG",
                        data=buf,
                        file_name="speed_comparison_chart.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    
                    plt.close(fig)  # Clean up
                    
                    # Additional chart: Time comparison
                    st.markdown("#### Time Comparison")
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    
                    x = range(len(models))
                    width = 0.35
                    
                    total_times = stats_df["Total Time (seconds)"].tolist()
                    first_token_times = stats_df["Time to First Token (seconds)"].tolist()
                    
                    bars1 = ax2.bar([i - width/2 for i in x], total_times, width, label='Total Time', color='#2196F3')
                    bars2 = ax2.bar([i + width/2 for i in x], first_token_times, width, label='Time to First Token', color='#FF9800')
                    
                    ax2.set_xlabel('Model')
                    ax2.set_ylabel('Time (seconds)')
                    ax2.set_title('Model Time Comparison')
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(models)
                    ax2.legend()
                    
                    if len(models) > 3:
                        plt.xticks(rotation=45, ha='right')
                    
                    plt.tight_layout()
                    st.pyplot(fig2)
                    
                    # Export time chart as PNG
                    buf2 = io.BytesIO()
                    fig2.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
                    buf2.seek(0)
                    
                    st.download_button(
                        label="Export Time Chart as PNG",
                        data=buf2,
                        file_name="time_comparison_chart.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    
                    plt.close(fig2)  # Clean up
    
    # Create download buttons and add Evaluate button
    results_df = pd.DataFrame({
        "Model": list(st.session_state.results.keys()),
        "Response": list(st.session_state.results.values()),
        "Length": [len(response) for response in st.session_state.results.values()]
    })
    
    # Only show evaluate button if evaluation result is not present
    if not st.session_state.evaluation_result:
        if st.button("Evaluate Responses", key="evaluate_button", use_container_width=True):
            if evaluation_model:
                with st.spinner(f"Evaluating responses using {evaluation_model}..."):
                    evaluation_result = evaluate_responses(
                        evaluation_model, 
                        st.session_state.results, 
                        user_prompt, 
                        evaluation_prompt, 
                        temperature
                    )
                    st.session_state.evaluation_result = evaluation_result
            else:
                st.error("Please select an evaluation model in the Settings tab")
    
    # Show evaluation results if available
    if st.session_state.evaluation_result:
        st.subheader("Evaluation Results")
        st.markdown(f"<div class='model-response'>{html.escape(st.session_state.evaluation_result)}</div>", 
                  unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download as CSV",
            data=results_df.to_csv(index=False),
            file_name="ollama_model_comparison.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    json_results = json.dumps({
        "prompt": user_prompt,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "results": st.session_state.results,
        "response_lengths": {model: len(response) for model, response in st.session_state.results.items()},
        "performance_stats": {
            model: stats.to_dict() 
            for model, stats in st.session_state.performance_stats.items()
        } if st.session_state.performance_stats else {},
        "evaluation": st.session_state.evaluation_result
    }, indent=2)
    
    with col2:
        st.download_button(
            label="Download as JSON",
            data=json_results,
            file_name="ollama_model_comparison.json",
            mime="application/json",
            use_container_width=True
        )

    # Tabs for different view modes
    tab1, tab2 = st.tabs(["Side by Side", "Stacked"])
    
    with tab1:
        # Side by side view
        models_with_results = [m for m in selected_models if m in st.session_state.results]
        
        if not models_with_results:
            models_with_results = list(st.session_state.results.keys())
        
        if len(models_with_results) == 1:
            model_name = models_with_results[0]
            response = st.session_state.results[model_name]
            # --- Remove <think>...</think> if setting is enabled ---
            if remove_think_blocks_setting:
                response = remove_think_blocks(response)
            st.subheader(f"{model_name} ({len(response)} chars)")
            # Show inline stats
            if model_name in st.session_state.performance_stats:
                stats = st.session_state.performance_stats[model_name]
                st.caption(format_performance_stats(stats))
            st.markdown(f"<div class='model-response'>{html.escape(response)}</div>", 
                       unsafe_allow_html=True)
        else:
            for i in range(0, len(models_with_results), 2):
                row_cols = st.columns(2)
                with row_cols[0]:
                    model_name = models_with_results[i]
                    response = st.session_state.results[model_name]
                    if remove_think_blocks_setting:
                        response = remove_think_blocks(response)
                    st.subheader(f"{model_name} ({len(response)} chars)")
                    # Show inline stats
                    if model_name in st.session_state.performance_stats:
                        stats = st.session_state.performance_stats[model_name]
                        st.caption(format_performance_stats(stats))
                    st.markdown(f"<div class='model-response'>{html.escape(response)}</div>",
                               unsafe_allow_html=True)
                if i + 1 < len(models_with_results):
                    with row_cols[1]:
                        model_name = models_with_results[i + 1]
                        response = st.session_state.results[model_name]
                        if remove_think_blocks_setting:
                            response = remove_think_blocks(response)
                        st.subheader(f"{model_name} ({len(response)} chars)")
                        # Show inline stats
                        if model_name in st.session_state.performance_stats:
                            stats = st.session_state.performance_stats[model_name]
                            st.caption(format_performance_stats(stats))
                        st.markdown(f"<div class='model-response'>{html.escape(response)}</div>", 
                                   unsafe_allow_html=True)
    
    with tab2:
        # Stacked view
        for model, response in st.session_state.results.items():
            if remove_think_blocks_setting:
                response = remove_think_blocks(response)
            # Include stats in expander title
            stats_summary = ""
            if model in st.session_state.performance_stats:
                stats = st.session_state.performance_stats[model]
                if stats.tokens_per_second > 0:
                    stats_summary = f" | {stats.tokens_per_second:.1f} tok/s"
            with st.expander(f"{model} ({len(response)} chars{stats_summary})", expanded=True):
                # Show detailed stats inside expander
                if model in st.session_state.performance_stats:
                    st.caption(format_performance_stats(st.session_state.performance_stats[model]))
                st.markdown(f"<div class='model-response'>{html.escape(response)}</div>", unsafe_allow_html=True)