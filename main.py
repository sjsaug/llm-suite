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
import csv
import copy
import tempfile
import concurrent.futures
from fpdf import FPDF
try:
    from fpdf.enums import XPos, YPos
except ImportError:
    # Fallback for older fpdf2 versions or if using fpdf 1.7.x
    XPos, YPos = None, None
from langchain_core.prompts import ChatPromptTemplate
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any

def sanitize_key(key: str) -> str:
    """Sanitize a string to be used as a Streamlit widget key.
    
    Replaces characters that are invalid in HTML element IDs (like colons)
    with underscores to prevent InvalidCharacterError in the browser.
    """
    # Replace colons and other problematic characters with underscores
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', key)

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

# Batch testset artifacts
if 'testset_batch_reports' not in st.session_state:
    st.session_state.testset_batch_reports = []

# Batch debug logs for testset runs
if 'testset_batch_debug_logs' not in st.session_state:
    st.session_state.testset_batch_debug_logs = []

# Persist uploaded testsets
if 'prompt_testsets' not in st.session_state:
    st.session_state.prompt_testsets = []

# Parallel prompt comparison results
if 'parallel_prompt_results' not in st.session_state:
    st.session_state.parallel_prompt_results = {}

if 'parallel_prompt_stats' not in st.session_state:
    st.session_state.parallel_prompt_stats = {}

# Enhanced parallel comparison (multi-model, testset support)
if 'parallel_results_all' not in st.session_state:
    st.session_state.parallel_results_all = {}

if 'parallel_stats_all' not in st.session_state:
    st.session_state.parallel_stats_all = {}

if 'parallel_testset_results' not in st.session_state:
    st.session_state.parallel_testset_results = []

if 'parallel_validation_results' not in st.session_state:
    st.session_state.parallel_validation_results = {}

# Store PDF reports for parallel prompts (each prompt gets its own report)
if 'parallel_prompt_reports' not in st.session_state:
    st.session_state.parallel_prompt_reports = {}  # Structure: {prompt_label: {'pdf': bytes, 'csv_df': df, 'agg_df': df}}

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
                temperature: float = 0.7, keep_alive: str = "10m") -> tuple[str, PerformanceStats]:
    """
    Query a model without streaming and return response with performance stats.
    
    Args:
        keep_alive: How long to keep model in memory (e.g., "10m", "1h", "-1" for indefinite)
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
            "options": {"temperature": temperature},
            "keep_alive": keep_alive  # Prevent model unloading during batch operations
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
                         streaming_display=None, keep_alive: str = "10m") -> tuple[str, PerformanceStats]:
    """
    Query a model with streaming and return response with performance stats.
    
    Args:
        keep_alive: How long to keep model in memory (e.g., "10m", "1h", "-1" for indefinite)
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
            "options": {"temperature": temperature},
            "keep_alive": keep_alive  # Prevent model unloading during batch operations
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
                    streaming_display.markdown(f"<div class='model-response'>{html.escape(full_response)}</div>", 
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


def run_parallel_prompts(model_name: str, prompts: list, system_prompt_text: Optional[str] = None, 
                         temp: float = 0.7, keep_alive: str = "10m") -> tuple:
    """
    Run the same model with multiple prompts in parallel using ThreadPoolExecutor.
    
    Args:
        model_name: The name of the model to use
        prompts: List of dicts with 'label' and 'prompt' keys
        system_prompt_text: Optional system prompt
        temp: Temperature setting
        keep_alive: How long to keep model in memory
    
    Returns:
        Tuple of (results dict, stats dict, debug_logs list)
    """
    results = {}
    stats = {}
    debug_logs = []  # Collect debug info locally to avoid thread-unsafe session_state access
    
    def execute_prompt(prompt_data):
        """Execute a single prompt and return the result with stats."""
        label = prompt_data['label']
        prompt_text = prompt_data['prompt']
        try:
            # Use a local query function that doesn't access session_state
            perf_stats = PerformanceStats(model_name=model_name)
            start_time = time.time()
            
            params = {
                "model": model_name,
                "prompt": prompt_text,
                "stream": False,
                "options": {"temperature": temp},
                "keep_alive": keep_alive
            }
            
            if system_prompt_text and system_prompt_text.strip():
                params["system"] = system_prompt_text
            
            response = ollama.generate(**params)
            
            end_time = time.time()
            perf_stats.total_time = end_time - start_time
            
            # Extract stats from response
            try:
                if hasattr(response, 'total_duration'):
                    perf_stats.total_time = response.total_duration / 1e9
                if hasattr(response, 'load_duration'):
                    perf_stats.load_duration_ms = response.load_duration / 1e6
                if hasattr(response, 'eval_duration'):
                    perf_stats.eval_duration_ms = response.eval_duration / 1e6
                    perf_stats.time_to_completion = response.eval_duration / 1e9
                if hasattr(response, 'prompt_eval_count'):
                    perf_stats.prompt_tokens = response.prompt_eval_count or 0
                if hasattr(response, 'eval_count'):
                    perf_stats.completion_tokens = response.eval_count or 0
                    perf_stats.total_tokens = perf_stats.prompt_tokens + perf_stats.completion_tokens
            except Exception:
                pass
            
            perf_stats.calculate_tokens_per_second()
            
            return label, response.response, perf_stats, None
        except Exception as e:
            error_stats = PerformanceStats(model_name=model_name)
            return label, f"Error: {str(e)}", error_stats, str(e)
    
    # Execute all prompts in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        # Submit all tasks
        futures = {
            executor.submit(execute_prompt, prompt_data): prompt_data['label'] 
            for prompt_data in prompts
        }
        
        # Wait for all to complete
        concurrent.futures.wait(futures.keys())
        
        # Collect results
        for future in futures:
            label, response, perf_stats, error = future.result()
            results[label] = response
            stats[label] = perf_stats
            if error:
                debug_logs.append(f"Parallel prompt '{label}' error: {error}")
            else:
                debug_logs.append(
                    f"Parallel prompt '{label}': {perf_stats.completion_tokens} tokens in "
                    f"{perf_stats.total_time:.2f}s ({perf_stats.tokens_per_second:.1f} tok/s)"
                )
    
    return results, stats, debug_logs


def evaluate_response_accuracy(judge_model: str, user_prompt: str, model_response: str, expected_responses: Optional[str] = None) -> bool:
    """
    Uses a judge model to determine if the response is accurate by comparing
    the model's response against the expected responses.
    
    Args:
        judge_model: The model to use as judge
        user_prompt: The original prompt (for context only)
        model_response: The LLM's actual response to evaluate
        expected_responses: Pipe-separated expected responses (e.g., "resp1 | resp2 | resp3")
    
    Returns True if accurate, False otherwise.
    """
    if not expected_responses:
        return False
    
    # Use Ollama structured output for reliable parsing
    try:
        # Build a balanced judge prompt focused on semantic equivalence
        judge_prompt = f"""You are an accuracy judge. Determine if MODEL_RESPONSE is semantically equivalent to ANY of the EXPECTED_RESPONSES.

=== EXPECTED_RESPONSES ===
{expected_responses}

=== MODEL_RESPONSE ===
{model_response}

=== JUDGMENT CRITERIA ===

Mark as "accurate" if:
1. The MODEL_RESPONSE conveys the SAME CORE MEANING as any expected response
2. The MODEL_RESPONSE asks the same question or makes the same statement, even if worded differently
3. The MODEL_RESPONSE is a more general or more specific version of an expected response, as long as the core topic is the same
4. Both MODEL_RESPONSE and an expected response use similar reference styles (e.g., both use "it" or both use explicit terms)

Mark as "not_accurate" if:
1. The MODEL_RESPONSE uses VAGUE REFERENCES (like "it", "they", "those", "this") when the expected responses use EXPLICIT, SPECIFIC terms - this indicates the model failed to resolve the reference
2. The MODEL_RESPONSE asks about or states something fundamentally DIFFERENT from all expected responses
3. The MODEL_RESPONSE is about a completely different topic or entity

=== KEY DISTINCTION ===
The critical test for vague references: Look at whether the EXPECTED_RESPONSES themselves use vague terms.
- If expected responses say "How can we source it?" → a model response with "it" is ACCURATE
- If expected responses say "How can we source the main material?" → a model response with "it" is NOT ACCURATE

=== EXAMPLES ===

# I've removved them for my use case. Feel free to add your own.

=== YOUR TASK ===
Determine if MODEL_RESPONSE is semantically equivalent to ANY expected response, considering the rules above."""

        # Use Ollama structured output to get consistent results
        structured_response = ollama.generate(
            model=judge_model,
            prompt=judge_prompt,
            format={
                "type": "object",
                "properties": {
                    "judgment": {
                        "type": "string",
                        "enum": ["accurate", "not_accurate"],
                        "description": "Whether the model response is semantically equivalent to any expected response"
                    }
                },
                "required": ["judgment"]
            },
            options={'temperature': 0.0},  # Zero temperature for deterministic judgment
            keep_alive="10m"  # Keep judge model loaded during batch evaluation
        )
        
        # Parse the structured response
        result = json.loads(structured_response['response'])
        judgment = result.get("judgment", "").lower()
        
        # Log the judgment for debugging
        if 'debug_info' in st.session_state:
            st.session_state.debug_info.append(
                f"Judge ({judge_model}): {judgment} | Response: '{model_response[:80]}...'"
            )
        
        return judgment == "accurate"
        
    except Exception as e:
        # Log error in debug info
        if 'debug_info' in st.session_state:
            st.session_state.debug_info.append(f"Accuracy judge error: {str(e)}")
        
        # Fallback: try non-structured approach if structured output fails
        try:
            fallback_prompt = f"""Compare MODEL_RESPONSE against EXPECTED_RESPONSES for semantic equivalence.

EXPECTED_RESPONSES: {expected_responses}

MODEL_RESPONSE: {model_response}

RULE: Mark accurate if the core meaning is the same. Only mark not_accurate if the model uses vague references (it/they/those) when expected responses use explicit terms, OR if the topic is completely different.

Answer ONLY with "accurate" or "not_accurate":"""
            
            response, _ = query_model(judge_model, fallback_prompt, None, temperature=0.0)
            response_lower = response.lower().strip()
            
            if "not_accurate" in response_lower or "not accurate" in response_lower:
                return False
            if "accurate" in response_lower:
                return True
            return False
        except:
            return False



def parse_testset_file(uploaded_file, input_vars: List[str]) -> List[Dict[str, str]]:
    """Parse a testset file into a list of variable dicts.
    Accepts CSV, pipe-delimited (|||), or simple comma-separated lines. 
    If the file has a header matching the input variable names, that will be used. 
    Otherwise values are mapped by position to `input_vars`.
    """
    if not uploaded_file:
        return []

    # Read bytes and decode
    try:
        content = uploaded_file.getvalue().decode('utf-8')
    except Exception:
        try:
            content = uploaded_file.getvalue().decode('latin-1')
        except Exception:
            content = str(uploaded_file.getvalue())

    rows = []
    
    # Detect delimiter: check if ||| is used (preferred for avoiding comma conflicts)
    lines = [line for line in content.splitlines() if line.strip()]
    if not lines:
        return []
    
    # Check first non-empty line for delimiter type
    use_pipe_delimiter = "|||" in lines[0]
    
    if use_pipe_delimiter:
        # Parse using ||| delimiter
        all_lines = []
        for line in lines:
            parts = [p.strip() for p in line.split("|||")]
            all_lines.append(parts)
    else:
        # Fall back to CSV parsing for backward compatibility
        reader = csv.reader(lines)
        all_lines = list(reader)
    
    if not all_lines:
        return []

    # If first row contains headers that match input_vars, use header mapping
    first = [c.strip() for c in all_lines[0]]
    mapped_start = 0
    if set([v for v in first]) >= set(input_vars) or set(input_vars) <= set(first):
        # treat first as header
        headers = first
        mapped_start = 1
        for line in all_lines[1:]:
            if not any(s.strip() for s in line):
                continue
            vals = [v.strip() for v in line]
            row = {}
            temp_expected = []
            
            # Map known variables
            for h, v in zip(headers, vals):
                if h in input_vars:
                    row[h] = v
                else: 
                     # Collect any columns not in input_vars as potential expected responses
                     # (Assuming simple CSV structure where extra cols are targets)
                     if v.strip():
                         temp_expected.append(v.strip())

            # Fill missing input_vars with empty strings
            for iv in input_vars:
                if iv not in row:
                    row[iv] = ""
            
            if temp_expected:
                row['_expected_responses'] = temp_expected
                
            rows.append(row)
    else:
        # No header: map by position to input_vars
        for line in all_lines:
            if not any(s.strip() for s in line):
                continue
            vals = [v.strip() for v in line]
            if len(vals) < len(input_vars):
                # pad with empty strings
                vals += [""] * (len(input_vars) - len(vals))
            row = {var: vals[idx] for idx, var in enumerate(input_vars)}
            
            # Capture any remaining columns as expected responses
            if len(vals) > len(input_vars):
                expected = [x.strip() for x in vals[len(input_vars):] if x.strip()]
                if expected:
                    row['_expected_responses'] = expected
            
            rows.append(row)

    return rows


def aggregate_testset_results_to_dataframe(testset_results: List[Dict]) -> pd.DataFrame:
    """Flatten testset results into a table suitable for CSV export.
    Each row corresponds to a single (iteration, model) pair.
    """
    records = []
    for idx, item in enumerate(testset_results, start=1):
        vars_map = item.get('vars', {})
        results = item.get('results', {})
        perf = item.get('performance_stats', {})
        validation = item.get('validation', {})

        # Clean vars map for display (remove internal keys)
        display_vars = {k: v for k, v in vars_map.items() if not k.startswith('_')}
        expected_responses = vars_map.get('_expected_responses', [])

        for model, resp in results.items():
            stats = perf.get(model)
            rec = {
                'iteration': idx,
                **{f'var_{k}': v for k, v in display_vars.items()},
                'model': model,
                'response': resp,
                'length': len(resp) if resp else 0,
            }

            if expected_responses:
                rec['expected_response'] = " | ".join(expected_responses)

            if validation:
                is_acc = validation.get(model, False)
                rec['is_accurate'] = is_acc
                rec['accuracy_label'] = "Accurate" if is_acc else "Not Accurate"

            if stats:
                rec.update({
                    'completion_tokens': stats.get('completion_tokens', ''),
                    'total_time': stats.get('total_time', ''),
                })
            records.append(rec)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)

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

def collect_model_ratings():
    """Collect ratings from session state."""
    model_ratings = {}
    if 'results' in st.session_state:
        for model in st.session_state.results.keys():
            side_key = f"rating_side_{model}"
            stacked_key = f"rating_stacked_{model}"
            
            side_rating = st.session_state.get(side_key, "Select...")
            stacked_rating = st.session_state.get(stacked_key, "Select...")
            
            final_rating = "N/A"
            if side_rating != "Select...":
                final_rating = side_rating
            elif stacked_rating != "Select...":
                final_rating = stacked_rating
                
            model_ratings[model] = final_rating
    return model_ratings


# --- Chart Generation Helpers ---
import matplotlib.pyplot as plt

def create_speed_chart(stats_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    models = stats_df["Model"].tolist()
    speeds = stats_df["Tokens per Second"].tolist()
    
    bars = ax.bar(models, speeds, color='#4CAF50')
    ax.set_xlabel('Model')
    ax.set_ylabel('Tokens per Second')
    ax.set_title('Model Speed Comparison')
    
    if len(models) > 3:
        plt.xticks(rotation=45, ha='right')
    
    for bar, speed in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{speed:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_time_chart(stats_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    models = stats_df["Model"].tolist()
    
    x = range(len(models))
    width = 0.35
    
    total_times = stats_df["Total Time (seconds)"].tolist()
    first_token_times = stats_df["Time to First Token (seconds)"].tolist()
    
    bars1 = ax.bar([i - width/2 for i in x], total_times, width, label='Total Time', color='#2196F3')
    bars2 = ax.bar([i + width/2 for i in x], first_token_times, width, label='Time to First Token', color='#FF9800')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Model Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    if len(models) > 3:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def create_accuracy_chart(model_ratings):
    rating_values = {"Accurate": 3, "Somewhat Accurate": 2, "Not Accurate": 1}
    rating_colors = {"Accurate": "#4CAF50", "Somewhat Accurate": "#FFC107", "Not Accurate": "#F44336"}
    
    chart_data = []
    colors = []
    
    for model in sorted(model_ratings.keys()):
        rating = model_ratings.get(model, "N/A")
        if rating in rating_values:
            chart_data.append({"Model": model, "Score": rating_values[rating], "Rating": rating})
            colors.append(rating_colors[rating])
    
    if not chart_data:
        return None
        
    df_chart = pd.DataFrame(chart_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df_chart["Model"], df_chart["Score"], color=colors)
    
    ax.set_title("Model Accuracy Ratings")
    ax.set_ylabel("Accuracy Level")
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Not Accurate", "Somewhat", "Accurate"])
    ax.set_ylim(0, 3.5)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    if len(df_chart) > 3:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def create_stacked_time_chart(stats_df):
    """Create a stacked bar chart for Time to First Token and Generation Time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    models = stats_df["Model"].tolist()
    
    ttft = stats_df["Time to First Token (seconds)"].tolist()
    # Generation time for the purpose of stacking is Total Time - TTFT
    # Ensure no negative values if TTFT > Total Time (shouldn't happen but safety first)
    gen_time = [max(0, total - first) for total, first in zip(stats_df["Total Time (seconds)"], ttft)]
    
    # Plot TTFT at bottom
    p1 = ax.bar(models, ttft, label='Time to First Token', color='#FF9800')
    # Plot Generation Time on top of TTFT
    p2 = ax.bar(models, gen_time, bottom=ttft, label='Total Time (stacked)', color='#2196F3')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Average Latency Distribution')
    ax.legend()
    
    if len(models) > 3:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def create_testset_accuracy_chart(agg_df):
    """Create accuracy chart based on percentage of accurate responses."""
    if 'Accuracy' not in agg_df.columns:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 6))
    models = agg_df["Model"].tolist()
    # Convert 0-1 range to 0-100
    scores = [x * 100 for x in agg_df["Accuracy"].tolist()]
    
    bars = ax.bar(models, scores, color='#4CAF50')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy (Testset)')
    ax.set_ylim(0, 100) # Percentage
    
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, score + 1,
               f'{score:.1f}%', ha='center', va='bottom', fontsize=9)
               
    if len(models) > 3:
        plt.xticks(rotation=45, ha='right')
        
    plt.tight_layout()
    return fig

class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        # Check if XPos and YPos are available (handled in import)
        try:
            if 'XPos' in globals() and XPos:
                self.cell(0, 10, 'Comparison Report', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
            else:
                self.cell(0, 10, 'Comparison Report', 0, 1, 'C')
        except Exception:
             self.cell(0, 10, 'Comparison Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        try:
            if 'XPos' in globals() and XPos:
                self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')
            else:
                self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        except:
             self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

def generate_pdf_report(user_prompt, system_prompt, results, stats, ratings, evaluation, prompt_vars=None, chart_paths=None):
    pdf = PDFReport()
    pdf.alias_nb_pages()

    # Try to register a Unicode TrueType font (DejaVuSans) if available so we can
    # include UTF-8 characters in the PDF. If not available, we'll fall back to
    # Latin-1 with replacement for unsupported characters.
    unicode_font_registered = False
    # Common TTF font paths across Linux, macOS and Windows
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/Library/Fonts/DejaVuSans.ttf",
        "/Library/Fonts/LiberationSans-Regular.ttf",
        "C:\\Windows\\Fonts\\DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\LiberationSans-Regular.ttf",
        "C:\\Windows\\Fonts\\Arial.ttf",
    ]
    for fp in font_paths:
        try:
            if os.path.exists(fp):
                pdf.add_page()  # add page after alias to avoid font issues
                pdf.add_font('DejaVu', '', fp, uni=True)
                unicode_font_registered = True
                break
        except Exception:
            unicode_font_registered = False

    # If a unicode font wasn't registered, ensure we still have a page
    if not unicode_font_registered:
        pdf.add_page()

    # Helper to sanitize text based on font availability
    def _safe_text(txt: str) -> str:
        if txt is None:
            return ""
        
        s = str(txt)
        
        # Always break extremely long words/sequences to prevent FPDF 
        # "Not enough horizontal space" errors
        def _break_long_words(text: str, maxlen: int = 30) -> str:
            return re.sub(r"(\S{%d,})" % maxlen,
                            lambda m: ' '.join([m.group(0)[i:i+maxlen] for i in range(0, len(m.group(0)), maxlen)]),
                            text)
        
        s = _break_long_words(s, maxlen=30)

        if unicode_font_registered:
            return s
            
        try:
            # If using non-unicode core fonts, ensure very long unbroken
            # sequences are split so FPDF can wrap them on all platforms
            return s.encode('latin-1', 'replace').decode('latin-1')
        except Exception:
            return s
    
    # Prompt Section
    if unicode_font_registered:
        pdf.set_font('DejaVu', '', 12)
    else:
        pdf.set_font("Helvetica", 'B', 12)
    
    # Use explicit new_x/new_y if available in recent fpdf2, otherwise fallback
    try:
        if XPos and YPos:
            pdf.cell(0, 10, "User Prompt:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.cell(0, 10, "User Prompt:", 0, 1)
    except Exception:
        pdf.cell(0, 10, "User Prompt:", 0, 1)

    if unicode_font_registered:
        pdf.set_font('DejaVu', '', 11)
    else:
        pdf.set_font("Helvetica", size=11)
    
    # Set X to left margin before printing long text blocks
    pdf.set_x(pdf.l_margin)
    try:
        pdf.multi_cell(0, 6, _safe_text(user_prompt))
    except Exception:
         pdf.cell(0, 6, "User prompt too long to display.", 0, 1)
    pdf.ln(5)

    # Prompt Variables Section
    if prompt_vars:
        if unicode_font_registered:
            pdf.set_font('DejaVu', '', 12)
        else:
            pdf.set_font("Helvetica", 'B', 12)
        try:
            if XPos and YPos:
                 pdf.cell(0, 10, "Prompt Variables:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                 pdf.cell(0, 10, "Prompt Variables:", 0, 1)
        except Exception:
            pdf.cell(0, 10, "Prompt Variables:", 0, 1)

        if unicode_font_registered:
            pdf.set_font('DejaVu', '', 11)
        else:
            pdf.set_font("Helvetica", size=11)
        
        pdf.set_x(pdf.l_margin)
        for var, val in prompt_vars.items():
            try:
                pdf.multi_cell(0, 6, _safe_text(f"{var}: {val}"))
            except Exception:
                pdf.cell(0, 6, _safe_text(f"{var}: [Content error]"), 0, 1)
        pdf.ln(5)

    # Charts Section (New)
    if chart_paths:
        # Check if we have space, else add page
        if pdf.get_y() > 200:
            pdf.add_page()
            
        pdf.set_font("Helvetica", 'B', 12)
        try:
            if XPos and YPos:
                pdf.cell(0, 10, "Performance & Accuracy Charts:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                pdf.cell(0, 10, "Performance & Accuracy Charts:", 0, 1)
        except:
             pdf.cell(0, 10, "Performance & Accuracy Charts:", 0, 1)

        pdf.ln(2)
        
        for title, path in chart_paths.items():
            if pdf.get_y() > 200:
                pdf.add_page()
            if unicode_font_registered:
                pdf.set_font('DejaVu', '', 11)
            else:
                pdf.set_font("Helvetica", 'B', 11)
            
            try:
                if XPos and YPos:
                     pdf.cell(0, 10, _safe_text(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                else:
                     pdf.cell(0, 10, _safe_text(title), 0, 1)
            except:
                 pdf.cell(0, 10, _safe_text(title), 0, 1)

            try:
                # Adjust image width to fit page mostly
                # A4 width is 210mm. Margins approx 10mm each side -> 190mm width
                pdf.image(path, x=10, w=190)
                pdf.ln(5)
            except Exception as e:
                if unicode_font_registered:
                    pdf.set_font('DejaVu', '', 10)
                else:
                    pdf.set_font("Helvetica", 'I', 10)
                pdf.cell(0, 10, _safe_text(f"Error adding chart: {str(e)}"), 0, 1)
            pdf.ln(5)
        pdf.ln(5)
    
    if system_prompt:
        if unicode_font_registered:
            pdf.set_font('DejaVu', '', 12)
        else:
            pdf.set_font("Helvetica", 'B', 12)
            
        try:
            if XPos and YPos:
                pdf.cell(0, 10, "System Prompt:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                pdf.cell(0, 10, "System Prompt:", 0, 1)
        except:
            pdf.cell(0, 10, "System Prompt:", 0, 1)

        if unicode_font_registered:
            pdf.set_font('DejaVu', '', 11)
        else:
            pdf.set_font("Helvetica", size=11)
            
        pdf.set_x(pdf.l_margin)
        try:
            pdf.multi_cell(0, 6, _safe_text(system_prompt))
        except:
             pdf.cell(0, 6, "System prompt too long to display.", 0, 1)
        pdf.ln(5)
        
    # Models Section
    for model, response in results.items():
        # Check if we have enough space for the header, otherwise add page
        if pdf.get_y() > 230:
            pdf.add_page()
        else:
            pdf.ln(10)

        if unicode_font_registered:
            pdf.set_font('DejaVu', '', 14)
        else:
            pdf.set_font("Helvetica", 'B', 14)
            
        try:
            if XPos and YPos:
                pdf.cell(0, 10, _safe_text(f"Model: {model}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                pdf.cell(0, 10, _safe_text(f"Model: {model}"), 0, 1)
        except:
             pdf.cell(0, 10, _safe_text(f"Model: {model}"), 0, 1)

        # Stats & Accuracy
        if unicode_font_registered:
            pdf.set_font('DejaVu', '', 10)
        else:
            pdf.set_font("Helvetica", 'B', 10)
        
        try:
            if 'XPos' in globals() and XPos:
                pdf.cell(0, 6, _safe_text("Performance & Accuracy:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                pdf.cell(0, 6, _safe_text("Performance & Accuracy:"), 0, 1)
        except:
             pdf.cell(0, 6, _safe_text("Performance & Accuracy:"), 0, 1)

        if unicode_font_registered:
            pdf.set_font('DejaVu', '', 10)
        else:
            pdf.set_font("Helvetica", size=10)

        stats_text = "No stats available"
        if model in stats:
            s = stats[model]
            stats_text = f"Time: {s.total_time:.2f}s | Speed: {s.tokens_per_second:.1f} t/s | TTFT: {s.time_to_first_token:.3f}s | Tokens: {s.completion_tokens}"

        rating = ratings.get(model, "N/A")
        
        try:
            if 'XPos' in globals() and XPos:
                pdf.cell(0, 6, _safe_text(f"{stats_text} | Accuracy: {rating}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                 pdf.cell(0, 6, _safe_text(f"{stats_text} | Accuracy: {rating}"), 0, 1)
        except:
             pdf.cell(0, 6, _safe_text(f"{stats_text} | Accuracy: {rating}"), 0, 1)

        pdf.ln(5)

        # Response
        if unicode_font_registered:
            pdf.set_font('DejaVu', '', 12)
        else:
            pdf.set_font("Helvetica", 'B', 12)
            
        try:
            if 'XPos' in globals() and XPos:
                 pdf.cell(0, 10, _safe_text("Response:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                 pdf.cell(0, 10, _safe_text("Response:"), 0, 1)
        except:
             pdf.cell(0, 10, _safe_text("Response:"), 0, 1)

        if unicode_font_registered:
            pdf.set_font('DejaVu', '', 10)
        else:
            pdf.set_font("Helvetica", size=10)

        # Ensure we are starting at the left margin to maximize available width
        pdf.set_x(pdf.l_margin)
        
        try:
            pdf.multi_cell(0, 5, _safe_text(response))
        except Exception as e:
            # Fallback for spacing errors: simplify text aggressively
            try:
                # Try printing just a safe snippet
                safe_snippet = _safe_text(response[:500]) if response else ""
                if 'XPos' in globals() and XPos:
                     pdf.multi_cell(0, 5, f"Preview (rendering error): {safe_snippet}...", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                else:
                     pdf.multi_cell(0, 5, f"Preview (rendering error): {safe_snippet}...")
            except:
                pass 
        
        pdf.ln(5)

    # Evaluation Section
    if evaluation:
        pdf.add_page()
        if unicode_font_registered:
            pdf.set_font('DejaVu', '', 14)
        else:
            pdf.set_font("Helvetica", 'B', 14)
        try:
            if 'XPos' in globals() and XPos:
                 pdf.cell(0, 10, _safe_text("Evaluation Result"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                 pdf.cell(0, 10, _safe_text("Evaluation Result"), 0, 1)
        except:
             pdf.cell(0, 10, _safe_text("Evaluation Result"), 0, 1)

        if unicode_font_registered:
            pdf.set_font('DejaVu', '', 10)
        else:
            pdf.set_font("Helvetica", size=10)
            
        try:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, _safe_text(evaluation))
        except Exception:
             try:
                 if 'XPos' in globals() and XPos:
                     pdf.cell(0, 5, "[Evaluation text could not be rendered due to spacing limits]", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                 else:
                     pdf.cell(0, 5, "[Evaluation text could not be rendered due to spacing limits]", 0, 1)
             except:
                 pass

    # Return bytes for the PDF.
    # fpdf2 `output()` may return bytes, bytearray, or string depending on version/args.
    pdf_output = pdf.output(dest='S')
    
    if isinstance(pdf_output, (bytes, bytearray)):
        return bytes(pdf_output)
        
    # If it's a string (older FPDF), encode it.
    try:
        return pdf_output.encode('latin-1')
    except Exception:
        return pdf_output.encode('latin-1', 'replace')

def generate_testset_pdf_report(chart_paths, rankings=None):
    pdf = PDFReport()
    pdf.alias_nb_pages()

    # Try to register a Unicode TrueType font (DejaVuSans)
    unicode_font_registered = False
    # Common TTF font paths across Linux, macOS and Windows
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/Library/Fonts/DejaVuSans.ttf",
        "/Library/Fonts/LiberationSans-Regular.ttf",
        "C:\\Windows\\Fonts\\DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\LiberationSans-Regular.ttf",
        "C:\\Windows\\Fonts\\Arial.ttf",
    ]
    for fp in font_paths:
        try:
            if os.path.exists(fp):
                pdf.add_page()  # add page after alias to avoid font issues
                pdf.add_font('DejaVu', '', fp, uni=True)
                unicode_font_registered = True
                break
        except Exception:
            unicode_font_registered = False

    if not unicode_font_registered:
        pdf.add_page()

    # Helper to sanitize text based on font availability
    def _safe_text(txt: str) -> str:
        if txt is None:
            return ""
        s = str(txt)
        if unicode_font_registered:
            return s
        try:
            return s.encode('latin-1', 'replace').decode('latin-1')
        except Exception:
            return s
            
    # Title
    if unicode_font_registered:
        pdf.set_font('DejaVu', '', 16)
    else:
        pdf.set_font('Helvetica', 'B', 16)
        
    try:
        if 'XPos' in globals() and XPos:
            pdf.cell(0, 10, 'Testset Performance Report', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        else:
            pdf.cell(0, 10, 'Testset Performance Report', 0, 1, 'C')
    except:
         pdf.cell(0, 10, 'Testset Performance Report', 0, 1, 'C')
    pdf.ln(10)

    # Charts and Rankings
    for title, path in chart_paths.items():
        if pdf.get_y() > 180: # Check for space for chart + rankings
            pdf.add_page()
            
        if unicode_font_registered:
            pdf.set_font('DejaVu', '', 14)
        else:
            pdf.set_font("Helvetica", 'B', 14)
            
        try:
            if 'XPos' in globals() and XPos:
                 pdf.cell(0, 10, _safe_text(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                 pdf.cell(0, 10, _safe_text(title), 0, 1)
        except:
             pdf.cell(0, 10, _safe_text(title), 0, 1)

        # Draw Chart
        try:
            pdf.image(path, x=10, w=190)
        except Exception as e:
            if unicode_font_registered:
                pdf.set_font('DejaVu', '', 10)
            else:
                pdf.set_font("Helvetica", 'I', 10)
            pdf.cell(0, 10, _safe_text(f"Error adding chart: {str(e)}"), 0, 1)
        
        pdf.ln(5)

        # Add ranking table if available for this chart
        if rankings and title in rankings:
            if unicode_font_registered:
                pdf.set_font('DejaVu', '', 10)
            else:
                pdf.set_font("Helvetica", '', 10)
            
            # Table Header
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(15, 8, "Rank", 1, 0, 'C', True)
            pdf.cell(80, 8, "Model", 1, 0, 'L', True)
            pdf.cell(95, 8, "Average Stats", 1, 1, 'L', True)
            
            # Table Rows
            for idx, row in enumerate(rankings[title], 1):
                model_name = row.get('model', '')
                stats_str = row.get('stats', '')
                
                pdf.cell(15, 8, str(idx), 1, 0, 'C')
                pdf.cell(80, 8, _safe_text(model_name), 1, 0, 'L')
                pdf.cell(95, 8, _safe_text(stats_str), 1, 1, 'L')
                
        pdf.ln(10)
        
    pdf_output = pdf.output(dest='S')
    
    if isinstance(pdf_output, (bytes, bytearray)):
        return bytes(pdf_output)
    try:
        return pdf_output.encode('latin-1')
    except Exception:
        return pdf_output.encode('latin-1', 'replace')


def safe_filename(name: str) -> str:
    """Generate a filesystem-safe filename from arbitrary text."""
    if not name:
        return "testset"
    sanitized = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name))
    sanitized = sanitized.strip('_')
    return sanitized or "testset"


def build_testset_report_assets(testset_results: List[Dict]) -> Dict[str, Any]:
    """Create aggregated CSV/PDF artifacts from testset run results."""
    if not testset_results:
        return {"pdf": None, "csv_df": pd.DataFrame(), "agg_df": pd.DataFrame()}

    # Aggregate per-model stats across all iterations
    model_stats: Dict[str, Dict[str, List[float]]] = {}
    for item in testset_results:
        perf = item.get('performance_stats', {}) or {}
        validation = item.get('validation', {}) or {}

        for model_name, stats in perf.items():
            if model_name not in model_stats:
                model_stats[model_name] = {'total_time': [], 'ttft': [], 'tps': [], 'accurate': []}

            if stats:
                model_stats[model_name]['total_time'].append(stats.get('total_time', 0))
                model_stats[model_name]['ttft'].append(stats.get('time_to_first_token', 0))
                model_stats[model_name]['tps'].append(stats.get('tokens_per_second', 0))

            if model_name in validation:
                model_stats[model_name]['accurate'].append(1 if validation[model_name] else 0)

    agg_data = []
    for model_name, data in model_stats.items():
        row: Dict[str, Any] = {'Model': model_name}
        if data['total_time']:
            row['Total Time (seconds)'] = sum(data['total_time']) / len(data['total_time'])
        if data['ttft']:
            row['Time to First Token (seconds)'] = sum(data['ttft']) / len(data['ttft'])
        if data['tps']:
            row['Tokens per Second'] = sum(data['tps']) / len(data['tps'])
        if data['accurate']:
            row['Accuracy'] = sum(data['accurate']) / len(data['accurate'])
            row['Accuracy_Count'] = f"({sum(data['accurate'])}/{len(data['accurate'])})"
        agg_data.append(row)

    agg_df = pd.DataFrame(agg_data)
    chart_paths: Dict[str, str] = {}
    rankings: Dict[str, List[Dict[str, str]]] = {}

    if not agg_df.empty:
        # 1) Latency (stacked) chart
        try:
            time_fig = create_stacked_time_chart(agg_df)
            if time_fig:
                title = 'Average Latency Distribution (Stacked)'
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    time_fig.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                    chart_paths[title] = tmp.name
                plt.close(time_fig)

                if 'Total Time (seconds)' in agg_df.columns:
                    sorted_df = agg_df.sort_values(by='Total Time (seconds)', ascending=True)
                    rankings[title] = []
                    for _, row in sorted_df.iterrows():
                        stats_str = f"Total: {row['Total Time (seconds)']:.2f}s"
                        if 'Time to First Token (seconds)' in row:
                            stats_str += f" | TTFT: {row['Time to First Token (seconds)']:.3f}s"
                        rankings[title].append({'model': row['Model'], 'stats': stats_str})
        except Exception as e:
            st.warning(f"Could not generate stacked time chart: {e}")

        # 2) Speed chart
        if 'Tokens per Second' in agg_df.columns:
            try:
                speed_fig = create_speed_chart(agg_df)
                if speed_fig:
                    title = 'Average Speed (Tokens/s)'
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                        speed_fig.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                        chart_paths[title] = tmp.name
                    plt.close(speed_fig)

                    sorted_df = agg_df.sort_values(by='Tokens per Second', ascending=False)
                    rankings[title] = []
                    for _, row in sorted_df.iterrows():
                        rankings[title].append({'model': row['Model'], 'stats': f"{row['Tokens per Second']:.1f} tok/s"})
            except Exception as e:
                st.warning(f"Could not generate speed chart: {e}")

        # 3) Accuracy chart
        if 'Accuracy' in agg_df.columns:
            try:
                acc_fig = create_testset_accuracy_chart(agg_df)
                if acc_fig:
                    title = 'Accuracy'
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                        acc_fig.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                        chart_paths[title] = tmp.name
                    plt.close(acc_fig)

                    sorted_df = agg_df.sort_values(by='Accuracy', ascending=False)
                    rankings[title] = []
                    for _, row in sorted_df.iterrows():
                        acc_pct = row['Accuracy'] * 100
                        count_str = row.get('Accuracy_Count', '')
                        rankings[title].append({'model': row['Model'], 'stats': f"{acc_pct:.1f}% {count_str}"})
            except Exception as e:
                st.warning(f"Could not generate accuracy chart: {e}")

    pdf_bytes = generate_testset_pdf_report(chart_paths, rankings)

    # Cleanup temp files created for charts
    for path in chart_paths.values():
        try:
            os.remove(path)
        except Exception:
            pass

    csv_df = aggregate_testset_results_to_dataframe(testset_results)
    return {"pdf": pdf_bytes, "csv_df": csv_df, "agg_df": agg_df}


def build_parallel_prompt_report_assets(prompt_label: str, testset_results: List[Dict]) -> Dict[str, Any]:
    """
    Create PDF/CSV artifacts for a specific prompt from parallel testset results.
    
    Args:
        prompt_label: The label of the prompt to generate report for
        testset_results: List of iteration results from parallel comparison
    
    Returns:
        Dictionary with 'pdf' (bytes), 'csv_df' (DataFrame), 'agg_df' (DataFrame)
    """
    if not testset_results:
        return {"pdf": None, "csv_df": pd.DataFrame(), "agg_df": pd.DataFrame()}
    
    # Filter and aggregate stats for this specific prompt across all models/iterations
    model_stats: Dict[str, Dict[str, List[float]]] = {}
    
    for result in testset_results:
        model_name = result.get('model', 'Unknown')
        stats = result.get('stats', {}).get(prompt_label, {})
        validation = result.get('validation', {})
        
        if model_name not in model_stats:
            model_stats[model_name] = {'total_time': [], 'ttft': [], 'tps': [], 'accurate': []}
        
        if stats:
            model_stats[model_name]['total_time'].append(stats.get('total_time', 0))
            model_stats[model_name]['ttft'].append(stats.get('time_to_first_token', 0))
            model_stats[model_name]['tps'].append(stats.get('tokens_per_second', 0))
        
        # Check validation for this prompt
        if prompt_label in validation:
            model_stats[model_name]['accurate'].append(1 if validation[prompt_label] else 0)
    
    # Build aggregated dataframe
    agg_data = []
    for model_name, data in model_stats.items():
        row: Dict[str, Any] = {'Model': model_name}
        if data['total_time']:
            row['Total Time (seconds)'] = sum(data['total_time']) / len(data['total_time'])
        if data['ttft']:
            row['Time to First Token (seconds)'] = sum(data['ttft']) / len(data['ttft'])
        if data['tps']:
            row['Tokens per Second'] = sum(data['tps']) / len(data['tps'])
        if data['accurate']:
            row['Accuracy'] = sum(data['accurate']) / len(data['accurate'])
            row['Accuracy_Count'] = f"({sum(data['accurate'])}/{len(data['accurate'])})"
        agg_data.append(row)
    
    agg_df = pd.DataFrame(agg_data)
    chart_paths: Dict[str, str] = {}
    rankings: Dict[str, List[Dict[str, str]]] = {}
    
    if not agg_df.empty:
        import matplotlib.pyplot as plt
        
        # 1) Latency (stacked) chart
        try:
            if 'Total Time (seconds)' in agg_df.columns and 'Time to First Token (seconds)' in agg_df.columns:
                time_fig = create_stacked_time_chart(agg_df)
                if time_fig:
                    title = f'Average Latency - {prompt_label}'
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                        time_fig.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                        chart_paths[title] = tmp.name
                    plt.close(time_fig)
                    
                    sorted_df = agg_df.sort_values(by='Total Time (seconds)', ascending=True)
                    rankings[title] = []
                    for _, row in sorted_df.iterrows():
                        stats_str = f"Total: {row['Total Time (seconds)']:.2f}s"
                        if 'Time to First Token (seconds)' in row:
                            stats_str += f" | TTFT: {row['Time to First Token (seconds)']:.3f}s"
                        rankings[title].append({'model': row['Model'], 'stats': stats_str})
        except Exception:
            pass
        
        # 2) Speed chart
        if 'Tokens per Second' in agg_df.columns:
            try:
                speed_fig = create_speed_chart(agg_df)
                if speed_fig:
                    title = f'Average Speed - {prompt_label}'
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                        speed_fig.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                        chart_paths[title] = tmp.name
                    plt.close(speed_fig)
                    
                    sorted_df = agg_df.sort_values(by='Tokens per Second', ascending=False)
                    rankings[title] = []
                    for _, row in sorted_df.iterrows():
                        rankings[title].append({'model': row['Model'], 'stats': f"{row['Tokens per Second']:.1f} tok/s"})
            except Exception:
                pass
        
        # 3) Accuracy chart
        if 'Accuracy' in agg_df.columns:
            try:
                acc_fig = create_testset_accuracy_chart(agg_df)
                if acc_fig:
                    title = f'Accuracy - {prompt_label}'
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                        acc_fig.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                        chart_paths[title] = tmp.name
                    plt.close(acc_fig)
                    
                    sorted_df = agg_df.sort_values(by='Accuracy', ascending=False)
                    rankings[title] = []
                    for _, row in sorted_df.iterrows():
                        acc_pct = row['Accuracy'] * 100
                        count_str = row.get('Accuracy_Count', '')
                        rankings[title].append({'model': row['Model'], 'stats': f"{acc_pct:.1f}% {count_str}"})
            except Exception:
                pass
    
    # Generate PDF report
    pdf_bytes = generate_testset_pdf_report(chart_paths, rankings) if chart_paths else None
    
    # Cleanup temp chart files
    for path in chart_paths.values():
        try:
            os.remove(path)
        except Exception:
            pass
    
    # Build CSV data for this prompt
    csv_rows = []
    for result in testset_results:
        model_name = result.get('model', 'Unknown')
        response = result.get('results', {}).get(prompt_label, '')
        stats = result.get('stats', {}).get(prompt_label, {})
        is_accurate = result.get('validation', {}).get(prompt_label, '')
        
        # Get expected responses for this prompt
        expected = ''
        if result.get('individual_mode'):
            prompt_expected = result.get('prompt_expected_responses', {}).get(prompt_label, [])
            expected = " | ".join(prompt_expected) if isinstance(prompt_expected, list) else str(prompt_expected)
        else:
            vars_map = result.get('vars', {})
            if isinstance(vars_map, dict):
                exp_resp = vars_map.get('_expected_responses', [])
                expected = " | ".join(exp_resp) if isinstance(exp_resp, list) else str(exp_resp)
        
        # Get variables (non-internal)
        vars_display = {}
        if result.get('individual_mode'):
            prompt_vars = result.get('vars', {}).get(prompt_label, {})
            if isinstance(prompt_vars, dict):
                vars_display = {k: v for k, v in prompt_vars.items() if not k.startswith('_')}
        else:
            vars_map = result.get('vars', {})
            if isinstance(vars_map, dict):
                vars_display = {k: v for k, v in vars_map.items() if not k.startswith('_')}
        
        csv_rows.append({
            'iteration': result.get('iteration', 0),
            'model': model_name,
            'prompt_label': prompt_label,
            **{f'var_{k}': v for k, v in vars_display.items()},
            'response': response,
            'expected': expected,
            'is_accurate': is_accurate,
            **{f'stat_{k}': v for k, v in stats.items()}
        })
    
    csv_df = pd.DataFrame(csv_rows)
    return {"pdf": pdf_bytes, "csv_df": csv_df, "agg_df": agg_df}


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
                        
                        if st.checkbox(f"{model_name}", key=f"model_{sanitize_key(model_name)}", 
                                      help=help_text, value=preselect):
                            selected_models.append(model_name)
                else:
                    with st.expander(f"{base_name} ({len(versions)} versions)", expanded=True):
                        # Check if all versions in this family should be selected
                        family_models_in_profile = [m for m in models_to_preselect if m in versions]
                        all_family_selected = len(family_models_in_profile) == len(versions)
                        
                        select_all_family = st.checkbox(
                            f"Select all {base_name} versions",
                            key=f"select_all_{sanitize_key(base_name)}",
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
                                
                                if st.checkbox(f"{version}", key=f"model_{sanitize_key(version)}", 
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
                    key=f"remove_{sanitize_key(model['name'])}",
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

# --- Handle Context Variables in User Prompt ---
user_prompt_vars = {}
prompt_template = None
template_error = None

if user_prompt:
    try:
        # Detect variables in the user prompt
        prompt_template = ChatPromptTemplate.from_template(user_prompt)
        input_vars = prompt_template.input_variables
        
        if input_vars:
            st.markdown("### Prompt Variables")
            style_cols = st.columns(3) # Use fixed columns for better layout
            
            for i, var in enumerate(input_vars):
                col_idx = i % 3
                with style_cols[col_idx]:
                    # Use session state to persist values
                    key = f"user_prompt_var_{var}"
                    user_prompt_vars[var] = st.text_input(
                        f"Value for {{{var}}}", 
                        key=key
                    )
            # Testset file upload for iterative runs
            st.markdown("#### Prompt Variable Testsets (optional)")
            uploaded_testsets = st.file_uploader(
                "Upload one or more testset files (CSV or lines of comma-separated values)",
                type=['csv','txt'],
                key='testset_uploader',
                accept_multiple_files=True
            )

            parsed_testsets = []
            if uploaded_testsets:
                for uploaded_file in uploaded_testsets:
                    try:
                        rows = parse_testset_file(uploaded_file, input_vars)
                        parsed_testsets.append({"name": uploaded_file.name, "rows": rows})
                        st.info(f"{uploaded_file.name}: parsed {len(rows)} rows from testset")
                        if len(rows) > 0:
                            preview_df = pd.DataFrame(rows[:5])
                            st.dataframe(preview_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error parsing testset file {uploaded_file.name}: {e}")
            else:
                # Clear parsed testsets if none are uploaded to avoid stale state
                st.session_state['prompt_testsets'] = []

            # Store parsed rows/sets in session for later use
            st.session_state['prompt_testsets'] = parsed_testsets
            st.session_state['prompt_testset_rows'] = parsed_testsets[0]['rows'] if parsed_testsets else []
    except Exception as e:
        # Just catch it, we'll display error if they try to run
        template_error = f"Error parsing prompt template: {str(e)}"

# Compare Models button (standard comparison)
compare_button = st.button("Compare Models", use_container_width=True)

if template_error:
    st.error(template_error)

# Add a stop button that only shows during inference
stop_button_container = st.empty()

# Progress container and bar
progress_container = st.empty()
progress_bar = st.empty()

# Section for live streaming display
streaming_section = st.empty()

# --- Parallel Prompt Comparison Section ---
st.divider()
st.header("Parallel Prompt Comparison")
st.markdown("""
Run multiple models with different prompt variations in parallel. Each model will process all prompts simultaneously, 
then move to the next model. Supports template variables and testset loading for batch comparisons.
""")

# Initialize session state for parallel prompts
if 'parallel_prompts' not in st.session_state:
    st.session_state.parallel_prompts = [
        {'label': 'Prompt A', 'prompt': ''},
        {'label': 'Prompt B', 'prompt': ''}
    ]

if 'parallel_results_all' not in st.session_state:
    st.session_state.parallel_results_all = {}  # Structure: {model: {prompt_label: response}}

if 'parallel_stats_all' not in st.session_state:
    st.session_state.parallel_stats_all = {}  # Structure: {model: {prompt_label: stats}}

if 'parallel_testset_results' not in st.session_state:
    st.session_state.parallel_testset_results = []  # List of iteration results

if 'parallel_validation_results' not in st.session_state:
    st.session_state.parallel_validation_results = {}  # Structure: {model: {prompt_label: {iteration: is_accurate}}}

# Model selection - use checkboxes to select multiple models
st.markdown("### Select Models")
st.markdown("Select one or more models to run with parallel prompts:")

base_model_names_parallel, models_info_parallel = get_available_models()
model_names_list_parallel = [model["name"] for model in models_info_parallel]

parallel_selected_models = []
if model_names_list_parallel:
    # Use the same models selected in the sidebar, or allow separate selection
    use_sidebar_models = st.checkbox(
        "Use models selected in sidebar", 
        value=True, 
        key="parallel_use_sidebar_models",
        help="If checked, uses the same models selected in the Models tab"
    )
    
    if use_sidebar_models:
        parallel_selected_models = selected_models if selected_models else []
        if parallel_selected_models:
            st.info(f"Using {len(parallel_selected_models)} model(s) from sidebar: {', '.join(parallel_selected_models)}")
        else:
            st.warning("No models selected in sidebar. Please select models in the Models tab or uncheck this option.")
    else:
        # Show multiselect for parallel-specific model selection
        parallel_selected_models = st.multiselect(
            "Select models for parallel comparison",
            options=model_names_list_parallel,
            default=[],
            key="parallel_model_multiselect",
            help="Select one or more models to run with parallel prompts"
        )
else:
    st.warning("No models available. Please ensure Ollama is running.")

# Number of prompts control
num_prompts = st.number_input(
    "Number of parallel prompts",
    min_value=2,
    max_value=10,
    value=len(st.session_state.parallel_prompts),
    key="num_parallel_prompts",
    help="How many different prompts to run in parallel (2-10)"
)

# Adjust parallel_prompts list based on num_prompts
if num_prompts > len(st.session_state.parallel_prompts):
    for i in range(len(st.session_state.parallel_prompts), num_prompts):
        st.session_state.parallel_prompts.append({'label': f'Prompt {chr(65+i)}', 'prompt': ''})
elif num_prompts < len(st.session_state.parallel_prompts):
    st.session_state.parallel_prompts = st.session_state.parallel_prompts[:num_prompts]

# Display prompt inputs
st.markdown("### Define Prompt Templates")
st.markdown("Use `{variable_name}` syntax to include template variables that can be filled from testsets.")

# Option for individual testsets per prompt
use_individual_testsets = st.checkbox(
    "Use individual testsets for each prompt",
    value=False,
    key="use_individual_testsets",
    help="When enabled, you can upload a separate testset file for each prompt template"
)

parallel_prompts_data = []
parallel_prompt_vars = set()  # Collect all variables across prompts
individual_testsets = {}  # Store testsets per prompt label
individual_prompt_vars = {}  # Store detected vars per prompt

for i, prompt_data in enumerate(st.session_state.parallel_prompts):
    with st.expander(f"**{prompt_data['label']}**", expanded=True):
        col1, col2 = st.columns([1, 4])
        with col1:
            label = st.text_input(
                "Label",
                value=prompt_data['label'],
                key=f"parallel_label_{i}",
                help="A short identifier for this prompt"
            )
        with col2:
            prompt_text = st.text_area(
                "Prompt Template",
                value=prompt_data['prompt'],
                key=f"parallel_prompt_{i}",
                height=100,
                help="The prompt text. Use {var_name} for template variables."
            )
        
        # Detect variables in this prompt
        prompt_specific_vars = set()
        if prompt_text:
            try:
                temp_template = ChatPromptTemplate.from_template(prompt_text)
                prompt_specific_vars = set(temp_template.input_variables)
                parallel_prompt_vars.update(prompt_specific_vars)
            except:
                pass
        
        # Show detected variables for this prompt
        if prompt_specific_vars:
            st.caption(f"Variables: {', '.join(sorted(prompt_specific_vars))}")
        
        # Individual testset upload (when enabled)
        if use_individual_testsets and label and prompt_text:
            st.markdown("---")
            individual_testset_file = st.file_uploader(
                f"Testset for {label}",
                type=['csv', 'txt'],
                key=f'individual_testset_{i}',
                help=f"Upload a testset file specific to this prompt. Should contain columns: {', '.join(sorted(prompt_specific_vars)) if prompt_specific_vars else 'matching your variables'}"
            )
            
            if individual_testset_file and prompt_specific_vars:
                try:
                    rows = parse_testset_file(individual_testset_file, list(prompt_specific_vars))
                    individual_testsets[label] = rows
                    individual_prompt_vars[label] = prompt_specific_vars
                    st.success(f"Parsed {len(rows)} rows")
                    
                    # Check for expected responses
                    has_expected = any('_expected_responses' in row for row in rows)
                    if has_expected:
                        st.caption("✓ Expected responses detected")
                except Exception as e:
                    st.error(f"Error parsing: {e}")
        
        # Update session state
        st.session_state.parallel_prompts[i] = {'label': label, 'prompt': prompt_text}
        
        if label and prompt_text:
            parallel_prompts_data.append({'label': label, 'prompt': prompt_text, 'vars': prompt_specific_vars})

# Show detected variables
if parallel_prompt_vars:
    st.info(f"Detected template variables (all prompts): {', '.join(sorted(parallel_prompt_vars))}")

# --- Testset Upload for Parallel Prompts ---
# Only show shared testset upload when NOT using individual testsets
parallel_testset_rows = []

if not use_individual_testsets:
    st.markdown("### Shared Testset for All Prompts (Optional)")
    st.markdown("Upload a testset file to run all parallel prompts across multiple variable combinations.")

    parallel_testset_file = st.file_uploader(
        "Upload testset file (CSV or pipe-delimited)",
        type=['csv', 'txt'],
        key='parallel_testset_uploader',
        help="File should contain columns matching template variables. Use ||| as delimiter for complex data."
    )

    if parallel_testset_file and parallel_prompt_vars:
        try:
            parallel_testset_rows = parse_testset_file(parallel_testset_file, list(parallel_prompt_vars))
            st.success(f"Parsed {len(parallel_testset_rows)} rows from testset")
            if parallel_testset_rows:
                # Show preview
                preview_df = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith('_')} 
                                           for row in parallel_testset_rows[:5]])
                st.dataframe(preview_df, use_container_width=True)
                
                # Check for expected responses
                has_expected = any('_expected_responses' in row for row in parallel_testset_rows)
                if has_expected:
                    st.info("✓ Expected responses detected - accuracy checking will be enabled")
        except Exception as e:
            st.error(f"Error parsing testset: {e}")
else:
    # Show summary of individual testsets
    if individual_testsets:
        st.markdown("### Individual Testsets Summary")
        for prompt_label, rows in individual_testsets.items():
            has_expected = any('_expected_responses' in row for row in rows)
            expected_indicator = " ✓" if has_expected else ""
            st.caption(f"**{prompt_label}**: {len(rows)} rows{expected_indicator}")

# Manual variable input (if no testset or for single run, and not using individual testsets)
parallel_manual_vars = {}
if parallel_prompt_vars and not parallel_testset_rows and not use_individual_testsets:
    st.markdown("#### Manual Variable Input")
    var_cols = st.columns(min(3, len(parallel_prompt_vars)))
    for i, var in enumerate(sorted(parallel_prompt_vars)):
        with var_cols[i % len(var_cols)]:
            parallel_manual_vars[var] = st.text_input(
                f"Value for {{{var}}}",
                key=f"parallel_var_{var}"
            )

# Button to run parallel prompts
parallel_compare_button = st.button(
    "Run Parallel Comparison", 
    use_container_width=True,
    disabled=len(parallel_selected_models) == 0 or len(parallel_prompts_data) < 2,
    key="parallel_compare_btn"
)

# Handle parallel comparison execution
if parallel_compare_button:
    if len(parallel_prompts_data) < 2:
        st.error("Please define at least 2 prompts with labels to run parallel comparison.")
    elif not parallel_selected_models:
        st.error("Please select at least one model for parallel comparison.")
    else:
        # Clear previous results
        st.session_state.parallel_results_all = {}
        st.session_state.parallel_stats_all = {}
        st.session_state.parallel_testset_results = []
        st.session_state.parallel_validation_results = {}
        st.session_state.parallel_prompt_reports = {}
        st.session_state.debug_info = []
        
        # Determine execution mode based on testset configuration
        use_individual_mode = use_individual_testsets and individual_testsets
        
        if use_individual_mode:
            # Individual testsets mode: each prompt has its own testset
            # Find max iterations across all individual testsets
            max_iterations = max(len(rows) for rows in individual_testsets.values()) if individual_testsets else 1
            
            # Create iteration structure where each prompt uses its own testset row
            iterations = []
            for iter_idx in range(max_iterations):
                iter_data = {
                    '_individual_mode': True,
                    '_iteration_index': iter_idx
                }
                # For each prompt, get its corresponding testset row (or cycle if shorter)
                for prompt_label, testset_rows in individual_testsets.items():
                    if testset_rows:
                        row_idx = iter_idx % len(testset_rows)
                        iter_data[prompt_label] = testset_rows[row_idx]
                iterations.append(iter_data)
            
            st.session_state.debug_info.append(
                f"Individual testsets mode: {len(individual_testsets)} prompts with testsets, "
                f"{max_iterations} max iterations"
            )
        elif parallel_testset_rows:
            # Shared testset mode
            iterations = parallel_testset_rows
        elif parallel_prompt_vars and all(parallel_manual_vars.get(v) for v in parallel_prompt_vars):
            iterations = [parallel_manual_vars]
        elif not parallel_prompt_vars:
            # No variables, just run prompts as-is
            iterations = [{}]
        else:
            st.error("Please provide values for all template variables or upload a testset.")
            iterations = []
        
        if iterations:
            total_operations = len(parallel_selected_models) * len(parallel_prompts_data) * len(iterations)
            st.session_state.debug_info.append(
                f"Starting parallel comparison: {len(parallel_selected_models)} models × "
                f"{len(parallel_prompts_data)} prompts × {len(iterations)} iterations = {total_operations} total operations"
            )
            
            # Progress tracking
            parallel_progress_bar = st.progress(0)
            parallel_status = st.empty()
            completed_ops = 0
            
            # Process each model
            for model_idx, model_name in enumerate(parallel_selected_models):
                if st.session_state.get('stop_inference', False):
                    break
                    
                st.session_state.parallel_results_all[model_name] = {}
                st.session_state.parallel_stats_all[model_name] = {}
                
                parallel_status.markdown(f"**Processing model {model_idx + 1}/{len(parallel_selected_models)}: {model_name}**")
                st.session_state.debug_info.append(f"\n=== Model: {model_name} ===")
                
                # Process each iteration (testset row or single run)
                for iter_idx, vars_map in enumerate(iterations):
                    if st.session_state.get('stop_inference', False):
                        break
                    
                    # Check if this is individual testset mode
                    is_individual_mode = vars_map.get('_individual_mode', False) if isinstance(vars_map, dict) else False
                    
                    # Build formatted prompts for this iteration
                    formatted_prompts = []
                    prompt_expected_responses = {}  # Store expected responses per prompt for individual mode
                    
                    for prompt_data in parallel_prompts_data:
                        prompt_label = prompt_data['label']
                        try:
                            if is_individual_mode:
                                # Individual testset mode: get vars specific to this prompt
                                prompt_vars = vars_map.get(prompt_label, {})
                                if prompt_vars and isinstance(prompt_vars, dict):
                                    # Store expected responses for this prompt
                                    if '_expected_responses' in prompt_vars:
                                        prompt_expected_responses[prompt_label] = prompt_vars['_expected_responses']
                                    
                                    # Format with prompt-specific variables
                                    temp_template = ChatPromptTemplate.from_template(prompt_data['prompt'])
                                    # Filter to only vars needed by this prompt
                                    relevant_vars = {k: v for k, v in prompt_vars.items() if not k.startswith('_')}
                                    messages = temp_template.format_messages(**relevant_vars)
                                    if messages and hasattr(messages[0], 'content'):
                                        formatted_text = messages[0].content
                                    else:
                                        formatted_text = temp_template.format(**relevant_vars)
                                else:
                                    formatted_text = prompt_data['prompt']
                            elif vars_map and parallel_prompt_vars:
                                # Shared testset mode: format the prompt with shared variables
                                temp_template = ChatPromptTemplate.from_template(prompt_data['prompt'])
                                messages = temp_template.format_messages(**vars_map)
                                if messages and hasattr(messages[0], 'content'):
                                    formatted_text = messages[0].content
                                else:
                                    formatted_text = temp_template.format(**vars_map)
                            else:
                                formatted_text = prompt_data['prompt']
                            
                            formatted_prompts.append({
                                'label': prompt_label,
                                'prompt': formatted_text,
                                'original': prompt_data['prompt']
                            })
                        except Exception as e:
                            st.session_state.debug_info.append(f"Error formatting {prompt_label}: {e}")
                            formatted_prompts.append({
                                'label': prompt_label,
                                'prompt': prompt_data['prompt'],
                                'original': prompt_data['prompt'],
                                'error': str(e)
                            })
                    
                    # Run all prompts in parallel for this model and iteration
                    iter_label = f"Iteration {iter_idx + 1}" if len(iterations) > 1 else "Single Run"
                    parallel_status.markdown(
                        f"**Model {model_idx + 1}/{len(parallel_selected_models)}: {model_name}** | "
                        f"**{iter_label}/{len(iterations)}** | Running {len(formatted_prompts)} prompts in parallel..."
                    )
                    
                    results, stats, debug_logs = run_parallel_prompts(
                        model_name,
                        formatted_prompts,
                        system_prompt,
                        temperature
                    )
                    
                    # Append collected debug logs to session state (thread-safe now)
                    st.session_state.debug_info.extend(debug_logs)
                    
                    # Store results - include per-prompt vars for individual mode
                    iteration_result = {
                        'iteration': iter_idx + 1,
                        'vars': vars_map,
                        'model': model_name,
                        'results': results,
                        'stats': {k: v.to_dict() if hasattr(v, 'to_dict') else {} for k, v in stats.items()},
                        'validation': {},
                        'individual_mode': is_individual_mode,
                        'prompt_expected_responses': prompt_expected_responses
                    }
                    
                    # Update aggregate results (for single iteration display)
                    if len(iterations) == 1:
                        st.session_state.parallel_results_all[model_name] = results
                        st.session_state.parallel_stats_all[model_name] = stats
                    
                    # Run accuracy checking if expected responses exist
                    eval_model = st.session_state.get("evaluation_model_value", "") or evaluation_model
                    
                    if is_individual_mode and prompt_expected_responses:
                        # Individual mode: each prompt has its own expected responses
                        if eval_model:
                            for prompt_label, response in results.items():
                                expected = prompt_expected_responses.get(prompt_label, [])
                                if expected:
                                    expected_str = " | ".join(expected) if isinstance(expected, list) else str(expected)
                                    try:
                                        formatted_prompt = next(
                                            (p['prompt'] for p in formatted_prompts if p['label'] == prompt_label), 
                                            ""
                                        )
                                        is_accurate = evaluate_response_accuracy(
                                            eval_model, formatted_prompt, response, expected_str
                                        )
                                        iteration_result['validation'][prompt_label] = is_accurate
                                        st.session_state.debug_info.append(
                                            f"  {prompt_label}: {'✓ Accurate' if is_accurate else '✗ Not Accurate'}"
                                        )
                                    except Exception as e:
                                        st.session_state.debug_info.append(f"  {prompt_label}: Error judging - {e}")
                                        iteration_result['validation'][prompt_label] = False
                    else:
                        # Shared mode: all prompts use the same expected responses
                        expected_responses = vars_map.get('_expected_responses', []) if vars_map and isinstance(vars_map, dict) else []
                        if expected_responses:
                            expected_str = " | ".join(expected_responses) if isinstance(expected_responses, list) else str(expected_responses)
                            
                            if eval_model:
                                for prompt_label, response in results.items():
                                    try:
                                        # Get the formatted prompt for context
                                        formatted_prompt = next(
                                            (p['prompt'] for p in formatted_prompts if p['label'] == prompt_label), 
                                            ""
                                        )
                                        is_accurate = evaluate_response_accuracy(
                                            eval_model, formatted_prompt, response, expected_str
                                        )
                                        iteration_result['validation'][prompt_label] = is_accurate
                                        st.session_state.debug_info.append(
                                            f"  {prompt_label}: {'✓ Accurate' if is_accurate else '✗ Not Accurate'}"
                                        )
                                    except Exception as e:
                                        st.session_state.debug_info.append(f"  {prompt_label}: Error judging - {e}")
                                        iteration_result['validation'][prompt_label] = False
                    
                    st.session_state.parallel_testset_results.append(iteration_result)
                    
                    completed_ops += len(formatted_prompts)
                    parallel_progress_bar.progress(completed_ops / total_operations)
            
            parallel_progress_bar.empty()
            parallel_status.empty()
            
            # Generate PDF reports for each prompt (individual testset mode or shared testset mode)
            if use_individual_mode or len(iterations) > 1:
                st.session_state.parallel_prompt_reports = {}
                parallel_status.markdown("**Generating PDF reports for each prompt...**")
                
                # Get unique prompt labels from the results
                prompt_labels = set()
                for result in st.session_state.parallel_testset_results:
                    prompt_labels.update(result.get('results', {}).keys())
                
                for prompt_label in prompt_labels:
                    try:
                        assets = build_parallel_prompt_report_assets(
                            prompt_label, 
                            st.session_state.parallel_testset_results
                        )
                        st.session_state.parallel_prompt_reports[prompt_label] = assets
                        st.session_state.debug_info.append(f"Generated PDF report for prompt: {prompt_label}")
                    except Exception as e:
                        st.session_state.debug_info.append(f"Error generating PDF for {prompt_label}: {e}")
                
                parallel_status.empty()
            
            st.success(f"Completed parallel comparison across {len(parallel_selected_models)} model(s)!")

# Display parallel comparison results
if st.session_state.parallel_results_all or st.session_state.parallel_testset_results:
    st.markdown("### Parallel Comparison Results")
    
    # Check if this is a testset run with multiple iterations
    is_testset_run = len(st.session_state.parallel_testset_results) > len(parallel_selected_models) if parallel_selected_models else False
    
    if is_testset_run:
        # Testset results view
        st.markdown("#### Testset Results Summary")
        
        # Aggregate stats by model and prompt label
        agg_data = {}
        for result in st.session_state.parallel_testset_results:
            model = result['model']
            if model not in agg_data:
                agg_data[model] = {}
            
            for prompt_label, response in result['results'].items():
                if prompt_label not in agg_data[model]:
                    agg_data[model][prompt_label] = {
                        'count': 0,
                        'accurate': 0,
                        'total_time': 0,
                        'total_tokens': 0
                    }
                
                agg_data[model][prompt_label]['count'] += 1
                if result['validation'].get(prompt_label, False):
                    agg_data[model][prompt_label]['accurate'] += 1
                
                stats = result['stats'].get(prompt_label, {})
                agg_data[model][prompt_label]['total_time'] += stats.get('total_time', 0)
                agg_data[model][prompt_label]['total_tokens'] += stats.get('completion_tokens', 0)
        
        # Create summary table
        summary_rows = []
        for model, prompts in agg_data.items():
            for prompt_label, data in prompts.items():
                accuracy_pct = (data['accurate'] / data['count'] * 100) if data['count'] > 0 else 0
                avg_time = data['total_time'] / data['count'] if data['count'] > 0 else 0
                summary_rows.append({
                    'Model': model,
                    'Prompt': prompt_label,
                    'Iterations': data['count'],
                    'Accurate': data['accurate'],
                    'Accuracy %': f"{accuracy_pct:.1f}%",
                    'Avg Time (s)': round(avg_time, 2),
                    'Total Tokens': data['total_tokens']
                })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Detailed results in expander
        with st.expander("Detailed Iteration Results"):
            for result in st.session_state.parallel_testset_results:
                vars_display = {k: v for k, v in result['vars'].items() if not k.startswith('_')}
                st.markdown(f"**{result['model']}** - Iteration {result['iteration']}: {vars_display}")
                
                result_cols = st.columns(len(result['results']))
                for i, (label, response) in enumerate(result['results'].items()):
                    with result_cols[i]:
                        is_accurate = result['validation'].get(label)
                        accuracy_icon = "✓" if is_accurate else "✗" if is_accurate is not None else ""
                        st.markdown(f"**{label}** {accuracy_icon}")
                        st.text(response[:200] + "..." if len(response) > 200 else response)
                st.divider()
        
        # Export testset results
        st.markdown("#### Export Testset Results")
        
        # Build CSV data
        csv_rows = []
        for result in st.session_state.parallel_testset_results:
            vars_display = {k: v for k, v in result['vars'].items() if not k.startswith('_')}
            expected = result['vars'].get('_expected_responses', [])
            expected_str = " | ".join(expected) if isinstance(expected, list) else str(expected)
            
            for prompt_label, response in result['results'].items():
                csv_rows.append({
                    'iteration': result['iteration'],
                    'model': result['model'],
                    'prompt_label': prompt_label,
                    **{f'var_{k}': v for k, v in vars_display.items()},
                    'response': response,
                    'expected': expected_str,
                    'is_accurate': result['validation'].get(prompt_label, ''),
                    **{f'stat_{k}': v for k, v in result['stats'].get(prompt_label, {}).items()}
                })
        
        if csv_rows:
            csv_df = pd.DataFrame(csv_rows)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Testset Results (CSV)",
                    data=csv_df.to_csv(index=False),
                    file_name="parallel_testset_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="parallel_testset_csv_download"
                )
            with col2:
                st.download_button(
                    label="Download Testset Results (JSON)",
                    data=json.dumps(st.session_state.parallel_testset_results, indent=2, default=str),
                    file_name="parallel_testset_results.json",
                    mime="application/json",
                    use_container_width=True,
                    key="parallel_testset_json_download"
                )
        
        # Display PDF reports for each prompt
        if st.session_state.get('parallel_prompt_reports'):
            st.markdown("#### PDF Reports by Prompt")
            st.markdown("Each prompt has its own PDF report with performance charts and accuracy metrics.")
            
            for prompt_label, assets in st.session_state.parallel_prompt_reports.items():
                safe_name = safe_filename(prompt_label) or f"prompt_{prompt_label}"
                
                with st.expander(f"**{prompt_label}** Report", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    # PDF download
                    if assets.get('pdf'):
                        with col1:
                            st.download_button(
                                label=f"📄 PDF Report",
                                data=assets['pdf'],
                                file_name=f"parallel_{safe_name}_report.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key=f"parallel_pdf_{safe_name}"
                            )
                    
                    # CSV download
                    csv_df = assets.get('csv_df')
                    if csv_df is not None and not csv_df.empty:
                        with col2:
                            st.download_button(
                                label=f"📊 CSV Data",
                                data=csv_df.to_csv(index=False),
                                file_name=f"parallel_{safe_name}_results.csv",
                                mime="text/csv",
                                use_container_width=True,
                                key=f"parallel_csv_{safe_name}"
                            )
                    
                    # Show aggregated stats summary
                    agg_df = assets.get('agg_df')
                    if agg_df is not None and not agg_df.empty:
                        with col3:
                            st.caption("Summary Stats")
                        st.dataframe(agg_df, use_container_width=True, hide_index=True)
    
    else:
        # Single run results view (multiple models, single iteration)
        if st.session_state.parallel_results_all:
            # Performance stats table across all models
            with st.expander("Performance Statistics", expanded=True):
                stats_data = []
                for model_name, prompt_stats in st.session_state.parallel_stats_all.items():
                    for label, stats in prompt_stats.items():
                        stats_data.append({
                            "Model": model_name,
                            "Prompt": label,
                            "Total Time (s)": round(stats.total_time, 2) if hasattr(stats, 'total_time') else 0,
                            "Tokens Generated": stats.completion_tokens if hasattr(stats, 'completion_tokens') else 0,
                            "Tokens/sec": round(stats.tokens_per_second, 1) if hasattr(stats, 'tokens_per_second') else 0,
                        })
                
                if stats_data:
                    parallel_stats_df = pd.DataFrame(stats_data)
                    st.dataframe(parallel_stats_df, use_container_width=True, hide_index=True)
            
            # Results display - organized by model
            for model_name, results in st.session_state.parallel_results_all.items():
                st.markdown(f"#### Model: {model_name}")
                
                result_labels = list(results.keys())
                if len(result_labels) <= 3:
                    cols = st.columns(len(result_labels))
                    for i, label in enumerate(result_labels):
                        with cols[i]:
                            response = results[label]
                            if remove_think_blocks_setting:
                                response = remove_think_blocks(response)
                            st.markdown(f"**{label}**")
                            st.markdown(f"<div class='model-response'>{html.escape(response)}</div>", 
                                       unsafe_allow_html=True)
                else:
                    tabs = st.tabs(result_labels)
                    for i, label in enumerate(result_labels):
                        with tabs[i]:
                            response = results[label]
                            if remove_think_blocks_setting:
                                response = remove_think_blocks(response)
                            st.markdown(f"<div class='model-response'>{html.escape(response)}</div>", 
                                       unsafe_allow_html=True)
                
                st.divider()
            
            # Export options
            st.markdown("### Export Parallel Results")
            
            # Build flat results for CSV
            csv_rows = []
            for model_name, results in st.session_state.parallel_results_all.items():
                for label, response in results.items():
                    stats = st.session_state.parallel_stats_all.get(model_name, {}).get(label)
                    csv_rows.append({
                        "Model": model_name,
                        "Prompt Label": label,
                        "Response": response,
                        "Response Length": len(response),
                        "Total Time (s)": stats.total_time if stats and hasattr(stats, 'total_time') else '',
                        "Tokens Generated": stats.completion_tokens if stats and hasattr(stats, 'completion_tokens') else '',
                    })
            
            parallel_results_df = pd.DataFrame(csv_rows)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Parallel Results (CSV)",
                    data=parallel_results_df.to_csv(index=False),
                    file_name="parallel_prompt_comparison.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="parallel_csv_download"
                )
            
            with col2:
                parallel_json_data = {
                    "models": list(st.session_state.parallel_results_all.keys()),
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                    "prompts": parallel_prompts_data,
                    "results_by_model": {
                        model: {
                            "responses": results,
                            "stats": {
                                k: v.to_dict() if hasattr(v, 'to_dict') else {}
                                for k, v in st.session_state.parallel_stats_all.get(model, {}).items()
                            }
                        }
                        for model, results in st.session_state.parallel_results_all.items()
                    }
                }
                st.download_button(
                    label="Download Parallel Results (JSON)",
                    data=json.dumps(parallel_json_data, indent=2),
                    file_name="parallel_prompt_comparison.json",
                    mime="application/json",
                    use_container_width=True,
                    key="parallel_json_download"
                )
    
    # Debug info for parallel runs
    with st.expander("Parallel Run Debug Information"):
        st.code("\n".join(st.session_state.debug_info))
    
    # Clear results button
    if st.button("Clear Parallel Results", key="clear_parallel_results"):
        st.session_state.parallel_results_all = {}
        st.session_state.parallel_stats_all = {}
        st.session_state.parallel_testset_results = []
        st.session_state.parallel_validation_results = {}
        st.session_state.parallel_prompt_reports = {}
        st.rerun()

# Define the model processing function
def process_models(selected_models, system_prompt_text=None, user_prompt_text=None, iteration_info=None, render_debug=True):
    """Process all selected models and collect responses with performance stats."""
    # Default to globol user_prompt if not provided
    if user_prompt_text is None:
        user_prompt_text = user_prompt
        
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
                msg = f"**Model {i+1}/{len(selected_models)}:** {model} | Starting..."
                if iteration_info:
                    msg = f"**{iteration_info}** | " + msg
                progress_container.markdown(msg)
                progress_bar.progress((i) / len(selected_models))
                
                # Reset streaming display for next model
                st.session_state.current_streaming_text = ""
                
                response, stats = query_model_streaming(
                    model, 
                    user_prompt_text, 
                    system_prompt_text, 
                    temperature, 
                    progress_container,
                    streaming_display
                )
            else:
                # Show processing message when not streaming
                msg = f"**Generating {i+1}/{len(selected_models)}:** {model}"
                if iteration_info:
                    msg = f"**{iteration_info}** | " + msg
                progress_container.markdown(msg)
                progress_bar.progress((i) / len(selected_models))
                response, stats = query_model(model, user_prompt_text, system_prompt_text, temperature)
            
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
    
    # Show debug information in an expander if requested
    if render_debug:
        with st.expander("Debug Information"):
            st.code("\n".join(st.session_state.debug_info))
    
    # Clear stop button when done
    stop_button_container.empty()
    
    # Set inference running flag to false
    st.session_state.inference_running = False


def remove_think_blocks(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

if compare_button:
    # --- Check for variable errors ---
    validation_error = None
    final_user_prompt = user_prompt
    # If a testset was uploaded, we should not require individual prompt variable inputs
    testset_rows = st.session_state.get('prompt_testset_rows', []) or []
    
    if prompt_template:
        # If a testset is present, skip per-variable validation here —
        # iterations will build prompts from each testset row.
        if not testset_rows:
            missing_vars = [var for var, val in user_prompt_vars.items() if not val]
            if missing_vars:
                validation_error = f"Missing values for prompt variables: {', '.join(missing_vars)}"
            else:
                try:
                    # Format the user prompt
                    # ChatPromptTemplate.format returns a string if the template produces a string
                    messages = prompt_template.format_messages(**user_prompt_vars)
                    # Extract content from the first message
                    if messages and hasattr(messages[0], 'content'):
                        final_user_prompt = messages[0].content
                    else:
                        final_user_prompt = prompt_template.format(**user_prompt_vars)
                except Exception as e:
                    validation_error = f"Error formatting prompt: {str(e)}"
        else:
            # When using a testset, we won't build a single final_user_prompt here.
            final_user_prompt = user_prompt

    if validation_error:
        st.error(validation_error)
    elif not selected_models:
        st.warning("Please select at least one model to compare.")
    else:
        # Clear accuracy ratings
        keys_to_remove = [k for k in st.session_state.keys() if k.startswith("rating_")]
        for k in keys_to_remove:
            del st.session_state[k]

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
                
        # Determine if one or more testsets were uploaded and parsed
        testsets = st.session_state.get('prompt_testsets', []) or []
        fallback_rows = st.session_state.get('prompt_testset_rows', []) or []
        if not testsets and fallback_rows:
            # Backward compatibility: treat single parsed rows as one testset
            testsets = [{"name": "Testset 1", "rows": fallback_rows}]

        if testsets:
            st.session_state.testset_batch_reports = []
            st.session_state.testset_batch_debug_logs = []
            
            # Calculate total operations across ALL test sets for batch progress tracking
            total_batch_ops = 0
            for ts in testsets:
                ts_rows = ts.get('rows', []) or []
                num_entries = len(ts_rows)
                # Operations per testset: (models * entries) for inference + (models * entries_with_expected) for judging
                total_batch_ops += len(selected_models) * num_entries
                # Estimate judging operations (assume all entries have expected responses for simplicity)
                total_batch_ops += len(selected_models) * num_entries
            
            batch_progress_container = st.empty()
            batch_progress_bar = st.empty()
            completed_batch_ops = 0
            
            st.info(f"Running {len(testsets)} testset(s) sequentially")

            for ts_idx, testset in enumerate(testsets, start=1):
                if st.session_state.stop_inference:
                    st.session_state.debug_info.append("Testset run stopped by user")
                    break

                ts_name = testset.get('name') or f"Testset {ts_idx}"
                testset_rows = testset.get('rows', []) or []
                if not testset_rows:
                    continue

                st.session_state.testset_results = []
                consolidated_debug_info = []
                batch_progress_container.markdown(f"**Batch Progress:** Testset {ts_idx}/{len(testsets)} - {ts_name}")
                st.info(f"{ts_name}: running {len(testset_rows)} entries across {len(selected_models)} models (optimized batch mode)")
                
                # OPTIMIZED APPROACH: Run all test entries on each model before switching
                # This minimizes model loading/unloading overhead
                
                # Step 1: Build all prompts upfront
                all_prompts = []
                for idx, vars_map in enumerate(testset_rows, start=1):
                    try:
                        messages = prompt_template.format_messages(**vars_map)
                        if messages and hasattr(messages[0], 'content'):
                            iter_prompt = messages[0].content
                        else:
                            iter_prompt = prompt_template.format(**vars_map)
                        all_prompts.append({'idx': idx, 'prompt': iter_prompt, 'vars': vars_map})
                    except Exception as e:
                        st.error(f"Error formatting prompt for iteration {idx}: {e}")
                        all_prompts.append({'idx': idx, 'prompt': None, 'vars': vars_map, 'error': str(e)})
                
                # Initialize results storage: dict keyed by entry index, each containing model results
                # Structure: {entry_idx: {'vars': vars_map, 'results': {model: response}, 'performance_stats': {model: stats}}}
                entry_results = {}
                for prompt_data in all_prompts:
                    entry_results[prompt_data['idx']] = {
                        'vars': prompt_data['vars'],
                        'results': {},
                        'performance_stats': {},
                        'validation': {}
                    }
                
                # Step 2: Process each model across ALL test entries before moving to next model
                total_operations = len(selected_models) * len(all_prompts)
                current_op = 0
                
                for model_idx, model in enumerate(selected_models, start=1):
                    if st.session_state.stop_inference:
                        st.session_state.debug_info.append("Testset run stopped by user")
                        consolidated_debug_info.append("Testset run stopped by user")
                        break
                    
                    consolidated_debug_info.append(f"=== Loading model: {model} ===")
                    
                    # Process all entries with this model
                    for prompt_data in all_prompts:
                        if st.session_state.stop_inference:
                            break
                        
                        entry_idx = prompt_data['idx']
                        iter_prompt = prompt_data.get('prompt')
                        
                        if iter_prompt is None:
                            # Skip entries that had formatting errors
                            current_op += 1
                            completed_batch_ops += 1
                            continue
                        
                        current_op += 1
                        iteration_info = f"{ts_name} | Model {model_idx}/{len(selected_models)}: {model} | Entry {entry_idx}/{len(all_prompts)}"
                        
                        # Set up streaming display if enabled
                        streaming_display = streaming_section.empty() if enable_streaming else None
                        
                        try:
                            if enable_streaming:
                                progress_container.markdown(f"**{iteration_info}** | Starting...")
                                progress_bar.progress(current_op / total_operations)
                                st.session_state.current_streaming_text = ""
                                
                                response, stats = query_model_streaming(
                                    model, iter_prompt, system_prompt, temperature,
                                    progress_container, streaming_display
                                )
                            else:
                                progress_container.markdown(f"**{iteration_info}**")
                                progress_bar.progress(current_op / total_operations)
                                response, stats = query_model(model, iter_prompt, system_prompt, temperature)
                            
                            entry_results[entry_idx]['results'][model] = response
                            entry_results[entry_idx]['performance_stats'][model] = stats.to_dict()
                            
                            consolidated_debug_info.append(
                                f"  Entry {entry_idx}: {stats.completion_tokens} tokens in {stats.total_time:.2f}s "
                                f"({stats.tokens_per_second:.1f} tok/s)"
                            )
                            
                        except Exception as e:
                            error_msg = f"Error with {model} on entry {entry_idx}: {str(e)}"
                            consolidated_debug_info.append(f"  {error_msg}")
                            entry_results[entry_idx]['results'][model] = f"Error: {str(e)}"
                            entry_results[entry_idx]['performance_stats'][model] = PerformanceStats(model_name=model).to_dict()
                        
                        # Update batch progress
                        completed_batch_ops += 1
                        batch_progress_bar.progress(completed_batch_ops / max(total_batch_ops, 1))
                        batch_progress_container.markdown(f"**Batch Progress:** Testset {ts_idx}/{len(testsets)} | {completed_batch_ops}/{total_batch_ops} operations")
                        
                        if streaming_display:
                            streaming_display.empty()
                    
                    consolidated_debug_info.append(f"=== Completed {model} for all entries ===")
                    consolidated_debug_info.append("-" * 40)
                
                # Step 3: After all models have processed all entries, run the judge model for accuracy
                evaluation_model_name = st.session_state.get("evaluation_model_value", "") or evaluation_model
                has_expected_responses = any('_expected_responses' in prompt_data['vars'] for prompt_data in all_prompts)
                
                if evaluation_model_name and has_expected_responses:
                    consolidated_debug_info.append(f"=== Loading judge model: {evaluation_model_name} ===")
                    progress_container.markdown(f"**{ts_name}** | Running accuracy evaluation with {evaluation_model_name}...")
                    
                    total_judgments = sum(
                        len(entry_results[prompt_data['idx']]['results']) 
                        for prompt_data in all_prompts 
                        if '_expected_responses' in prompt_data['vars']
                    )
                    judgment_count = 0
                    
                    for prompt_data in all_prompts:
                        if st.session_state.stop_inference:
                            break
                        
                        entry_idx = prompt_data['idx']
                        vars_map = prompt_data['vars']
                        iter_prompt = prompt_data.get('prompt')
                        
                        if iter_prompt is None:
                            continue
                        
                        # Get expected responses
                        expected_resp_str = None
                        if '_expected_responses' in vars_map:
                            exps = vars_map['_expected_responses']
                            if isinstance(exps, list):
                                expected_resp_str = " | ".join(exps)
                            else:
                                expected_resp_str = str(exps)
                        
                        if expected_resp_str:
                            consolidated_debug_info.append(f"  Judging entry {entry_idx}...")
                            
                            for model, response in entry_results[entry_idx]['results'].items():
                                if st.session_state.stop_inference:
                                    break
                                
                                judgment_count += 1
                                progress_bar.progress(judgment_count / max(total_judgments, 1))
                                progress_container.markdown(
                                    f"**{ts_name}** | Judging entry {entry_idx}, model {model} ({judgment_count}/{total_judgments})"
                                )
                                
                                try:
                                    is_accurate = evaluate_response_accuracy(
                                        evaluation_model_name, iter_prompt, response, expected_resp_str
                                    )
                                    entry_results[entry_idx]['validation'][model] = is_accurate
                                    consolidated_debug_info.append(
                                        f"    {model}: {'Accurate' if is_accurate else 'Not Accurate'}"
                                    )
                                except Exception as e:
                                    consolidated_debug_info.append(f"    {model}: Error during judgment - {str(e)}")
                                    entry_results[entry_idx]['validation'][model] = False
                                
                                # Update batch progress for judging
                                completed_batch_ops += 1
                                batch_progress_bar.progress(completed_batch_ops / max(total_batch_ops, 1))
                                batch_progress_container.markdown(f"**Batch Progress:** Testset {ts_idx}/{len(testsets)} | {completed_batch_ops}/{total_batch_ops} operations")
                    
                    consolidated_debug_info.append(f"=== Completed accuracy evaluation ===")
                elif not evaluation_model_name:
                    st.warning("No evaluation model selected. Accuracy checking skipped.")
                elif not has_expected_responses:
                    st.warning("No expected responses found in testset. Accuracy checking skipped.")
                
                # Clear progress indicators
                progress_bar.empty()
                progress_container.empty()
                streaming_section.empty()
                
                # Convert entry_results to the expected testset_results format
                st.session_state.testset_results = [
                    entry_results[prompt_data['idx']] for prompt_data in all_prompts
                ]

                assets = build_testset_report_assets(st.session_state.testset_results)
                csv_df = assets.get('csv_df', pd.DataFrame())
                csv_data = csv_df.to_csv(index=False) if not csv_df.empty else ""
                st.session_state.testset_batch_reports.append({
                    'name': ts_name,
                    'rows': len(testset_rows),
                    'csv': csv_data,
                    'pdf': assets.get('pdf'),
                    'agg_df': assets.get('agg_df')
                })
                st.session_state.testset_batch_debug_logs.append({'name': ts_name, 'logs': consolidated_debug_info})
                st.success(f"Completed {ts_name}")

            # Clear batch progress indicators
            batch_progress_bar.empty()
            batch_progress_container.empty()
        else:
            # Process models with or without spinner based on streaming setting
            if not enable_streaming:
                with st.spinner("Generating responses..."):
                    process_models(selected_models, system_prompt, final_user_prompt)
            else:
                process_models(selected_models, system_prompt, final_user_prompt)

# Display testset batch reports (outside compare_button block so they persist across reruns)
if st.session_state.testset_batch_reports:
    st.success(f"Testset processing complete for {len(st.session_state.testset_batch_reports)} set(s)")
    st.markdown("### Testset Reports")
    
    # Add a button to clear all reports
    if st.button("Clear All Reports", key="clear_batch_reports"):
        st.session_state.testset_batch_reports = []
        st.session_state.testset_batch_debug_logs = []
        st.session_state.testset_results = []
        st.rerun()
    
    for idx, report in enumerate(st.session_state.testset_batch_reports, start=1):
        safe_name = safe_filename(report.get('name')) or f"testset_{idx}"
        st.markdown(f"**{report.get('name', f'Testset {idx}')}** ({report.get('rows', 0)} rows)")
        
        col1, col2 = st.columns(2)
        with col1:
            if report.get('csv'):
                st.download_button(
                    label="Download Results (CSV)",
                    data=report['csv'],
                    file_name=f"{safe_name}_results.csv",
                    mime="text/csv",
                    key=f"csv_{safe_name}_{idx}",
                    use_container_width=True
                )
        with col2:
            if report.get('pdf'):
                pdf_bytes = report['pdf']
                pdf_data = pdf_bytes.encode('latin-1') if isinstance(pdf_bytes, str) else bytes(pdf_bytes)
                st.download_button(
                    label="Download Testset PDF",
                    data=pdf_data,
                    file_name=f"{safe_name}_report.pdf",
                    mime="application/pdf",
                    key=f"pdf_{safe_name}_{idx}",
                    use_container_width=True
                )
        st.divider()

    # Show debug logs in expanders
    if st.session_state.testset_batch_debug_logs:
        for entry in st.session_state.testset_batch_debug_logs:
            with st.expander(f"Debug Information ({entry['name']})"):
                st.code("\n".join(entry['logs']))

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
                    
                    # Create chart using helper
                    fig = create_speed_chart(stats_df)
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
                    
                    fig2 = create_time_chart(stats_df)
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
    
    # Collect ratings for export
    current_ratings = collect_model_ratings()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="Download as CSV",
            data=results_df.to_csv(index=False),
            file_name="ollama_model_comparison.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Prepare JSON data with accuracy ratings
    json_data = {
        "prompt": user_prompt,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "results": [],  # Changing structure slightly to include accuracy per result
        "performance_stats": {
            model: stats.to_dict() 
            for model, stats in st.session_state.performance_stats.items()
        } if st.session_state.performance_stats else {},
        "evaluation": st.session_state.evaluation_result
    }
    
    # Populate results list with individual objects
    for model, response in st.session_state.results.items():
        json_data["results"].append({
            "model": model,
            "response": response,
            "length": len(response),
            "accuracy": current_ratings.get(model, "N/A")
        })

    json_results = json.dumps(json_data, indent=2)
    
    with col2:
        st.download_button(
            label="Download as JSON",
            data=json_results,
            file_name="ollama_model_comparison.json",
            mime="application/json",
            use_container_width=True
        )

    # Generate PDF Report
    try:
        # Generate charts for PDF
        import tempfile
        chart_paths = {}
        
        testset_results = st.session_state.get('testset_results', [])
        is_testset_mode = len(testset_results) > 0
        
        if is_testset_mode:
            assets = build_testset_report_assets(testset_results)
            pdf_bytes = assets.get('pdf')
        else:
            # Speed Chart
            if len(st.session_state.performance_stats) > 0:
                stats_df = create_stats_dataframe(st.session_state.performance_stats)
                if not stats_df.empty:
                    try:
                        speed_fig = create_speed_chart(stats_df)
                        if speed_fig:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                                speed_fig.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                                chart_paths['Speed Comparison'] = tmp.name
                            plt.close(speed_fig)
                            
                        time_fig = create_time_chart(stats_df)
                        if time_fig:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                                time_fig.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                                chart_paths['Time Comparison'] = tmp.name
                            plt.close(time_fig)
                    except Exception as e:
                        st.warning(f"Could not generate performance charts for PDF: {e}")

            # Accuracy Chart
            if current_ratings:
                try:
                    acc_fig = create_accuracy_chart(current_ratings)
                    if acc_fig:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                            acc_fig.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                            chart_paths['Accuracy Ratings'] = tmp.name
                        plt.close(acc_fig)
                except Exception as e:
                    st.warning(f"Could not generate accuracy chart for PDF: {e}")

            pdf_bytes = generate_pdf_report(
                user_prompt,
                system_prompt,
                st.session_state.results,
                st.session_state.performance_stats,
                current_ratings,
                st.session_state.evaluation_result,
                user_prompt_vars,
                chart_paths
            )
        
        # Cleanup temp files
        for path in chart_paths.values():
            try:
                os.remove(path)
            except:
                pass

        # Handle potential string return (legacy behavioral or config)
        if isinstance(pdf_bytes, str):
            pdf_data = pdf_bytes.encode('latin-1')
        else:
            pdf_data = bytes(pdf_bytes)
        
        with col3:
            st.download_button(
                label="Download as PDF",
                data=pdf_data,
                file_name="ollama_model_comparison.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Error generating PDF: {e}")

        # If testset was run, provide aggregated results and export
        testset_results = st.session_state.get('testset_results', [])
        if testset_results:
            st.markdown("### Testset Results Summary")
            agg_df = aggregate_testset_results_to_dataframe(testset_results)
            if not agg_df.empty:
                st.dataframe(agg_df, use_container_width=True)
                csv_data = agg_df.to_csv(index=False)
                st.download_button(
                    label="Download Testset Results as CSV",
                    data=csv_data,
                    file_name="testset_results.csv",
                    mime="text/csv",
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
            st.selectbox(
                "Accuracy",
                options=["Select...", "Accurate", "Somewhat Accurate", "Not Accurate"],
                key=f"rating_side_{sanitize_key(model_name)}"
            )
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
                    st.selectbox(
                        "Accuracy",
                        options=["Select...", "Accurate", "Somewhat Accurate", "Not Accurate"],
                        key=f"rating_side_{sanitize_key(model_name)}"
                    )
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
                        st.selectbox(
                            "Accuracy",
                            options=["Select...", "Accurate", "Somewhat Accurate", "Not Accurate"],
                            key=f"rating_side_{sanitize_key(model_name)}"
                        )
    
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
                st.selectbox(
                    "Accuracy",
                    options=["Select...", "Accurate", "Somewhat Accurate", "Not Accurate"],
                    key=f"rating_stacked_{sanitize_key(model)}"
                )

    st.divider()
    st.subheader("Accuracy Analysis")

    if st.button("Generate Accuracy Chart", key="generate_accuracy_chart"):
        # Collect ratings
        model_ratings = {}
        for model in st.session_state.results.keys():
            # Check both possible keys (using sanitized keys to match the widget keys)
            side_key = f"rating_side_{sanitize_key(model)}"
            stacked_key = f"rating_stacked_{sanitize_key(model)}"
            
            side_rating = st.session_state.get(side_key, "Select...")
            stacked_rating = st.session_state.get(stacked_key, "Select...")
            
            final_rating = "Select..."
            if side_rating != "Select...":
                final_rating = side_rating
            elif stacked_rating != "Select...":
                final_rating = stacked_rating
                
            if final_rating != "Select...":
                model_ratings[model] = final_rating
        
        if not model_ratings:
            st.warning("No ratings provided yet. Please rate the models above.")
        else:
            # Generate chart
            # Mapping: Not Accurate=1, Somewhat=2, Accurate=3
            rating_values = {"Accurate": 3, "Somewhat Accurate": 2, "Not Accurate": 1}
            rating_colors = {"Accurate": "#4CAF50", "Somewhat Accurate": "#FFC107", "Not Accurate": "#F44336"}
            
            chart_data = []
            colors = []
            
            # Sort by model name for consistency
            for model in sorted(model_ratings.keys()):
                rating = model_ratings[model]
                chart_data.append({"Model": model, "Score": rating_values[rating], "Rating": rating})
                colors.append(rating_colors[rating])
            
            if not chart_data:
                st.warning("No valid ratings found.")
            else:
                df_chart = pd.DataFrame(chart_data)
                
                # Import matplotlib just in case (already imported at top but safe to use)
                import matplotlib.pyplot as plt
                import io

                fig3, ax3 = plt.subplots(figsize=(10, 6))
                bars = ax3.bar(df_chart["Model"], df_chart["Score"], color=colors)
                
                ax3.set_title("Model Accuracy Ratings")
                ax3.set_ylabel("Accuracy Level")
                ax3.set_yticks([1, 2, 3])
                ax3.set_yticklabels(["Not Accurate", "Somewhat", "Accurate"])
                ax3.set_ylim(0, 3.5)
                ax3.grid(axis='y', linestyle='--', alpha=0.7)
                
                if len(df_chart) > 3:
                    plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
                st.pyplot(fig3)
                
                # Download button
                buf3 = io.BytesIO()
                fig3.savefig(buf3, format='png', dpi=150, bbox_inches='tight')
                buf3.seek(0)
                
                st.download_button(
                    label="Download Accuracy Chart",
                    data=buf3,
                    file_name="accuracy_chart.png",
                    mime="image/png",
                    use_container_width=True
                )
                plt.close(fig3)