import requests
import json

def list_ollama_models():
    response = requests.get("http://localhost:11434/api/tags")
    if response.ok:
        return [m["name"] for m in response.json().get("models", [])]
    return []

def get_model_info(model_name, model_file="model_info.json"):

    with open(model_file, "r") as f:
        json_data = json.load(f)
    
    """Extract a specific model by name"""
    model = {}
    models = json_data.get('models', [])
    for m in models:
        if m.get('name') == model_name:
            model = m

    """Get a summary of key information for a specific model"""
    return {
        'name': model.get('name'),
        'size_gb': round(model.get('size', 0) / (1024**3), 2),
        'parameter_size': model.get('details', {}).get('parameter_size'),
        'family': model.get('details', {}).get('family'),
        'quantization': model.get('details', {}).get('quantization_level'),
        'modified_at': model.get('modified_at')
    }