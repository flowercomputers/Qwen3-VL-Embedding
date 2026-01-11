source_environment() {
    echo "Sourcing environment..."
    source .venv/bin/activate
    echo "Environment sourced successfully!"
}

download_models() {
    echo "Downloading models..."
    uv pip install huggingface-hub
    uv run huggingface-cli download Qwen/Qwen3-VL-Embedding-8B --local-dir ./models/Qwen3-VL-Embedding-8B
    echo "Models downloaded successfully!"
}

install_flash_attn() {
    echo "Installing flash-attn (this may take several minutes)..."
    
    echo "NVCC version: $(nvcc --version)"
    pip install flash-attn --no-build-isolation
}

main() {
    source_environment
    download_models
    echo "Complete setup completed successfully!"
}

main