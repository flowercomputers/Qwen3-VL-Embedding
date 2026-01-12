source_environment() {
    echo "Sourcing environment..."
    source .venv/bin/activate
    echo "Environment sourced successfully!"
}

download_models() {
    echo "Downloading models..."
    huggingface-cli download Qwen/Qwen3-VL-Embedding-2B --local-dir ./models/Qwen3-VL-Embedding-2B
    echo "Models downloaded successfully!"
}

install_flash_attn() {
    echo "Installing flash-attn (this may take several minutes)..."
    
    echo "NVCC version: $(nvcc --version)"
    pip install flash-attn --no-build-isolation
}

main() {
    echo "Starting complete setup..."
    install_flash_attn
    download_models
    echo "Complete setup completed successfully!"
}

main