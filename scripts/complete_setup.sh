source_environment() {
    print_info "Sourcing environment..."
    source .venv/bin/activate
    print_info "Environment sourced successfully!"
}

download_models() {
    print_info "Downloading models..."
    uv pip install huggingface-hub
    uv run huggingface-cli download Qwen/Qwen3-VL-Embedding-8B --local-dir ./models/Qwen3-VL-Embedding-8B
    print_info "Models downloaded successfully!"
}

main() {
    source_environment
    download_models
    print_info "Complete setup completed successfully!"
}

main