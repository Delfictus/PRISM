FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# CRITICAL: GPU runtime environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set working directory
WORKDIR /prism-ai

# Copy source code
COPY . .

# Build the project
RUN cargo build --release

# Create output directory
RUN mkdir -p /output

# Set Rust environment
ENV RUST_BACKTRACE=1
ENV RUST_LOG=info

# Volume for output
VOLUME ["/output"]

# Default entrypoint
ENTRYPOINT ["/prism-ai/target/release/prism-ai"]
CMD ["--help"]
