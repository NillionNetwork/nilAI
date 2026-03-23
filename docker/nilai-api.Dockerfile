# Stage 1: Build
FROM rust:1.88-bookworm AS builder

WORKDIR /app

# Copy workspace manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY crates/nilai-domain/Cargo.toml crates/nilai-domain/Cargo.toml
COPY crates/nilai-infra/Cargo.toml crates/nilai-infra/Cargo.toml
COPY crates/nilai-auth/Cargo.toml crates/nilai-auth/Cargo.toml
COPY crates/nilai-discovery/Cargo.toml crates/nilai-discovery/Cargo.toml
COPY crates/nilai-api/Cargo.toml crates/nilai-api/Cargo.toml
COPY crates/nilai-model-daemon/Cargo.toml crates/nilai-model-daemon/Cargo.toml
COPY crates/nilai-lmstudio-announcer/Cargo.toml crates/nilai-lmstudio-announcer/Cargo.toml

# Create dummy src files so cargo can resolve deps
RUN mkdir -p crates/nilai-domain/src && echo "pub fn dummy() {}" > crates/nilai-domain/src/lib.rs && \
    mkdir -p crates/nilai-infra/src && echo "pub fn dummy() {}" > crates/nilai-infra/src/lib.rs && \
    mkdir -p crates/nilai-auth/src && echo "pub fn dummy() {}" > crates/nilai-auth/src/lib.rs && \
    mkdir -p crates/nilai-discovery/src && echo "pub fn dummy() {}" > crates/nilai-discovery/src/lib.rs && \
    mkdir -p crates/nilai-api/src && echo "fn main() {}" > crates/nilai-api/src/main.rs && \
    mkdir -p crates/nilai-model-daemon/src && echo "fn main() {}" > crates/nilai-model-daemon/src/main.rs && \
    mkdir -p crates/nilai-lmstudio-announcer/src && echo "fn main() {}" > crates/nilai-lmstudio-announcer/src/main.rs

# Cache dependency build
RUN cargo build --release --workspace 2>/dev/null || true

# Copy actual source code
COPY crates/ crates/

# Touch src files to invalidate cache for our code only
RUN find crates -name "main.rs" -o -name "lib.rs" | xargs touch

# Build release binaries
RUN cargo build --release \
    --bin nilai-api \
    --bin nilai-model-daemon \
    --bin nilai-lmstudio-announcer

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binaries from builder
COPY --from=builder /app/target/release/nilai-api /usr/local/bin/nilai-api
COPY --from=builder /app/target/release/nilai-model-daemon /usr/local/bin/nilai-model-daemon
COPY --from=builder /app/target/release/nilai-lmstudio-announcer /usr/local/bin/nilai-lmstudio-announcer

# Default to running the API
EXPOSE 8081
CMD ["nilai-api"]
