# Use default output filename (output.yml)
./scripts/docker-composed.sh --dev

# Specify custom output filename
./scripts/docker-composed.sh --prod -o production.yml

# Complex example with multiple files and custom output
./scripts/docker-composed.sh --dev --prod -f docker-compose.llama-3b-gpu.yml -o final-compose.yml

# Production build with specific output
./scripts/docker-composed.sh --prod -f docker-compose.llama-1b-cpu.yml -o nilai-production.yml
