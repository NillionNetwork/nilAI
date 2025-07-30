# Use default output filename (output.yml)
./scripts/docker-composer.sh --dev

# Specify custom output filename
./scripts/docker-composer.sh --prod -o production.yml

# Complex example with multiple files and custom output
./scripts/docker-composer.sh --dev --prod -f docker-compose.llama-3b-gpu.yml -o final-compose.yml

# Production build with specific output
./scripts/docker-composer.sh --prod -f docker-compose.llama-1b-cpu.yml -o nilai-production.yml
