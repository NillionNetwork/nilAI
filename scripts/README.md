# Use default output filename (output.yml)
python3 ./scripts/docker-composer.py --dev

# Specify custom output filename
python3 ./scripts/docker-composer.py --prod -o production.yml

# Complex example with multiple files and custom output
python3 ./scripts/docker-composer.py --dev --prod -f docker-compose.llama-3b-gpu.yml -o final-compose.yml

# Production build with specific output
python3 ./scripts/docker-composer.py --prod -f docker-compose.llama-1b-cpu.yml -o nilai-production.yml
