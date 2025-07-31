#!/bin/bash

# This script must be used to compose the final docker-compose.yml file to be used in the ISO creation

# It always uses the docker-compose.yml file
# If prompted with --dev it will use the docker-compose.dev.yml file
# If prompted with --prod it will use the docker-compose.prod.yml file

# Additional -f files passed as arguments will be searched in the docker/compose/ directory

# It executes docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.prod.yml -f <additional-files> config

set -e

# Detect which docker compose command to use
if command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker-compose"
elif docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    echo "Error: Neither 'docker-compose' nor 'docker compose' is available"
    exit 1
fi

# Initialize variables
USE_DEV=false
USE_PROD=false
ADDITIONAL_FILES=()
COMPOSE_FILES=()
OUTPUT_FILE="output.yml"
IMAGE_SUBSTITUTIONS=()
MAKE_PORTABLE=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            USE_DEV=true
            shift
            ;;
        --prod)
            USE_PROD=true
            shift
            ;;
        -f|--file)
            if [[ -n $2 ]]; then
                ADDITIONAL_FILES+=("$2")
                shift 2
            else
                echo "Error: -f requires a filename argument"
                exit 1
            fi
            ;;
        -o|--output)
            if [[ -n $2 ]]; then
                OUTPUT_FILE="$2"
                shift 2
            else
                echo "Error: -o requires a filename argument"
                exit 1
            fi
            ;;
        --image)
            if [[ -n $2 ]]; then
                # Parse image substitution in format "old_image=new_image"
                if [[ "$2" =~ ^([^=]+)=(.+)$ ]]; then
                    OLD_IMAGE="${BASH_REMATCH[1]}"
                    NEW_IMAGE="${BASH_REMATCH[2]}"
                    IMAGE_SUBSTITUTIONS+=("$OLD_IMAGE|$NEW_IMAGE")
                    echo "Will substitute image: $OLD_IMAGE -> $NEW_IMAGE"
                else
                    echo "Error: --image requires format 'old_image=new_image'"
                    exit 1
                fi
                shift 2
            else
                echo "Error: --image requires an image substitution argument"
                exit 1
            fi
            ;;
        --no-portable)
            MAKE_PORTABLE=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--dev] [--prod] [-f <additional-compose-file>] [-o <output-file>] [--image <old=new>] [--no-portable] ..."
            echo ""
            echo "Options:"
            echo "  --dev                    Include docker-compose.dev.yml"
            echo "  --prod                   Include docker-compose.prod.yml"
            echo "  -f, --file <filename>    Include additional compose file from docker/compose/ directory"
            echo "  -o, --output <filename>  Output filename (default: output.yml)"
            echo "  --image <old=new>        Substitute Docker image (can be used multiple times)"
            echo "  --no-portable            Keep absolute paths in bind mounts (default: convert to relative)"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --dev"
            echo "  $0 --prod -f docker-compose.llama-3b-gpu.yml"
            echo "  $0 --dev --prod -f docker-compose.llama-1b-cpu.yml -o final-compose.yml"
            echo "  $0 --prod -o production.yml"
            echo "  $0 --dev --image 'nillion/nilai-vllm:latest=public.ecr.aws/k5d9x2g2/nilai-vllm:v0.1.0-rc1'"
            echo "  $0 --prod --image 'jcabrero/nillion-nilai-api:latest=custom-registry/api:v2.0'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Always include the base docker-compose.yml
COMPOSE_FILES+=("-f" "docker-compose.yml")

# Add dev compose file if requested
if [[ "$USE_DEV" == true ]]; then
    if [[ -f "docker-compose.dev.yml" ]]; then
        COMPOSE_FILES+=("-f" "docker-compose.dev.yml")
        echo "Including docker-compose.dev.yml"
    else
        echo "Warning: docker-compose.dev.yml not found"
    fi
fi

# Add prod compose file if requested
if [[ "$USE_PROD" == true ]]; then
    if [[ -f "docker-compose.prod.yml" ]]; then
        COMPOSE_FILES+=("-f" "docker-compose.prod.yml")
        echo "Including docker-compose.prod.yml"
    else
        echo "Warning: docker-compose.prod.yml not found"
    fi
fi

# Add additional compose files from docker/compose/ directory
for file in "${ADDITIONAL_FILES[@]}"; do
    # Check if file exists in docker/compose/ directory
    if [[ -f $file ]]; then
        COMPOSE_FILES+=("-f" "$file")
        echo "Including $file"
    else
        echo "Error: Additional compose file $file not found"
        exit 1
    fi
done

# Display the command that will be executed
echo "Executing: $DOCKER_COMPOSE_CMD ${COMPOSE_FILES[*]} config > $OUTPUT_FILE"

# Execute docker compose config with all specified files
if [[ ${#IMAGE_SUBSTITUTIONS[@]} -eq 0 && "$MAKE_PORTABLE" == false ]]; then
    # No image substitutions and no path conversion needed
    $DOCKER_COMPOSE_CMD "${COMPOSE_FILES[@]}" config > "$OUTPUT_FILE"
else
    # Generate config and apply modifications
    $DOCKER_COMPOSE_CMD "${COMPOSE_FILES[@]}" config > "$OUTPUT_FILE.tmp"

    # Copy the temp file to output
    cp "$OUTPUT_FILE.tmp" "$OUTPUT_FILE"

    # Apply image substitutions if any
    if [[ ${#IMAGE_SUBSTITUTIONS[@]} -gt 0 ]]; then
        for substitution in "${IMAGE_SUBSTITUTIONS[@]}"; do
            # Split the substitution string on the pipe character
            old_image="${substitution%|*}"
            new_image="${substitution#*|}"
            echo "Applying substitution: $old_image -> $new_image"
            # Use sed to replace the image in the YAML file
            # Simple substitution - just replace the image name wherever it appears
            sed -i.bak "s|${old_image}|${new_image}|g" "$OUTPUT_FILE"
        done
    fi

    # Make paths portable if requested
    if [[ "$MAKE_PORTABLE" == true ]]; then
        echo "Converting absolute paths to relative paths for portability..."

        # Get the current working directory to use as reference
        CURRENT_DIR=$(pwd)

        # Convert absolute paths to relative paths in bind mounts
        # This handles the "source: /absolute/path" format in YAML
        sed -i.bak2 "s|source: ${CURRENT_DIR}/|source: ./|g" "$OUTPUT_FILE"

        # Also handle cases where the path might be quoted
        sed -i.bak3 "s|source: \"${CURRENT_DIR}/|source: \"./|g" "$OUTPUT_FILE"
        sed -i.bak4 "s|source: '${CURRENT_DIR}/|source: './|g" "$OUTPUT_FILE"

        echo "Converted absolute paths to relative paths"
    fi

    # Clean up temporary files
    rm -f "$OUTPUT_FILE.tmp" "$OUTPUT_FILE.bak" "$OUTPUT_FILE.bak2" "$OUTPUT_FILE.bak3" "$OUTPUT_FILE.bak4"

    if [[ ${#IMAGE_SUBSTITUTIONS[@]} -gt 0 || "$MAKE_PORTABLE" == true ]]; then
        echo "Modifications completed. Output written to $OUTPUT_FILE"
    fi
fi
