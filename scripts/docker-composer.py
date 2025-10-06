#!/usr/bin/env python3

"""
This script must be used to compose the final docker-compose.yml file to be used in the ISO creation

It always uses the docker-compose.yml file
If prompted with --dev it will use the docker-compose.dev.yml file
If prompted with --prod it will use the docker-compose.prod.yml file

Additional -f files passed as arguments will be searched in the docker/compose/ directory

It executes docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.prod.yml -f <additional-files> config
"""

import argparse
import os
import subprocess
import sys
import shutil
import yaml
from dotenv import load_dotenv


def load_environment_variables(is_dev):
    """
    Load environment variables from .env.mainnet file.

    Returns:
        dict: Environment variables dictionary with .env.mainnet loaded
    """
    env = os.environ.copy()

    # Load .env.mainnet file if it exists
    env_file = ".env.mainnet" if not is_dev else ".env.dev"
    if os.path.isfile(env_file):
        print(f"Loading environment variables from {env_file}")
        load_dotenv(env_file, override=True)
        # Update the env dict with the loaded variables
        env.update(os.environ)
    else:
        print(f"Warning: {env_file} not found, using system environment variables only")

    return env


def find_docker_compose_command():
    """
    Detect which docker compose command to use - prefer docker compose (v2) over docker-compose (v1).

    Returns:
        str: The docker compose command to use ("docker compose" or "docker-compose").

    Raises:
        SystemExit: If neither 'docker compose' nor 'docker-compose' is available, the program exits with code 1.
    """
    if (
        subprocess.run(
            ["docker", "compose", "version"], capture_output=True, check=False
        ).returncode
        == 0
    ):
        return "docker compose"
    elif shutil.which("docker-compose"):
        return "docker-compose"
    else:
        print("Error: Neither 'docker compose' nor 'docker-compose' are available")
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Compose Docker Compose files for ISO creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dev
  %(prog)s --prod -f docker-compose.llama-3b-gpu.yml
  %(prog)s --dev --prod -f docker-compose.llama-1b-cpu.yml -o final-compose.yml
  %(prog)s --prod -o production.yml
  %(prog)s --testnet --dev
  %(prog)s --testnet --prod
  %(prog)s --testnet --dev --prod
  %(prog)s --dev --image 'nillion/nilai-vllm:latest=public.ecr.aws/k5d9x2g2/nilai-vllm:v0.1.0-rc1'
  %(prog)s --prod --image 'jcabrero/nillion-nilai-api:latest=custom-registry/api:v2.0'
        """,
    )

    parser.add_argument(
        "--dev", action="store_true", help="Include docker-compose.dev.yml"
    )
    parser.add_argument(
        "--prod", action="store_true", help="Include docker-compose.prod.yml"
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Include testnet compose files (.testnet.yml, and .testnet.dev.yml/.testnet.prod.yml if --dev/--prod specified)",
    )
    parser.add_argument(
        "-f",
        "--file",
        action="append",
        dest="additional_files",
        help="Include additional compose file from docker/compose/ directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.yml",
        help="Output filename (default: output.yml)",
    )
    parser.add_argument(
        "--image",
        action="append",
        dest="image_substitutions",
        help="Substitute Docker image (format: old=new, can be used multiple times)",
    )
    parser.add_argument(
        "--no-portable",
        action="store_true",
        help="Keep absolute paths in bind mounts (default: convert to relative)",
    )

    return parser.parse_args()


def validate_image_substitution(substitution):
    """Validate and parse image substitution string"""
    if "=" not in substitution:
        print("Error: --image requires format 'old_image=new_image'")
        sys.exit(1)

    old_image, new_image = substitution.split("=", 1)
    if not old_image or not new_image:
        print("Error: --image requires format 'old_image=new_image'")
        sys.exit(1)

    print(f"Will substitute image: {old_image} -> {new_image}")
    return old_image, new_image


def build_compose_files_list(args):
    """Build list of compose files to include"""
    compose_files = ["-f", "docker-compose.yml"]

    # Add testnet base file if requested
    if args.testnet:
        if os.path.isfile("docker-compose.testnet.yml"):
            compose_files.extend(["-f", "docker-compose.testnet.yml"])
            print("Including docker-compose.testnet.yml")
        else:
            print("Warning: docker-compose.testnet.yml not found")

    # Add dev compose file if requested
    if args.dev:
        if os.path.isfile("docker-compose.dev.yml"):
            compose_files.extend(["-f", "docker-compose.dev.yml"])
            print("Including docker-compose.dev.yml")
        else:
            print("Warning: docker-compose.dev.yml not found")

        # Add testnet dev file if both testnet and dev are requested
        if args.testnet:
            if os.path.isfile("docker-compose.testnet.dev.yml"):
                compose_files.extend(["-f", "docker-compose.testnet.dev.yml"])
                print("Including docker-compose.testnet.dev.yml")
            else:
                print("Warning: docker-compose.testnet.dev.yml not found")

    # Add prod compose file if requested
    if args.prod:
        if os.path.isfile("docker-compose.prod.yml"):
            compose_files.extend(["-f", "docker-compose.prod.yml"])
            print("Including docker-compose.prod.yml")
        else:
            print("Warning: docker-compose.prod.yml not found")

        # Add testnet prod file if both testnet and prod are requested
        if args.testnet:
            if os.path.isfile("docker-compose.testnet.prod.yml"):
                compose_files.extend(["-f", "docker-compose.testnet.prod.yml"])
                print("Including docker-compose.testnet.prod.yml")
            else:
                print("Warning: docker-compose.testnet.prod.yml not found")

    # Add additional compose files
    if args.additional_files:
        for file in args.additional_files:
            if os.path.isfile(file):
                compose_files.extend(["-f", file])
                print(f"Including {file}")
            else:
                print(f"Error: Additional compose file {file} not found")
                sys.exit(1)

    return compose_files


def apply_image_substitutions(output_file, image_substitutions):
    """Apply image substitutions to the output file"""
    if not image_substitutions:
        return

    with open(output_file, "r") as f:
        content = f.read()

    for old_image, new_image in image_substitutions:
        print(f"Applying substitution: {old_image} -> {new_image}")
        content = content.replace(old_image, new_image)

    with open(output_file, "w") as f:
        f.write(content)


def restore_files_variable(output_file, files_placeholder):
    """Replace the FILES placeholder with ${FILES}"""
    print("Restoring ${FILES} variable...")

    with open(output_file, "r") as f:
        content = f.read()

    content = content.replace(files_placeholder, "$FILES")

    with open(output_file, "w") as f:
        f.write(content)

    print("Restored ${FILES} variable")


def process_compose_yaml(output_file, preserve_volumes=False):
    """Process the compose YAML file to remove volumes and convert bind mount formats"""
    if preserve_volumes:
        print(
            "Processing compose YAML for bind mount conversions (preserving volumes)..."
        )
    else:
        print(
            "Processing compose YAML for volume removal and bind mount conversions..."
        )

    with open(output_file, "r") as f:
        content = f.read()

    try:
        # Parse YAML
        compose_data = yaml.safe_load(content)

        # Remove global volumes section entirely (unless preserving volumes)
        if "volumes" in compose_data and not preserve_volumes:
            print("Removing global volumes section...")
            del compose_data["volumes"]

        # Process service volume mounts - remove volume mounts, convert bind mounts
        if "services" in compose_data:
            for service_name, service_config in compose_data["services"].items():
                if "volumes" in service_config and isinstance(
                    service_config["volumes"], list
                ):
                    new_volumes = []
                    for volume in service_config["volumes"]:
                        if isinstance(volume, dict):
                            # Handle long-form definitions
                            if (
                                volume.get("type") == "bind"
                                and "source" in volume
                                and "target" in volume
                            ):
                                # Keep bind mounts, convert to short form
                                source = volume["source"]
                                target = volume["target"]
                                # Check for read_only flag
                                if volume.get("read_only"):
                                    new_volumes.append(f"{source}:{target}:ro")
                                else:
                                    new_volumes.append(f"{source}:{target}")
                            elif volume.get("type") == "volume":
                                if preserve_volumes:
                                    # Keep volume mounts when preserving volumes
                                    new_volumes.append(volume)
                                else:
                                    # Remove volume mounts entirely
                                    print(
                                        f"Removing volume mount from service {service_name}: {volume}"
                                    )
                                    continue
                            else:
                                # Keep other types as-is
                                new_volumes.append(volume)
                        else:
                            # Handle string format volumes
                            volume_str = str(volume)
                            if ":" in volume_str:
                                # Check if it's a bind mount (absolute path) or volume mount (volume name)
                                source_part = volume_str.split(":")[0]
                                if (
                                    source_part.startswith("/")
                                    or source_part.startswith("./")
                                    or source_part.startswith("${")
                                ):
                                    # It's a bind mount (absolute path, relative path, or variable)
                                    new_volumes.append(volume)
                                else:
                                    if preserve_volumes:
                                        # Keep volume mount when preserving volumes
                                        new_volumes.append(volume)
                                    else:
                                        # It's a volume mount (named volume)
                                        print(
                                            f"Removing volume mount from service {service_name}: {volume}"
                                        )
                                        continue
                            else:
                                if preserve_volumes:
                                    # Keep volume mount when preserving volumes
                                    new_volumes.append(volume)
                                else:
                                    # Single name without colon - likely a volume mount
                                    print(
                                        f"Removing volume mount from service {service_name}: {volume}"
                                    )
                                    continue
                    if new_volumes:
                        service_config["volumes"] = new_volumes
                    else:
                        service_config.pop("volumes", None)

        # Write back to file with proper YAML formatting
        with open(output_file, "w") as f:
            yaml.dump(
                compose_data, f, default_flow_style=False, indent=2, sort_keys=False
            )

        print("Completed YAML processing")

    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        print("Falling back to original content")
        # If YAML parsing fails, keep original content
        pass


def make_paths_portable(output_file):
    """Convert absolute paths to relative paths for portability"""
    print("Converting absolute paths to relative paths for portability...")

    current_dir = os.getcwd()

    with open(output_file, "r") as f:
        content = f.read()

    # Convert absolute paths to relative paths in bind mounts
    # Handle different quote styles
    patterns = [
        (f"source: {current_dir}/", "source: ./"),
        (f'source: "{current_dir}/', 'source: "./'),
        (f"source: '{current_dir}/", "source: './"),
    ]

    for old_pattern, new_pattern in patterns:
        content = content.replace(old_pattern, new_pattern)

    with open(output_file, "w") as f:
        f.write(content)

    print("Converted absolute paths to relative paths")


def main():
    """Main function"""
    args = parse_arguments()

    # Find docker compose command
    docker_compose_cmd = find_docker_compose_command()

    # Parse image substitutions
    image_substitutions = []
    if args.image_substitutions:
        for substitution in args.image_substitutions:
            old_image, new_image = validate_image_substitution(substitution)
            image_substitutions.append((old_image, new_image))

    # Build compose files list
    compose_files = build_compose_files_list(args)

    # Build config command with optional flags
    config_cmd = ["config"]
    # Use --no-path-resolution if no_portable=False and docker compose (v2) is available
    use_no_path_resolution = (
        not args.no_portable and docker_compose_cmd == "docker compose"
    )
    if use_no_path_resolution:
        config_cmd.append("--no-path-resolution")

    # Load environment variables from .env.mainnet and set FILES environment variable
    env = load_environment_variables(args.dev)
    files_placeholder = "/tmp/files_placeholder"
    env["FILES"] = files_placeholder

    # Display the command that will be executed
    cmd_parts = docker_compose_cmd.split() + compose_files + config_cmd
    print(f"Executing: {' '.join(cmd_parts)} > {args.output}")

    # Execute docker compose config
    try:
        if not image_substitutions and (args.no_portable or use_no_path_resolution):
            # Direct output but still need to restore FILES variable
            with open(args.output, "w") as f:
                subprocess.run(cmd_parts, stdout=f, env=env, check=True)

            # Restore FILES variable
            restore_files_variable(args.output, files_placeholder)

            # Process YAML for volume and mount conversions
            process_compose_yaml(args.output, preserve_volumes=args.dev)
        else:
            # Generate config and apply modifications
            temp_file = f"{args.output}.tmp"

            with open(temp_file, "w") as f:
                subprocess.run(cmd_parts, stdout=f, env=env, check=True)

            # Copy temp file to output
            shutil.copy(temp_file, args.output)

            # Restore FILES variable first
            restore_files_variable(args.output, files_placeholder)

            # Process YAML for volume and mount conversions
            process_compose_yaml(args.output, preserve_volumes=args.dev)

            # Apply image substitutions
            apply_image_substitutions(args.output, image_substitutions)

            # Make paths portable if requested (default behavior)
            # Skip if no_portable=True, regardless of which docker compose version
            if not args.no_portable and not use_no_path_resolution:
                print("Making paths portable")
                make_paths_portable(args.output)

            # Clean up temporary file
            os.remove(temp_file)

            if image_substitutions or not args.no_portable:
                print(f"Modifications completed. Output written to {args.output}")

    except subprocess.CalledProcessError as e:
        print(f"Error executing docker compose command: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
