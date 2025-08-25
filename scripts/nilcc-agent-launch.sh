#! /bin/bash

# Usage: nilcc-agent-cli --url <URL> --api-key <API_KEY> launch [OPTIONS] --entrypoint <ENTRYPOINT> --domain <DOMAIN> --docker-compose <DOCKER_COMPOSE_PATH>

# Options:
#       --id <ID>
#           The id to use for the workload
#   -e, --env-var <ENV_VARS>
#           Add an environment variable to the workload, in the format `<name>=<value>`
#       --dotenv-file <DOTENV>
#           The path to a .env file to add environment variables from
#   -f, --file <FILES>
#           Add a file to the workload, in the format `<file-name>=<value>`
#       --docker-credentials <DOCKER_CREDENTIALS>
#           Add docker credentials, in the format `<server>:<username>:<password>`
#       --entrypoint <ENTRYPOINT>
#           The container entrypoint, in the format `<container-name>:<container-port>`
#       --cpus <CPUS>
#           The number of CPUs to use in the VM [default: 1]
#       --gpus <GPUS>
#           The number of GPUs to use in the VM [default: 0]
#       --memory-mb <MEMORY_MB>
#           The amount of RAM, in MBs [default: 2048]
#       --disk-space <DISK_SPACE_GB>
#           The amount of disk space, in GBs, to use for the VM's state disk [default: 10]
#       --domain <DOMAIN>
#           The domain for the VM
#       --docker-compose <DOCKER_COMPOSE_PATH>
#           The path to the docker compose file to be used
#   -h, --help
#           Print help

MOUNT_POINT=/tmp/mnt
NILCC_AGENT_CLI="docker run -it --rm --env-file ./nilcc.env -v .:$MOUNT_POINT ghcr.io/nillionnetwork/nilcc-agent-cli:latest"
$NILCC_AGENT_CLI \
  launch \
  --cpus 16 \
  --memory-mb 49152 \
  --disk-space 100 \
  --gpus 1 \
  --entrypoint caddy:80 \
  --domain dev.latitude.nilai.sandbox.nilogy.xyz \
  --docker-compose $MOUNT_POINT/docker-compose.yml \
  -f nilai-api/config.yaml=$MOUNT_POINT/files/nilai-api/config.yaml \
  -f testnet/nilai-api/config.yaml=$MOUNT_POINT/files/testnet/nilai-api/config.yaml \
  -f caddy/Caddyfile=$MOUNT_POINT/files/caddy/caddyfile \
  -f grafana/grafana.ini=$MOUNT_POINT/files/grafana/grafana.ini \
  -f grafana/datasource.yml=$MOUNT_POINT/files/grafana/datasource.yml \
  -f grafana/filesystem.yml=$MOUNT_POINT/files/grafana/filesystem.yml \
  -f grafana/query-data.json=$MOUNT_POINT/files/grafana/query-data.json \
  -f grafana/nuc-query-data.json=$MOUNT_POINT/files/grafana/nuc-query-data.json \
  -f grafana/testnet-nuc-query-data.json=$MOUNT_POINT/files/grafana/testnet-nuc-query-data.json \
  -f prometheus/prometheus.yml=$MOUNT_POINT/files/prometheus/prometheus.yml
