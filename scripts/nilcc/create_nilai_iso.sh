#!/bin/bash

# Execute as: ./create_nilai_iso.sh nilai

VM_NAME=$1
ISO_FILE=/opt/nilcc/user_vm_images/$VM_NAME.iso

./nilcc-agent iso create \
  --container caddy \
  --port 80 \
  --hostname nilai.sandbox.nilogy.xyz \ # TODO: change to the actual hostname
  --output $ISO_FILE \
  -f nilai-api/config.yaml=files/nilai-api/config.yaml \
  -f caddy/Caddyfile=files/caddy/Caddyfile \
  -f grafana/grafana.ini=files/grafana/grafana.ini \
  -f grafana/datasource.yml=files/grafana/datasource.yml \
  -f grafana/filesystem.yml=files/grafana/filesystem.yml \
  -f grafana/query-data.json=files/grafana/query-data.json \
  -f grafana/nuc-query-data.json=files/grafana/nuc-query-data.json \
  -f prometheus/prometheus.yml=files/prometheus/prometheus.yml \
  ./nilai-docker-compose-prod.yml
