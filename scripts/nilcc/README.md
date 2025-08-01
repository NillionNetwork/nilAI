# nilAI Production Deployment Guide

This guide provides step-by-step instructions for creating and deploying Nilai production images using the nilCC (Nillion Confidential Computing) infrastructure.

## Overview

The deployment process involves:
1. Creating a production Docker Compose configuration
2. Creating a confidential virtual machine
3. Launching the VM with appropriate resources
4. Configuring NVIDIA compute for confidential computing
5. Managing the CVM agent service

## Prerequisites

- Access to a system with NILCC installed
- NVIDIA GPU with confidential computing support
- Sufficient system resources (16+ CPU cores, 48+ GB RAM recommended for Llama-70B)
- Required permissions (sudo access)

## Step 1: Create Production Docker Compose Configuration

Generate the production-ready Docker Compose file with the specified images:

```bash
bash ./scripts/docker-composer.sh --prod \
  -f docker/compose/docker-compose.llama-70b-gpu.yml \
  --image 'nillion/nilai-api:latest=public.ecr.aws/k5d9x2g2/nilai-api:v0.2.0-alpha0' \
  --image 'nillion/nilai-vllm:latest=public.ecr.aws/k5d9x2g2/nilai-vllm:v0.2.0-alpha0' \
  --image 'nillion/nilai-attestation:latest=public.ecr.aws/k5d9x2g2/nilai-attestation:v0.2.0-alpha0' \
  -o production-compose.yml
```

This command:
- Uses the production configuration (`--prod`)
- Includes the Llama-70B GPU configuration
- Maps local image tags to specific ECR versions
- Outputs to `production-compose.yml`

# On nilCC VM:
## Step 2: Create Virtual Machine

Create a new confidential virtual machine:

```bash
sudo ./create_vm.sh nilai-llama-70b
```

This creates:
- A new VM named `nilai-llama-70b`
- A 100GB raw disk image at `/opt/nilcc/user_vm_images/nilai-llama-70b-state-disk.raw`

## Step 3: Create ISO

Create a new ISO containing the docker compose with the different file mappings:

```bash
bash create_nilai_iso.sh nilai-llama-70b
```

## Step 4: Launch Virtual Machine

Launch the VM with appropriate resource allocation:

```bash
sudo ./launch_vm.sh nilai-llama-70b --cpu 16 --memory 48 --gpu --portfwd '80:80;443:443' --fg
```

Parameters:
- `--cpu 16`: Allocate 16 CPU cores
- `--memory 48`: Allocate 48GB RAM
- `--gpu`: Enable GPU passthrough
- `--portfwd '80:80;443:443'`: Forward HTTP and HTTPS ports
- `--fg`: Run in foreground (for debugging)

### Alternative Launch (Background Mode)

To run in background mode (daemon), omit the `--fg` flag:

```bash
sudo ./launch_vm.sh nilai-llama-70b --cpu 16 --memory 48 --gpu --portfwd '80:80;443:443'
```

## Troubleshooting

After the VM is running, it can happen that NVIDIA confidential compute support is not configured and model files do not launch:

```bash
sudo nvidia-smi conf-compute -srs
```

Then restart the vLLM container to apply the changes:

```bash
docker restart <vllm_container_name>
```

### CVM Agent Service Issues

If you encounter issues with the CVM (Confidential Virtual Machine) agent, use these commands for diagnosis and resolution:

#### Check Service Logs
```bash
journalctl -u cvm-agent.service -e
```

This shows the most recent logs for the CVM agent service, which can help identify:
- Service startup failures
- Configuration issues
- Runtime errors
- Attestation problems

#### Restart CVM Agent Service
```bash
sudo systemctl restart cvm-agent.service
```

Use this if the service becomes unresponsive or after configuration changes.

#### Check Service Status
```bash
sudo systemctl status cvm-agent.service
```

#### Follow Live Logs
```bash
journalctl -u cvm-agent.service -f
```

#### Common Issues and Solutions

1. **Service fails to start**: Check logs for configuration errors or missing dependencies
2. **Attestation failures**: Verify NVIDIA confidential compute configuration
3. **GPU not accessible**: Ensure GPU passthrough is properly configured
4. **Network connectivity issues**: Verify port forwarding and firewall settings

### Additional Diagnostic Commands

#### Check GPU Status
```bash
nvidia-smi
```

#### Verify Confidential Compute Status
```bash
nvidia-smi conf-compute -gs
```

#### Check VM Process
```bash
ps aux | grep qemu
```

#### Monitor System Resources
```bash
htop
```

## Configuration Files

### Docker Compose Configuration
The production configuration uses:
- **API Service**: `public.ecr.aws/k5d9x2g2/nilai-api:v0.2.0-alpha0`
- **vLLM Service**: `public.ecr.aws/k5d9x2g2/nilai-vllm:v0.2.0-alpha0`
- **Attestation Service**: `public.ecr.aws/k5d9x2g2/nilai-attestation:v0.2.0-alpha0`

### Model Configuration
- **Model**: `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`
- **GPU Memory Utilization**: 95%
- **Max Model Length**: 60,000 tokens
- **Tensor Parallel Size**: 1

## Security Considerations

- The VM runs with SEV-SNP (Secure Encrypted Virtualization) for memory encryption
- All communications should use encrypted channels
- Verify attestation reports before processing sensitive data
- Monitor logs for security-related events

## Performance Optimization

- Allocate sufficient CPU cores (16+ recommended for Llama-70B)
- Ensure adequate memory (48GB+ for Llama-70B)
- Use high-performance storage for the state disk
- Monitor GPU utilization and adjust memory allocation as needed

## Support

For additional support:
1. Check the service logs using the troubleshooting commands above
2. Verify system requirements are met
3. Ensure all prerequisites are properly installed
4. Review NVIDIA confidential computing documentation
