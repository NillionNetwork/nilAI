#!/bin/bash

# Execute as: sudo ./launch_vm.sh nilai --cpu 16 --memory 48 --gpu --portfwd '80:80;443:443' --fg

#set -x

QEMU_DIR=/opt/nilcc/qemu

MEMORY=8
CPU=2
GPU="false"
PORT_FORWARD=""

function usage() {
    echo "./launch_vm.sh VM_NAME [ARGS]"
    echo "ARGS:"
    echo "  --cpu | -c NUM_CPU	Set the number of CPUs"
    echo "  --memory | -m MEMORY_SIZE Set the memory size in GB"
    echo "  --gpu 			Set to use a GPU"
    echo "  --portfwd PORT_FORWARD	Set port forwarding"
    echo "  --fg			Run in foreground (no daemonize)"
    echo "PORT_FORWARD:"
    echo " HOST_PORT:VM_PORT;HOST_PORT2:VM_PORT2;..."
    echo "Example:"
    echo "./launch_vm.sh my-vm --cpu 8 --memory 16 --gpu --portfwd '2222:22;8080:80;4443:443'"
}

VM_NAME=$1
ISO_FILE=/opt/nilcc/user_vm_images/$VM_NAME.iso
STATE_DISK="/opt/nilcc/user_vm_images/${VM_NAME}-state-disk.raw"
[[ ! -f "$STATE_DISK" ]] && echo "VM $VM_NAME not found, available vms:" && ls /opt/nilcc/user_vm_images && usage && exit 1
[[ ! -f "$ISO_FILE" ]] && echo "ISO for VM $VM_NAME not found, create one using nilcc-agent iso create ..." && ls /opt/nilcc/user_vm_images && usage && exit 1
shift

[[ ! -d "/opt/nilcc/run" ]] && mkdir -p /opt/nilcc/run

DAEMONIZE="-daemonize -monitor unix:/opt/nilcc/run/$VM_NAME-monitor.sock,server,nowait -serial unix:/opt/nilcc/run/$VM_NAME-serial.sock,server,nowait"
GPU_OR_CPU="cpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fg)
	DAEMONIZE="-nographic"
	shift
	;;
    --memory | -m)
      MEMORY="$2"
      shift 2
      ;;
    --cpu | -c)
      CPU="$2"
      shift 2
      ;;
    --gpu)
      GPU="true"
      GPU_OR_CPU="gpu"
      shift
      ;;
    --portfwd)
      PORT_FORWARD="$2"
      shift 2
      ;;
    --help | -h)
	    usage
	    exit 0
	    ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

VM_IMAGE="/opt/nilcc/vm_images/cvm-${GPU_OR_CPU}.qcow2"
VM_IMAGE_HASH_DEV="/opt/nilcc/vm_images/cvm-${GPU_OR_CPU}-verity/verity-hash-dev"
VM_IMAGE_ROOT_HASH="$(cat /opt/nilcc/vm_images/cvm-${GPU_OR_CPU}-verity/root-hash)"
INITRD="/opt/nilcc/initramfs/initramfs.cpio.gz"

# Mount the ISO file to figure out what the docker compose file hash is.
ISO_MOUNT=$(mktemp -d)
mount -o ro,loop "$ISO_FILE" "$ISO_MOUNT"
if [ ! -f "$ISO_MOUNT/docker-compose.yaml" ]; then
  echo "Missing docker-compose.yaml file in ISO"
  umount "$ISO_MOUNT"
  exit 1
fi

DOCKER_COMPOSE_HASH=$(sha256sum "$ISO_MOUNT/docker-compose.yaml" | awk '{{ print $1 }}')
umount "$ISO_MOUNT"

cleanup() {
  rmdir "${ISO_MOUNT}"
}

trap cleanup EXIT SIGINT


DEBUG_ARGS="console=ttyS0 earlyprintk=serial "
BOOT_ARGS="panic=-1 root=/dev/sda2 verity_disk=/dev/sdb verity_roothash=${VM_IMAGE_ROOT_HASH} state_disk=/dev/sdc docker_compose_disk=/dev/sr0 docker_compose_hash=$DOCKER_COMPOSE_HASH"
KERNEL_ARGS="${DEBUG_ARGS}${BOOT_ARGS}"

#Hardware Settings
GPU_ARGS=""
if [[ "$GPU" == "true" ]]; then
	NVIDIA_GPU=01:00.0

	NVIDIA_GPU=$(lspci -d 10de: | awk '/NVIDIA/{print $1}')
	NVIDIA_PASSTHROUGH=$(lspci -n -s $NVIDIA_GPU | awk -F: '{print $4}' | awk '{print $1}')

         #echo 10de $NVIDIA_PASSTHROUGH > /sys/bus/pci/drivers/vfio-pci/new_id
	 GPU_ARGS="-device vfio-pci,host=$NVIDIA_GPU,bus=pci.1"
fi

FWD_ARGS=""
if [[ $PORT_FORWARD != "" ]]; then
	for pair in  $( echo $PORT_FORWARD | tr ";" "\n" ); do
		FWD_ARGS+=","
		HOST=$(echo $pair | cut -d ":" -f1);
		VM=$(echo $pair | cut -d ":" -f2);
	        FWD_ARGS+="hostfwd=tcp::${HOST}-:${VM}"
	done
fi

PORT_FORWARD_ARGS="-netdev user,id=vmnic$FWD_ARGS"

echo $QEMU_DIR/usr/local/bin/qemu-system-x86_64 \
-machine confidential-guest-support=sev0,vmport=off ${DAEMONIZE} \
-object sev-snp-guest,id=sev0,cbitpos=51,reduced-phys-bits=1,kernel-hashes=on \
-enable-kvm -no-reboot \
-cpu EPYC-v4 -machine q35 -smp $CPU,maxcpus=$CPU -m ${MEMORY}G,slots=2,maxmem=512G \
-bios /opt/nilcc/vm_images/ovmf/OVMF.fd \
-kernel /opt/nilcc/vm_images/kernel/${GPU_OR_CPU}-vmlinuz \
-initrd "${INITRD}" \
-append "${KERNEL_ARGS}" \
-drive file=${VM_IMAGE},if=none,id=disk0,format=qcow2 \
-device virtio-scsi-pci,id=scsi0,disable-legacy=on,iommu_platform=true \
-device scsi-hd,drive=disk0 \
-drive file=${VM_IMAGE_HASH_DEV},if=none,id=root-disk,format=raw \
-device virtio-scsi-pci,id=scsi1,disable-legacy=on,iommu_platform=true \
-device scsi-hd,drive=root-disk \
-drive file=${STATE_DISK},if=none,id=state-disk,format=raw \
-device virtio-scsi-pci,id=scsi2,disable-legacy=on,iommu_platform=true \
-device scsi-hd,drive=state-disk \
-drive file=${ISO_FILE},if=none,id=docker-compose-cdrom,readonly=true \
-device virtio-scsi-pci,id=scsi3 \
-device scsi-cd,bus=scsi3.0,drive=docker-compose-cdrom \
-device pcie-root-port,id=pci.1,bus=pcie.0 $GPU_ARGS $PORT_FORWARD_ARGS \
-fw_cfg name=opt/ovmf/X-PciMmio64Mb,string=151072


$QEMU_DIR/usr/local/bin/qemu-system-x86_64 \
-machine confidential-guest-support=sev0,vmport=off ${DAEMONIZE} \
-object sev-snp-guest,id=sev0,cbitpos=51,reduced-phys-bits=1,kernel-hashes=on \
-enable-kvm -no-reboot \
-cpu EPYC-v4 -machine q35 -smp $CPU,maxcpus=$CPU -m ${MEMORY}G,slots=2,maxmem=512G \
-bios /opt/nilcc/vm_images/ovmf/OVMF.fd \
-kernel /opt/nilcc/vm_images/kernel/${GPU_OR_CPU}-vmlinuz \
-initrd "${INITRD}" \
-append "${KERNEL_ARGS}" \
-drive file=${VM_IMAGE},if=none,id=disk0,format=qcow2 \
-device virtio-scsi-pci,id=scsi0,disable-legacy=on,iommu_platform=true \
-device scsi-hd,drive=disk0 \
-drive file=${VM_IMAGE_HASH_DEV},if=none,id=root-disk,format=raw \
-device virtio-scsi-pci,id=scsi1,disable-legacy=on,iommu_platform=true \
-device virtio-net-pci,disable-legacy=on,iommu_platform=true,netdev=vmnic,romfile= $PORT_FORWARD_ARGS \
-device scsi-hd,drive=root-disk \
-drive file=${STATE_DISK},if=none,id=state-disk,format=raw \
-device virtio-scsi-pci,id=scsi2,disable-legacy=on,iommu_platform=true \
-device scsi-hd,drive=state-disk \
-drive file=${ISO_FILE},if=none,id=docker-compose-cdrom,readonly=true \
-device virtio-scsi-pci,id=scsi3 \
-device scsi-cd,bus=scsi3.0,drive=docker-compose-cdrom \
-device pcie-root-port,id=pci.1,bus=pcie.0 $GPU_ARGS \
-fw_cfg name=opt/ovmf/X-PciMmio64Mb,string=151072
