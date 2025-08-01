#!/bin/bash

# Execute as: sudo ./create_vm.sh nilai

QEMU_DIR=/opt/nilcc/qemu
IMAGES_PATH="/opt/nilcc/user_vm_images"

[[ ! -d "$IMAGES_PATH" ]] && mkdir -p "$IMAGES_PATH"

VM_NAME=$1
[[ "$VM_NAME" == "" ]] && echo "first argument vm name is needed" && exit 1

STATE_DISK="/opt/nilcc/user_vm_images/${VM_NAME}-state-disk.raw"

[[ -f "$STATE_DISK" ]] && echo "VM $VM_NAME already exist" && exit 1

$QEMU_DIR/usr/local/bin/qemu-img create -f raw "$STATE_DISK" 100G

echo "Sucessfully created VM $VM_NAME"
