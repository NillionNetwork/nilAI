#!/bin/bash

# Wait for the services to be ready
API_HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' nilai-api 2>/dev/null)
MODEL_HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' nilai_gpt_20b_gpu_1 2>/dev/null)
NUC_API_HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' nilai-nuc-api 2>/dev/null)
MAX_ATTEMPTS=30
ATTEMPT=1

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo "Waiting for nilai to become healthy... API:[$API_HEALTH_STATUS] MODEL:[$MODEL_HEALTH_STATUS] NUC_API:[$NUC_API_HEALTH_STATUS] (Attempt $ATTEMPT/$MAX_ATTEMPTS)"
    
    echo "===== Model Container Logs (last 50 lines) ====="
    docker logs --tail 50 nilai_gpt_20b_gpu_1 2>&1
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
    echo "================================================="
    
    sleep 30
    API_HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' nilai-api 2>/dev/null)
    MODEL_HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' nilai_gpt_20b_gpu_1 2>/dev/null)
    NUC_API_HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' nilai-nuc-api 2>/dev/null)
    if [ "$API_HEALTH_STATUS" = "healthy" ] && [ "$MODEL_HEALTH_STATUS" = "healthy" ] && [ "$NUC_API_HEALTH_STATUS" = "healthy" ]; then
        break
    fi

    ATTEMPT=$((ATTEMPT + 1))
done

echo "API_HEALTH_STATUS: $API_HEALTH_STATUS"
if [ "$API_HEALTH_STATUS" != "healthy" ]; then
    echo "Error: nilai-api failed to become healthy after $MAX_ATTEMPTS attempts"
    exit 1
fi

echo "MODEL_HEALTH_STATUS: $MODEL_HEALTH_STATUS"
if [ "$MODEL_HEALTH_STATUS" != "healthy" ]; then
    echo "Error: nilai_gpt_20b_gpu_1 failed to become healthy after $MAX_ATTEMPTS attempts"
    exit 1
fi

echo "NUC_API_HEALTH_STATUS: $NUC_API_HEALTH_STATUS"
if [ "$NUC_API_HEALTH_STATUS" != "healthy" ]; then
    echo "Error: nilai-nuc-api failed to become healthy after $MAX_ATTEMPTS attempts"
    exit 1
fi
