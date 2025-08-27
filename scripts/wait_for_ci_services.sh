#!/bin/bash

# Wait for the services to be ready
API_HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' nilai-api 2>/dev/null)
MODEL_HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' nilai-llama_1b_gpu 2>/dev/null)
NUC_API_HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' nilai-nuc-api 2>/dev/null)
MAX_ATTEMPTS=30
ATTEMPT=1

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo "Waiting for nilai to become healthy... API:[$API_HEALTH_STATUS] MODEL:[$MODEL_HEALTH_STATUS] NUC_API:[$NUC_API_HEALTH_STATUS] (Attempt $ATTEMPT/$MAX_ATTEMPTS)"
    sleep 30
    API_HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' nilai-api 2>/dev/null)
    MODEL_HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' nilai-llama_1b_gpu 2>/dev/null)
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
    echo "Error: nilai-llama_1b_gpu failed to become healthy after $MAX_ATTEMPTS attempts"
    exit 1
fi

echo "NUC_API_HEALTH_STATUS: $NUC_API_HEALTH_STATUS"
if [ "$NUC_API_HEALTH_STATUS" != "healthy" ]; then
    echo "Error: nilai-nuc-api failed to become healthy after $MAX_ATTEMPTS attempts"
    exit 1
fi
