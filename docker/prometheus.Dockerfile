FROM prom/prometheus:v3.1.0

COPY prometheus/config/prometheus.yml /etc/prometheus/prometheus.yml
