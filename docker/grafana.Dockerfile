# Use the official Grafana image as a base
FROM grafana/grafana:11.5.1

# Copy the custom configuration files into the image
COPY grafana/datasources/datasource.yml /etc/grafana/provisioning/datasources/datasource.yml
COPY grafana/dashboards/filesystem.yml /etc/grafana/provisioning/dashboards/filesystem.yml
COPY grafana/config/grafana.ini /etc/grafana/grafana.ini
