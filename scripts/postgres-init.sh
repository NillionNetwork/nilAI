#!/bin/bash

# PostgreSQL initialization script for Nilai development environment
# This script is automatically executed by PostgreSQL Docker container
# when mounted to /docker-entrypoint-initdb.d/
#
# Creates the additional databases used by docker-compose.dev.yml services:
# - POSTGRES_DB is created automatically by the container
# - POSTGRES_DB_NUC (NUC API database)
# - POSTGRES_DB_TESTNET (Testnet NUC API database)

set -e

echo "Starting Nilai database initialization..."

# Function to create database if it doesn't exist
create_database_if_not_exists() {
    local db_name="$1"
    local db_description="$2"

    if [ -z "$db_name" ]; then
        echo "Warning: Database name is empty for $db_description, skipping..."
        return
    fi

    echo "Checking if database '$db_name' exists..."

    if psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -lqt | cut -d \| -f 1 | grep -qw "$db_name"; then
        echo "Database '$db_name' already exists"
    else
        echo "Creating database '$db_name' for $db_description..."
        psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
            CREATE DATABASE "$db_name";
EOSQL
        echo "Database '$db_name' created successfully"
    fi
}

# Create NUC database
if [ -n "$POSTGRES_DB_NUC" ]; then
    create_database_if_not_exists "$POSTGRES_DB_NUC" "NUC API"
else
    echo "POSTGRES_DB_NUC not set, skipping NUC database creation"
fi

# Create Testnet database
if [ -n "$POSTGRES_DB_TESTNET" ]; then
    create_database_if_not_exists "$POSTGRES_DB_TESTNET" "Testnet NUC API"
else
    echo "POSTGRES_DB_TESTNET not set, skipping Testnet database creation"
fi

echo ""
echo "Database initialization completed!"
echo ""
echo "Available databases:"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c "SELECT datname FROM pg_database WHERE datname NOT IN ('template0', 'template1') ORDER BY datname;"
