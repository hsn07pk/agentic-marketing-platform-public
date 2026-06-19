#!/bin/bash
# scripts/restore.sh
# Database restore script for Agentic AI Agent Platform

set -e

BACKUP_DIR="data/backups"

if [ -z "$1" ]; then
    echo "Usage: ./scripts/restore.sh <backup_file>"
    echo ""
    echo "Available backups:"
    ls -lh "${BACKUP_DIR}"/*.sql.gz 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_FILE="$1"

if [ ! -f "${BACKUP_FILE}" ]; then
    echo "ERROR: Backup file not found: ${BACKUP_FILE}"
    exit 1
fi

echo "WARNING: This will overwrite the current database!"
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Restore cancelled"
    exit 0
fi

echo "Starting database restore..."

if [[ "${BACKUP_FILE}" == *.gz ]]; then
    echo "Decompressing backup..."
    gunzip -c "${BACKUP_FILE}" | docker-compose exec -T postgres psql -U agentic agentic
else
    docker-compose exec -T postgres psql -U agentic agentic < "${BACKUP_FILE}"
fi

echo "✅ Database restored successfully from ${BACKUP_FILE}"