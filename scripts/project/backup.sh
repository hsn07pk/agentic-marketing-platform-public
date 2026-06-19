#!/bin/bash
# scripts/backup.sh
# Database backup script for Agentic AI Agent Platform

set -e

BACKUP_DIR="data/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/backup_${TIMESTAMP}.sql"

mkdir -p "${BACKUP_DIR}"

echo "Starting database backup..."

if [ -z "${DATABASE_URL}" ]; then
    echo "ERROR: DATABASE_URL not set"
    exit 1
fi

docker-compose exec -T postgres pg_dump -U agentic agentic > "${BACKUP_FILE}"

if [ -f "${BACKUP_FILE}" ]; then
    echo "✅ Backup created: ${BACKUP_FILE}"
    
    gzip "${BACKUP_FILE}"
    echo "✅ Compressed: ${BACKUP_FILE}.gz"
    
    find "${BACKUP_DIR}" -name "backup_*.sql.gz" -mtime +7 -delete
    echo "✅ Cleaned old backups (>7 days)"
    
    echo ""
    echo "Backup completed successfully!"
    ls -lh "${BACKUP_DIR}" | tail -5
else
    echo "❌ Backup failed"
    exit 1
fi