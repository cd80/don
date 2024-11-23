#!/bin/bash
set -e

# Configure logging
exec 1> >(logger -s -t $(basename $0)) 2>&1

# Load environment variables
source ~/.env

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/app/backups"
CHECKPOINT_DIR="/data/checkpoints"
BACKUP_RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-30}
HEALTHCHECK_PORT=${HEALTHCHECK_PORT:-8003}

# Start simple health check server
python3 -m http.server ${HEALTHCHECK_PORT} &
HEALTH_SERVER_PID=$!

# Ensure cleanup on script exit
cleanup() {
    echo "Cleaning up..."
    kill $HEALTH_SERVER_PID || true
    rm -f /tmp/backup_running
}
trap cleanup EXIT

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to upload to S3
upload_to_s3() {
    local file="$1"
    local bucket="$2"
    local retention="$3"
    
    if [ -f "$file" ]; then
        log "Uploading $file to S3..."
        aws s3 cp "$file" "s3://${bucket}/" \
            --storage-class STANDARD_IA \
            --metadata "retention=$retention"
    else
        log "Error: File $file not found"
        return 1
    fi
}

# Function to clean old backups
clean_old_backups() {
    local dir="$1"
    local days="$2"
    
    log "Cleaning backups older than $days days in $dir"
    find "$dir" -type f -mtime +$days -delete
    
    # Clean old backups from S3
    aws s3 ls "s3://${BACKUP_BUCKET}/" | while read -r line; do
        createDate=$(echo "$line" | awk {'print $1" "$2'})
        createDate=$(date -d "$createDate" +%s)
        olderThan=$(date -d "-$days days" +%s)
        if [[ $createDate -lt $olderThan ]]; then
            fileName=$(echo "$line" | awk {'print $4'})
            if [ "$fileName" != "" ]; then
                aws s3 rm "s3://${BACKUP_BUCKET}/$fileName"
            fi
        fi
    done
}

# Create backup lock file
touch /tmp/backup_running

# Backup model checkpoints
log "Starting model checkpoint backup..."
CHECKPOINT_BACKUP="${BACKUP_DIR}/checkpoints_${TIMESTAMP}.tar.gz"
tar -czf "$CHECKPOINT_BACKUP" -C "$CHECKPOINT_DIR" .
upload_to_s3 "$CHECKPOINT_BACKUP" "$BACKUP_BUCKET" "$BACKUP_RETENTION_DAYS"

# Backup database if credentials are provided
if [ ! -z "$DB_HOST" ] && [ ! -z "$DB_NAME" ] && [ ! -z "$DB_USER" ]; then
    log "Starting database backup..."
    DB_BACKUP="${BACKUP_DIR}/db_${TIMESTAMP}.sql.gz"
    PGPASSWORD="$DB_PASSWORD" pg_dump -h "$DB_HOST" -U "$DB_USER" "$DB_NAME" | gzip > "$DB_BACKUP"
    upload_to_s3 "$DB_BACKUP" "$BACKUP_BUCKET" "$BACKUP_RETENTION_DAYS"
fi

# Clean old backups
clean_old_backups "$BACKUP_DIR" "$BACKUP_RETENTION_DAYS"

# Remove backup lock file
rm -f /tmp/backup_running

log "Backup completed successfully"

# Keep the script running to maintain the health check server
while true; do
    sleep 3600
done
