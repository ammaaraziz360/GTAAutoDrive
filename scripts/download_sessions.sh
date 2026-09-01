#!/bin/bash
# Download session folders from remote server to local machine
# Usage: ./download_sessions.sh [remote_user@remote_host] [remote_path] [local_path]

# Configuration - Edit these defaults or pass as arguments
REMOTE_HOST="${1:-user@your-server.com}"
REMOTE_PATH="${2:-/path/to/remote/data/frames}"
LOCAL_PATH="${3:-./data/frames}"

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_PATH"

echo "============================================"
echo "Downloading sessions from remote server"
echo "Remote: $REMOTE_HOST:$REMOTE_PATH"
echo "Local:  $LOCAL_PATH"
echo "============================================"

# Get list of session folders from remote
echo "Fetching session list..."
SESSIONS=$(ssh "$REMOTE_HOST" "ls -d $REMOTE_PATH/session-*/ 2>/dev/null | xargs -n1 basename")

if [ -z "$SESSIONS" ]; then
    echo "No session folders found at $REMOTE_PATH"
    exit 1
fi

# Count sessions
TOTAL=$(echo "$SESSIONS" | wc -l)
COUNT=0

echo "Found $TOTAL session(s)"
echo ""

# Loop through each session and download
for SESSION in $SESSIONS; do
    COUNT=$((COUNT + 1))
    REMOTE_SESSION="$REMOTE_PATH/$SESSION"
    LOCAL_SESSION="$LOCAL_PATH/$SESSION"
    
    # Check if already downloaded
    if [ -d "$LOCAL_SESSION" ]; then
        LOCAL_COUNT=$(ls -1 "$LOCAL_SESSION"/*.jpg 2>/dev/null | wc -l)
        REMOTE_COUNT=$(ssh "$REMOTE_HOST" "ls -1 $REMOTE_SESSION/*.jpg 2>/dev/null | wc -l")
        
        if [ "$LOCAL_COUNT" -eq "$REMOTE_COUNT" ] && [ "$LOCAL_COUNT" -gt 0 ]; then
            echo "[$COUNT/$TOTAL] $SESSION - Already downloaded ($LOCAL_COUNT frames), skipping"
            continue
        else
            echo "[$COUNT/$TOTAL] $SESSION - Incomplete (local: $LOCAL_COUNT, remote: $REMOTE_COUNT), re-downloading..."
        fi
    else
        echo "[$COUNT/$TOTAL] $SESSION - Downloading..."
    fi
    
    # Download using rsync (resumable, shows progress)
    rsync -avz --progress "$REMOTE_HOST:$REMOTE_SESSION/" "$LOCAL_SESSION/"
    
    if [ $? -eq 0 ]; then
        FRAME_COUNT=$(ls -1 "$LOCAL_SESSION"/*.jpg 2>/dev/null | wc -l)
        echo "[$COUNT/$TOTAL] $SESSION - Done ($FRAME_COUNT frames)"
    else
        echo "[$COUNT/$TOTAL] $SESSION - FAILED"
    fi
    
    echo ""
done

echo "============================================"
echo "Download complete!"
echo "Total sessions: $TOTAL"
echo "============================================"
