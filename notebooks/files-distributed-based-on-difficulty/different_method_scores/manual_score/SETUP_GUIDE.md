# Hallucination Labeler - Setup Guide

## Overview

The Hallucination Labeler now uses a **Node.js backend server** to eliminate the page reload issue when editing the HTML file. Your edits to `frontend.html` will no longer cause the browser to refresh.

## Quick Start

### 1. Start the Backend Server

Open a terminal and run:

```bash
cd /Users/sterinsaji/Desktop/rag-research/notebooks/files-distributed-based-on-difficulty/different_method_scores/manual_score/frontend-backend
node server.js
```

Or use the convenience script:

```bash
cd /Users/sterinsaji/Desktop/rag-research/notebooks/files-distributed-based-on-difficulty/different_method_scores/manual_score
./start-server.sh
```

You should see:
```
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   Hallucination Labeler Backend Server                    ║
║                                                            ║
║   Server running at: http://localhost:3000                ║
║                                                            ║
║   Open in browser: http://localhost:3000                  ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 2. Open the Application

Open your browser and navigate to:
```
http://localhost:3000
```

### 3. Start Labeling

1. Select a JSONL file from the dropdown (it will auto-load available files)
2. Use the keyboard shortcuts or buttons to label records:
   - **H** - Mark as hallucinated
   - **N** - Mark as correct (not hallucinated)
   - **U** - Remove label
   - **← / →** - Navigate between records

## Key Features

✅ **No Page Reloads**: The HTML is now static - all logic is in `app.js`
✅ **Separated Code**: Edit `app.js` for logic changes without affecting the UI
✅ **Autosave**: Changes are automatically saved to the server every 450ms
✅ **Automatic Backups**: Original files are backed up before saving
✅ **Live Server Status**: See connection status in the header
✅ **Download Option**: Export labeled data as JSONL anytime

## File Structure

```
manual_score/
├── frontend.html              # Main UI (static HTML + CSS)
├── app.js                     # Application logic (separated!)
├── start-server.sh           # Quick start script
├── frontend-backend/         # Backend server
│   ├── server.js            # Express server
│   ├── package.json         # Dependencies
│   ├── README.md            # Backend documentation
│   └── node_modules/        # Installed packages
├── balanced_experiment_dataset-test.jsonl
├── balanced_experiment_dataset.jsonl
└── ... (other JSONL files)
```

## API Endpoints

The backend provides these REST endpoints:

- `GET /api/health` - Check server status
- `GET /api/files` - List available JSONL files
- `GET /api/load/:filename` - Load a specific file
- `POST /api/save/:filename` - Save labeled data

## Troubleshooting

### Server won't start
```bash
# Install dependencies first
cd frontend-backend
npm install
```

### Port 3000 already in use
```bash
# Use a different port
PORT=8080 node server.js
```

### Changes not saving
- Check the server status indicator in the header (should be green "● Online")
- Look at the browser console (F12) for error messages
- Ensure you have write permissions to the JSONL files

### Server appears offline
- Restart the backend server
- Click the "↻ Refresh" button to reconnect
- Check that no firewall is blocking localhost:3000

## Development Tips
Static HTML**: The `frontend.html` file is now static and rarely needs editing
2. **Edit Logic**: All application logic is in `app.js` - edit this file to change behavior
3. **No Reloads**: Since the HTML doesn't change during labeling, the browser won't reload
4. **Server auto-restarts**: For development, use `npm run dev` in the `frontend-backend` folder (requires nodemon)
5. **Backup files**: All saves create a `.backup` file automatically

### Why This Works

The JavaScript code is now in a separate `app.js` file. When you label records:
- The browser loads the static HTML once
- The `app.js` handles all the dynamic behavior
- The HTML file never changes, so no reload occurs
- Even if `app.js` changes, modern browsers cache it proper the `frontend-backend` folder (requires nodemon)
3. **Backup files**: All saves create a `.backup` file automatically

## Security Notes

- The server only serves files from the `manual_score` directory
- Directory traversal attacks are prevented
- CORS is enabled for development (consider restricting in production)
- Backups are created before each save operation

## Manual Save/Download

If you prefer manual control:
- Click **💾 Save** to save to the server immediately
- Click **⬇ Download** to download a local copy without affecting the server file

## Stopping the Server

Press `Ctrl+C` in the terminal running the server.

---

**Need help?** Check the logs in the terminal where the server is running.
