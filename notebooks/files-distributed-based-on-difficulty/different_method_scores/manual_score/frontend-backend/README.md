# Hallucination Labeler Backend

Backend server for the Hallucination Labeler application that prevents page reloads when editing the frontend.

## Quick Start

1. **Install dependencies:**
   ```bash
   cd frontend-backend
   npm install
   ```

2. **Start the server:**
   ```bash
   npm start
   ```

3. **Open in browser:**
   Navigate to `http://localhost:3000`

## Development Mode

For auto-reloading during development:
```bash
npm run dev
```

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/files` - List available JSONL files
- `GET /api/load/:filename` - Load a JSONL file
- `POST /api/save/:filename` - Save a JSONL file

## Features

- ✅ Automatic file backup before saving
- ✅ Real-time autosave (450ms debounce)
- ✅ No page reload on HTML edits
- ✅ Security: Directory traversal protection
- ✅ CORS enabled for development

## Port Configuration

Default port is 3000. To use a different port:
```bash
PORT=8080 npm start
```

## File Structure

```
frontend-backend/
├── server.js       # Express server
├── package.json    # Dependencies
└── README.md       # This file
```

The server serves files from the parent directory, so all your JSONL files remain in the same location.
