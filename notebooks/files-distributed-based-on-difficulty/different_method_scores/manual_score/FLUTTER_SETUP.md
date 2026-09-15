# Complete Setup Guide - Flutter + Backend

This guide covers setting up both the backend server and the Flutter desktop app.

## Architecture Overview

```
┌─────────────────────┐         ┌──────────────────────┐
│   Flutter App       │   HTTP  │   Express Server     │
│   (Desktop UI)      │ ◄─────► │   (localhost:3000)   │
│                     │         │                      │
│  - Material Design  │         │  - File operations   │
│  - Keyboard control │         │  - Auto-backup       │
│  - Auto-save        │         │  - JSONL parsing     │
└─────────────────────┘         └──────────────────────┘
                                          │
                                          ▼
                                  ┌───────────────┐
                                  │ JSONL Files   │
                                  │ (Parent dir)  │
                                  └───────────────┘
```

## Prerequisites

### Backend Requirements
- **Node.js** v14+ (check: `node --version`)
- **npm** (comes with Node.js)

### Flutter Requirements
- **Flutter SDK** 3.0.0+ (check: `flutter --version`)
- Platform-specific tools:
  - **macOS**: Xcode Command Line Tools
  - **Windows**: Visual Studio 2022 with C++ tools
  - **Linux**: Required dev libraries (see Flutter docs)

## Installation Steps

### Step 1: Install Node.js (if needed)

**macOS:**
```bash
brew install node
```

**Windows/Linux:**
Download from https://nodejs.org/

### Step 2: Install Flutter (if needed)

1. Download Flutter SDK: https://flutter.dev/docs/get-started/install
2. Extract and add to PATH
3. Run `flutter doctor` to check setup
4. Enable desktop support:
   ```bash
   flutter config --enable-macos-desktop   # or windows/linux
   ```

### Step 3: Setup Backend Server

```bash
cd frontend-backend
npm install
```

This installs:
- express (web server)
- cors (cross-origin requests)

### Step 4: Setup Flutter App

```bash
cd ../flutter_app
flutter pub get
flutter create --platforms=macos .    # or windows/linux
```

This installs:
- http (API calls)
- shared_preferences (local storage)
- flutter_keyboard_visibility (keyboard detection)

## Running the Application

### Method 1: Quick Start (Recommended)

**Terminal 1** - Start backend:
```bash
cd frontend-backend
./start-server.sh
```

**Terminal 2** - Start Flutter:
```bash
cd flutter_app
./start-flutter.sh
```

### Method 2: Manual Start

**Terminal 1** - Backend:
```bash
cd frontend-backend
node server.js
```

Wait for:
```
╔═══════════════════════════════════╗
║  Hallucination Labeler Backend   ║
╚═══════════════════════════════════╝
✓ Server ready on http://localhost:3000
```

**Terminal 2** - Flutter:
```bash
cd flutter_app
flutter run -d macos    # or windows/linux
```

## Using the App

### First Time Setup

1. **Start Backend**: The server must be running first
2. **Launch App**: The Flutter app will auto-detect the server
3. **Select File**: Choose a JSONL file from the dropdown
4. **Start Labeling**: Use keyboard shortcuts for speed

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `H` | Mark as **H**allucinated |
| `N` | Mark as **N**ot hallucinated (Correct) |
| `U` | **U**nlabel (remove label) |
| `←` | Previous record |
| `→` | Next record |

### Workflow Tips

1. **Use Filters**: Click filter chips to focus on unlabeled records
2. **Search**: Type in search box to find specific content
3. **Auto-save**: Changes save automatically after 450ms
4. **Backups**: Original files are backed up automatically
5. **Progress**: Watch the progress bar fill as you label

## Features Comparison

| Feature | Web (HTML/JS) | Flutter Desktop |
|---------|---------------|-----------------|
| Performance | Good | **Excellent** |
| Native Feel | Limited | **Full Native** |
| Keyboard Support | Basic | **Advanced** |
| Remembers Files | No | **Yes** |
| Offline Indicators | Basic | **Clear Status** |
| Hot Reload | No | **Yes (Dev Mode)** |

## Building for Distribution

### macOS App Bundle

```bash
cd flutter_app
flutter build macos --release
open build/macos/Build/Products/Release/
```

Double-click `hallucination_labeler.app` to run.

### Windows Executable

```bash
cd flutter_app
flutter build windows --release
start build\windows\runner\Release\
```

Run `hallucination_labeler.exe`.

### Linux Binary

```bash
cd flutter_app
flutter build linux --release
cd build/linux/x64/release/bundle/
./hallucination_labeler
```

## Troubleshooting

### Backend Issues

**Problem**: "Port 3000 already in use"
```bash
# Find and kill the process
lsof -ti:3000 | xargs kill -9
```

**Problem**: "Cannot find module 'express'"
```bash
cd frontend-backend
rm -rf node_modules package-lock.json
npm install
```

### Flutter Issues

**Problem**: "Flutter not found"
```bash
# Add to PATH (macOS/Linux)
export PATH="$PATH:`pwd`/flutter/bin"

# Or add to ~/.zshrc or ~/.bashrc permanently
```

**Problem**: "No devices detected"
```bash
# Enable desktop support
flutter config --enable-macos-desktop
flutter create --platforms=macos .
```

**Problem**: "Build failed"
```bash
flutter clean
flutter pub get
flutter doctor -v    # Check for issues
```

**Problem**: "Server offline" in app
- Ensure backend is running: `cd frontend-backend && node server.js`
- Check firewall isn't blocking port 3000
- Verify URL in `lib/main.dart` is correct

### Connection Issues

**Problem**: App can't connect to backend

1. Check backend is running (look for "Server ready" message)
2. Verify backend URL in Flutter app matches (default: `http://localhost:3000`)
3. Try accessing http://localhost:3000/api/health in a browser
4. Check firewall/antivirus settings

## Development Mode

### Hot Reload (Flutter)

While the app is running, press:
- `r` - Hot reload (fast, preserves state)
- `R` - Hot restart (full restart)
- `q` - Quit

### Watch Mode (Backend)

For auto-restart on changes:
```bash
npm install -g nodemon
cd frontend-backend
nodemon server.js
```

### Debug Logging

**Flutter**: Check the terminal for console output
**Backend**: All requests logged to terminal

## File Locations

```
manual_score/
├── flutter_app/              # Flutter desktop app
│   ├── lib/
│   │   └── main.dart        # Main app code
│   ├── pubspec.yaml         # Flutter dependencies
│   ├── start-flutter.sh     # Quick start script
│   └── README.md            # Flutter-specific docs
├── frontend-backend/         # Express server
│   ├── server.js            # Server code
│   ├── package.json         # Node dependencies
│   └── start-server.sh      # Quick start script
├── frontend.html            # Old web version (legacy)
├── app.js                   # Old web version (legacy)
└── *.jsonl                  # Your data files
```

## API Endpoints

The Flutter app uses these backend endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Check server status |
| `/api/files` | GET | List available JSONL files |
| `/api/load/:filename` | GET | Load file contents |
| `/api/save/:filename` | POST | Save labeled data |

## Security Notes

⚠️ This is a **local development tool**. Do not expose the backend server to the internet without proper authentication and security measures.

## Next Steps

1. ✅ Backend running on port 3000
2. ✅ Flutter app connected and showing files
3. ✅ Start labeling your dataset
4. 📊 Watch progress bar fill up
5. 🎉 Export or use labeled data

## Support

**Flutter Issues**: https://github.com/flutter/flutter/issues
**Node.js Issues**: https://nodejs.org/en/docs/

---

**Quick Reference:**

```bash
# Start everything
cd frontend-backend && node server.js &
cd flutter_app && flutter run -d macos

# Stop everything
fg    # Bring to foreground
Ctrl+C    # Stop process
```

Enjoy labeling! 🎯
