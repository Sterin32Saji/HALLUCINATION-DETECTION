# Hallucination Labeler - Flutter App

A Flutter desktop application for labeling hallucinations in JSONL datasets. This app provides a native desktop experience with the same functionality as the web version.

## Features

✅ **Native Desktop App**: Runs as a native macOS/Windows/Linux application
✅ **Material Design 3**: Modern, clean white-based UI
✅ **Keyboard Shortcuts**: Fast labeling with H, N, U, and arrow keys
✅ **Auto-save**: Changes saved automatically to the backend server
✅ **Filtering & Search**: Filter by label status and search across records
✅ **Remember Last File**: Automatically reopens your last working file

## Prerequisites

1. **Flutter SDK** (3.0.0 or higher)
   - Download from: https://flutter.dev/docs/get-started/install
   - Add Flutter to your PATH

2. **Backend Server Running**
   - The Express server must be running on `localhost:3000`
   - See backend setup in `../frontend-backend/README.md`

## Setup Instructions

### 1. Install Flutter Dependencies

```bash
cd flutter_app
flutter pub get
```

### 2. Enable Desktop Support

For macOS:
```bash
flutter config --enable-macos-desktop
```

For Windows:
```bash
flutter config --enable-windows-desktop
```

For Linux:
```bash
flutter config --enable-linux-desktop
```

### 3. Start the Backend Server

In a separate terminal:
```bash
cd ../frontend-backend
node server.js
```

Wait for the "Server ready" message.

### 4. Run the Flutter App

```bash
flutter run -d macos    # For macOS
# or
flutter run -d windows  # For Windows
# or
flutter run -d linux    # For Linux
```

The app will launch and connect to `http://localhost:3000`.

## Building for Distribution

### Build macOS App

```bash
flutter build macos --release
```

The app will be in `build/macos/Build/Products/Release/hallucination_labeler.app`

### Build Windows App

```bash
flutter build windows --release
```

The executable will be in `build/windows/runner/Release/`

### Build Linux App

```bash
flutter build linux --release
```

The executable will be in `build/linux/x64/release/bundle/`

## Usage

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `H` | Mark as Hallucinated |
| `N` | Mark as Correct (Not hallucinated) |
| `U` | Remove label (Unlabel) |
| `←` | Previous record |
| `→` | Next record |

### Workflow

1. **Select a File**: Choose a JSONL file from the dropdown
2. **Review Records**: Use the sidebar to browse through records
3. **Label**: Click buttons or use keyboard shortcuts
4. **Auto-save**: Changes are saved automatically after 450ms
5. **Filter**: Use filter chips to show only labeled/unlabeled records
6. **Search**: Type in the search box to find specific records

### UI Layout

```
┌─────────────────────────────────────────────┐
│ Header (File selector, Server status)      │
├──────────┬──────────────────────────────────┤
│          │                                  │
│ Sidebar  │  Detail View                     │
│ - Search │  - Label badge                   │
│ - Filter │  - Action buttons                │
│ - List   │  - Record fields                 │
│          │  - Navigation                    │
│          │                                  │
└──────────┴──────────────────────────────────┘
```

## API Endpoints Used

- `GET /api/health` - Check server status
- `GET /api/files` - List available JSONL files
- `GET /api/load/:filename` - Load a specific file
- `POST /api/save/:filename` - Save changes to file

## Configuration

To change the backend API URL, edit `lib/main.dart`:

```dart
static const String apiBase = 'http://localhost:3000';
```

## Troubleshooting

### "Server offline" message

- Ensure the backend server is running: `cd frontend-backend && node server.js`
- Check that it's running on port 3000
- Verify no firewall is blocking the connection

### App doesn't build

- Run `flutter doctor` to check your Flutter installation
- Ensure desktop support is enabled: `flutter config --enable-<platform>-desktop`
- Try `flutter clean && flutter pub get`

### Changes not saving

- Check the server status indicator in the top-right
- Look for error messages in the status bar
- Check the backend server terminal for error logs

## Development

### Hot Reload

While the app is running, press `r` in the terminal to hot reload changes, or `R` for a full restart.

### Debug Mode

Run in debug mode for better error messages:
```bash
flutter run -d macos --debug
```

### View Logs

The app prints debug information to the console. Check:
- Server connection status
- API call results
- Error messages

## Architecture

The app uses:
- **Material Design 3** for the UI
- **http** package for API calls
- **shared_preferences** for storing last opened file
- **StatefulWidget** for reactive state management
- **KeyboardListener** for keyboard shortcuts

## Differences from Web Version

✅ **Better Performance**: Native rendering
✅ **Offline Awareness**: Clear server status indicator  
✅ **Native Feel**: Platform-specific UI behaviors
✅ **Remembers State**: Reopens last file automatically
✅ **Better Keyboard**: Full keyboard support

## License

Same as the parent project.
