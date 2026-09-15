#!/bin/bash

# Quick start script for Flutter Hallucination Labeler

echo "🚀 Starting Hallucination Labeler (Flutter)"
echo "==========================================="

# Check if Flutter is installed
if ! command -v flutter &> /dev/null; then
    echo "❌ Flutter is not installed or not in PATH"
    echo "📥 Download from: https://flutter.dev/docs/get-started/install"
    exit 1
fi

echo "✅ Flutter found: $(flutter --version | head -n 1)"

# Check if we're in the right directory
if [ ! -f "pubspec.yaml" ]; then
    echo "❌ Not in flutter_app directory"
    echo "💡 Run: cd flutter_app"
    exit 1
fi

# Check if dependencies are installed
if [ ! -d ".dart_tool" ]; then
    echo "📦 Installing dependencies..."
    flutter pub get
else
    echo "✅ Dependencies already installed"
fi

# Check if desktop is enabled (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [ ! -d "macos" ]; then
        echo "🔧 Enabling macOS desktop support..."
        flutter config --enable-macos-desktop
        flutter create --platforms=macos .
    fi
    PLATFORM="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ ! -d "linux" ]; then
        echo "🔧 Enabling Linux desktop support..."
        flutter config --enable-linux-desktop
        flutter create --platforms=linux .
    fi
    PLATFORM="linux"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    if [ ! -d "windows" ]; then
        echo "🔧 Enabling Windows desktop support..."
        flutter config --enable-windows-desktop
        flutter create --platforms=windows .
    fi
    PLATFORM="windows"
else
    echo "❌ Unsupported platform: $OSTYPE"
    exit 1
fi

echo ""
echo "⚠️  IMPORTANT: Make sure the backend server is running!"
echo "   Run in another terminal: cd ../frontend-backend && node server.js"
echo ""
echo "🎯 Launching Flutter app on $PLATFORM..."
sleep 2

flutter run -d $PLATFORM
