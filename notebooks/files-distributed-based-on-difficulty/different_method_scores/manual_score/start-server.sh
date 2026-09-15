#!/bin/bash

# Hallucination Labeler - Quick Start Script

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║   Hallucination Labeler - Starting Backend Server         ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first:"
    echo "   https://nodejs.org/"
    exit 1
fi

echo "✓ Node.js version: $(node --version)"
echo ""

# Navigate to backend directory
cd frontend-backend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    echo ""
fi

# Start the server
echo "🚀 Starting server..."
echo ""
npm start
