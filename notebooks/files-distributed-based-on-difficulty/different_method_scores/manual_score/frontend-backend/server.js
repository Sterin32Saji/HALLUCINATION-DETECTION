const express = require('express');
const fs = require('fs').promises;
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, '..')));

// Data directory (parent folder where JSONL files are stored)
const DATA_DIR = path.join(__dirname, '..');

// API Routes

// List available JSONL files
app.get('/api/files', async (req, res) => {
    try {
        const files = await fs.readdir(DATA_DIR);
        const jsonlFiles = files.filter(file => 
            file.endsWith('.jsonl') || file.endsWith('.ndjson')
        );
        res.json({ files: jsonlFiles });
    } catch (error) {
        console.error('Error listing files:', error);
        res.status(500).json({ error: 'Failed to list files' });
    }
});

// Load a JSONL file
app.get('/api/load/:filename', async (req, res) => {
    try {
        const filename = req.params.filename;
        console.log(filename);
        
        // Security: prevent directory traversal
        if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
            return res.status(400).json({ error: 'Invalid filename' });
        }

        const filePath = path.join(DATA_DIR, filename);
        console.log(filePath);
        // Check if file exists
        try {
            await fs.access(filePath);
        } catch {
            console.error('File not found:', filePath);
            return res.status(404).json({ error: 'File not found' });
        }

        const content = await fs.readFile(filePath, 'utf-8');
        const lines = content.split(/\r?\n/).filter(line => line.trim());
        
        const records = lines.map((line, index) => {
            try {
                return JSON.parse(line);
            } catch (error) {
                throw new Error(`Invalid JSON on line ${index + 1}`);
            }
        });

        res.json({ 
            filename,
            records,
            count: records.length
        });
    } catch (error) {
        console.error('Error loading file:', error);
        res.status(500).json({ error: error.message || 'Failed to load file' });
    }
});

// Save a JSONL file
app.post('/api/save/:filename', async (req, res) => {
    try {
        const filename = req.params.filename;
        const { records } = req.body;

        // Security: prevent directory traversal
        if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
            return res.status(400).json({ error: 'Invalid filename' });
        }

        if (!Array.isArray(records)) {
            return res.status(400).json({ error: 'Records must be an array' });
        }

        const filePath = path.join(DATA_DIR, filename);
        
        // Create backup before saving
        try {
            await fs.access(filePath);
            const backupPath = filePath + '.backup';
            await fs.copyFile(filePath, backupPath);
        } catch {
            // File doesn't exist yet, no backup needed
        }

        // Write the file
        const content = records.map(record => JSON.stringify(record)).join('\n') + 
                       (records.length ? '\n' : '');
        
        await fs.writeFile(filePath, content, 'utf-8');

        res.json({ 
            success: true,
            filename,
            count: records.length,
            message: 'File saved successfully'
        });
    } catch (error) {
        console.error('Error saving file:', error);
        res.status(500).json({ error: error.message || 'Failed to save file' });
    }
});

// Health check
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Serve the main page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '..', 'frontend.html'));
});

// Start server
app.listen(PORT, () => {
    console.log(`
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   Hallucination Labeler Backend Server                    ║
║                                                            ║
║   Server running at: http://localhost:${PORT}                ║
║                                                            ║
║   Open in browser: http://localhost:${PORT}                 ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    `);
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('SIGTERM received, shutting down gracefully...');
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('\nSIGINT received, shutting down gracefully...');
    process.exit(0);
});
