#!/usr/bin/env python3
"""
AI Image Editor - Main Entry Point
Run this script to start the Flask server.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import app, HOST, PORT, DEBUG

if __name__ == '__main__':
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    AI Image Editor                           ║
╠══════════════════════════════════════════════════════════════╣
║  Starting server on http://{HOST}:{PORT}                      ║
║                                                              ║
║  Open http://localhost:{PORT} in your browser                 ║
║                                                              ║
║  Press Ctrl+C to stop the server                             ║
╚══════════════════════════════════════════════════════════════╝
    """)
    app.run(host=HOST, port=PORT, debug=DEBUG)
