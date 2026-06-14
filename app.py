"""
app.py
SecureLens — Main Flask Application Entry Point
Run this file to start the entire application.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cloud_server.server import create_app

if __name__ == "__main__":
    import os
    
    app = create_app()
    
    # Configuration from environment
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    host = os.getenv("SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("SERVER_PORT", "5000"))
    
    print("\n" + "=" * 55)
    print("  SecureLens Server Starting")
    print("=" * 55)
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Debug: {debug_mode}")
    print(f"  Open browser at: http://{host}:{port}")
    print("  Press Ctrl+C to stop")
    print("=" * 55 + "\n")
    
    app.run(
        host=host,
        port=port,
        debug=debug_mode,
        use_reloader=False   # Prevent double CKKS context init
    )