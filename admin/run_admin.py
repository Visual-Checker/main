#!/usr/bin/env python3
"""Run the standalone admin app and open browser when healthy"""
import os
import time
import threading
import webbrowser
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent

ADMIN_URL = f"http://{os.getenv('ADMIN_HOST','127.0.0.1')}:{os.getenv('ADMIN_PORT','8000')}/admin"
HEALTH_URL = f"http://{os.getenv('ADMIN_HOST','127.0.0.1')}:{os.getenv('ADMIN_PORT','8000')}/health"

print('Starting Admin (standalone)')
print(f'Admin URL: {ADMIN_URL}')

# wait for health then open browser

def open_browser_when_ready(url=ADMIN_URL, health_url=HEALTH_URL, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    print(f"Server healthy, opening browser at {url}")
                    try:
                        webbrowser.open(url)
                    except Exception as e:
                        print('Failed to open browser:', e)
                    return
        except Exception:
            pass
        time.sleep(0.5)
    print(f'Warning: Server did not become healthy within {timeout}s. Open {url} manually')

# Start browser opener
threading.Thread(target=open_browser_when_ready, daemon=True).start()

# Run app
from app import app
if __name__ == '__main__':
    host = os.getenv('ADMIN_HOST', '127.0.0.1')
    port = int(os.getenv('ADMIN_PORT', '8000'))
    debug = os.getenv('DEBUG', 'True').lower() in ('1', 'true', 'yes')
    app.run(host=host, port=port, debug=debug)
