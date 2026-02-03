# Admin (PyQt5 Desktop)

This Admin application is a standalone PyQt5 desktop app intended to connect to the Postgres and Redis instances managed by the project's Docker setup.

Quick start
1. Start DB/Redis via server/docker-compose.yml:
   - `cd ../server && sudo docker compose up -d --build`
   - Ensure Postgres port 5432 and Redis port 6379 are accessible (Postgres is exposed in compose)
2. Create a venv and install deps:
   - `python -m venv .venv && source .venv/bin/activate && pip install -r requirements-pyqt.txt`
3. Run the app:
   - `python admin_app.py`

Credentials
- login: `admin` / `admin123` (local testing only)

Config
- Use env vars to configure connections (`.env` file supported):
  - POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
  - REDIS_URL

Notes
- This app intentionally mirrors the functionality of the web-based admin but as a desktop application using PyQt5.
- Next steps: add charts (matplotlib), and live event stream (ZeroMQ/websocket) if desired.

Seeding & DB init
- Create tables and insert sample data: `python seed_data.py`
- Ensure the `server` Postgres and Redis are running (see server/README). 5432 (Postgres) and 6379 (Redis) should be reachable.
