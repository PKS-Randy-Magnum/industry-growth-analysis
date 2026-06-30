# Deployment: GitHub + Docker + Nginx

This app is a **Streamlit** dashboard backed by CSV snapshots in `data/snapshots/`. The server does **not** need BEA/BLS API keys if snapshots are committed or copied with the deploy.

## 1. Push to GitHub

From the project root:

```bash
git init   # if not already a repo
git add .
git commit -m "Industry growth & inflation dashboard with annual SARIMA forecast"
git branch -M main
git remote add origin git@github.com:YOUR_USER/industry-growth-inflation-analysis.git
git push -u origin main
```

**Do not commit** `.env` (already in `.gitignore`). API keys are only for local `python run.py --refresh`.

Optional: add a GitHub Action later to run `pytest` on push (not required for first deploy).

## 2. Target server

Typical layout:

| Piece | Suggestion |
|-------|------------|
| Host | `DEPLOY_HOST` (SSH as your user) |
| App path | `/opt/industry-growth-dashboard` or `~/apps/industry-growth` |
| Process | Docker Compose (recommended) or systemd + venv |
| Public URL | `https://YOUR_DOMAIN` or path on an existing site |
| Reverse proxy | Nginx or Caddy on the host |

Run deploy steps from your machine or on the server after `git clone`.

## 3. Docker deploy (recommended)

On the server:

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-plugin git
sudo usermod -aG docker $USER   # re-login
git clone git@github.com:YOUR_USER/industry-growth-inflation-analysis.git
cd industry-growth-inflation-analysis
docker compose up -d --build
```

App listens on **port 8501**. Verify locally on the server:

```bash
curl -s http://127.0.0.1:8501/_stcore/health
```

## 4. Nginx reverse proxy (example)

`/etc/nginx/sites-available/industry-dashboard`:

```nginx
server {
    listen 443 ssl http2;
    server_name YOUR_DOMAIN;

    # ssl_certificate ... (certbot or existing wildcard)

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

Enable site, `nginx -t`, reload. Point DNS A/AAAA record for `YOUR_DOMAIN` to the server IP.

## 5. Updating after data refresh

On your **dev machine**:

```powershell
python run.py --refresh
git add data/snapshots data/excel
git commit -m "Refresh snapshots through latest quarter"
git push
```

On the **server**:

```bash
cd $APP_DIR   # e.g. /opt/industry-growth-dashboard
git pull
docker compose up -d --build
```

## 6. Alternative: systemd (no Docker)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```

Use a systemd unit with `Restart=always` and the same Nginx proxy.

## 7. Security notes

- Keep Streamlit **behind** Nginx; do not expose 8501 to the public internet without auth if the dashboard is private.
- Optional: Nginx basic auth, Cloudflare Access, or VPN-only access to the host.
- Streamlit has no built-in multi-user auth for portfolio demos.
