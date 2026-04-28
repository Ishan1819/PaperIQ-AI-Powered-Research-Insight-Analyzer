# Production Deployment Checklist

## Pre-Deployment Setup

### 1. Environment Variables
Create `.env` file on your EC2 instance:
```bash
SECRET_KEY=<generate-a-long-random-secret-key>
FLASK_ENV=production
DB_HOST=localhost  # or your RDS endpoint
DB_PORT=5432
DB_NAME=paperiq_db
DB_USER=postgres
DB_PASSWORD=<your-secure-password>
```

**Generate a secure SECRET_KEY:**
```python
python -c "import secrets; print(secrets.token_hex(32))"
```

### 2. Install Dependencies
```bash
cd /path/to/research-paper-chatbot
pip install -r requirements.txt
```

### 3. Initialize Database
```bash
# Create database (if not exists)
sudo -u postgres psql
CREATE DATABASE paperiq_db;
CREATE USER paperiq_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE paperiq_db TO paperiq_user;
\q

# Initialize tables
python init_db.py
```

### 4. Test Database Connection
```bash
python -c "from database.db import init_db_pool, test_connection; init_db_pool(); print('Success!' if test_connection() else 'Failed')"
```

## Deployment Options

### Option 1: Gunicorn (Recommended)

#### Install Gunicorn
```bash
pip install gunicorn
```

#### Run with Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Create systemd Service
```bash
sudo nano /etc/systemd/system/paperiq.service
```

```ini
[Unit]
Description=PaperIQ Research Paper Chatbot
After=network.target postgresql.service

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/research-paper-chatbot
Environment="PATH=/home/ubuntu/venv/bin"
EnvironmentFile=/home/ubuntu/research-paper-chatbot/.env
ExecStart=/home/ubuntu/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

#### Start Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable paperiq
sudo systemctl start paperiq
sudo systemctl status paperiq
```

### Option 2: Nginx + Gunicorn (Production)

#### Configure Nginx
```bash
sudo nano /etc/nginx/sites-available/paperiq
```

```nginx
server {
    listen 80;
    server_name your-domain.com;  # or EC2 public IP

    client_max_body_size 16M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /home/ubuntu/research-paper-chatbot/static;
        expires 30d;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/paperiq /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Security Checklist

### ✅ Application Security
- [ ] Set strong `SECRET_KEY` in `.env`
- [ ] Set `FLASK_ENV=production`
- [ ] Verify `SESSION_COOKIE_SECURE=True` (HTTPS only)
- [ ] Database credentials in `.env` (not in code)
- [ ] `.env` file permissions: `chmod 600 .env`

### ✅ Database Security
- [ ] Strong PostgreSQL password
- [ ] Database user has minimal required permissions
- [ ] PostgreSQL listening on localhost only (not 0.0.0.0)
- [ ] Firewall rules restricting database access

### ✅ Server Security
- [ ] AWS Security Group allows only ports 22, 80, 443
- [ ] SSH key authentication (disable password auth)
- [ ] UFW firewall enabled
- [ ] Regular system updates: `sudo apt update && sudo apt upgrade`

### ✅ SSL/HTTPS (Recommended)
- [ ] Install Certbot: `sudo apt install certbot python3-certbot-nginx`
- [ ] Get certificate: `sudo certbot --nginx -d your-domain.com`
- [ ] Auto-renewal enabled: `sudo systemctl status certbot.timer`

## Testing Deployment

### 1. Test Health
```bash
curl http://your-server:5000/check-auth
# Should return: {"authenticated":false,"user":null}
```

### 2. Test Signup
```bash
curl -X POST http://your-server:5000/auth \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123456"}' \
  -v
# Should return 200 with success=true and set a cookie
```

### 3. Test Login
```bash
curl -X POST http://your-server:5000/auth \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123456"}' \
  -c cookies.txt
# Should return 200 with action="login"
```

### 4. Test Protected Endpoint
```bash
curl http://your-server:5000/me -b cookies.txt
# Should return authenticated=true with user data
```

### 5. Browser Test
1. Open browser to `http://your-server:5000`
2. Should redirect to `/login`
3. Enter email and password
4. Should redirect to landing page
5. Upload a document
6. Generate summary
7. Chat with document
8. Click logout - should redirect to login

## Monitoring & Maintenance

### View Logs
```bash
# Application logs
sudo journalctl -u paperiq -f

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-*.log
```

### Database Backup
```bash
# Backup
pg_dump -U paperiq_user paperiq_db > backup_$(date +%Y%m%d).sql

# Restore
psql -U paperiq_user paperiq_db < backup_20260312.sql
```

### Check Service Status
```bash
sudo systemctl status paperiq
sudo systemctl status nginx
sudo systemctl status postgresql
```

### Restart Services
```bash
sudo systemctl restart paperiq
sudo systemctl restart nginx
```

## Common Issues & Solutions

### Issue: Can't connect to database
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection
psql -U paperiq_user -d paperiq_db -h localhost

# Check .env file
cat .env | grep DB_
```

### Issue: 502 Bad Gateway (Nginx)
```bash
# Check Gunicorn is running
sudo systemctl status paperiq

# Check port 5000 is listening
sudo netstat -tlnp | grep 5000

# Restart application
sudo systemctl restart paperiq
```

### Issue: Session not persisting
```bash
# Check SECRET_KEY is set
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OK' if os.getenv('SECRET_KEY') else 'MISSING')"

# Check cookies in browser (Developer Tools > Application > Cookies)
```

### Issue: Database connection pool exhausted
```python
# In database/db.py, increase pool size:
init_db_pool(minconn=5, maxconn=20)
```

## Performance Optimization

### 1. Gunicorn Workers
```bash
# Rule of thumb: (2 x CPU cores) + 1
# For EC2 t2.micro (1 vCPU): 3 workers
# For EC2 t2.small (1 vCPU): 3 workers
# For EC2 t2.medium (2 vCPU): 5 workers

gunicorn -w 3 -b 0.0.0.0:5000 app:app
```

### 2. Database Connection Pool
```python
# In app.py
init_db_pool(minconn=2, maxconn=10)  # Adjust based on traffic
```

### 3. Nginx Caching
```nginx
# Add to nginx config
location /static {
    alias /path/to/static;
    expires 30d;
    add_header Cache-Control "public, immutable";
}
```

## Scaling Considerations

### Vertical Scaling (Easier)
- Upgrade EC2 instance type
- Increase Gunicorn workers
- Increase database connection pool

### Horizontal Scaling (Advanced)
- Use AWS RDS for PostgreSQL
- Multiple EC2 instances behind load balancer
- Session store in Redis/Memcached
- Static files on S3 + CloudFront

## Backup Strategy

### Daily Automated Backup
```bash
# Create backup script
nano ~/backup.sh
```

```bash
#!/bin/bash
BACKUP_DIR="/home/ubuntu/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Database backup
pg_dump -U paperiq_user paperiq_db > $BACKUP_DIR/db_$DATE.sql

# Keep only last 7 days
find $BACKUP_DIR -name "db_*.sql" -mtime +7 -delete

echo "Backup completed: $DATE"
```

```bash
chmod +x ~/backup.sh

# Add to crontab (daily at 2 AM)
crontab -e
0 2 * * * /home/ubuntu/backup.sh >> /home/ubuntu/backup.log 2>&1
```

## Health Check Endpoint (Optional)

Add to app.py:
```python
@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Test database connection
        from database.db import test_connection
        db_healthy = test_connection()
        
        return jsonify({
            'status': 'healthy' if db_healthy else 'unhealthy',
            'database': 'connected' if db_healthy else 'disconnected',
            'timestamp': datetime.now().isoformat()
        }), 200 if db_healthy else 503
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500
```

## Final Checklist

✅ Environment variables configured
✅ Database initialized with tables
✅ Gunicorn service running
✅ Nginx configured (if using)
✅ SSL certificate installed (if using HTTPS)
✅ Firewall rules configured
✅ Application accessible from browser
✅ Login/signup working
✅ Document upload working
✅ Chat functionality working
✅ Chat history saving to database
✅ Logout working
✅ Logs being monitored
✅ Backup script configured

## Support

If you encounter issues:
1. Check logs: `sudo journalctl -u paperiq -n 50`
2. Test database: `python init_db.py`
3. Test authentication: `python test_auth.py`
4. Review error messages in browser console (F12)

Your Flask application with authentication is now production-ready! 🚀
