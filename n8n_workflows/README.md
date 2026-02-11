# n8n Workflows for Wind Forecast Automation

This folder contains n8n workflow files for automating wind forecast operations.

## 📋 Available Workflows

### 1. **wind_forecast_automation.json**
Complete automation workflow for wind forecasting with alerts and notifications.

**Features:**
- ⏰ Runs every 6 hours automatically
- 🌐 Calls wind forecast API
- 📊 Processes and analyzes forecast data
- 🚨 Triggers alerts for high wind/gale conditions
- 💾 Saves forecasts to PostgreSQL database
- 📧 Sends notifications via Slack, MS Teams, and Email
- 🔗 Webhook endpoint for manual/external triggers

**Alert Triggers:**
- **HIGH:** Gale force winds detected
- **MEDIUM:** >6 hours of high winds (>12 m/s) or rapid changes (>3 m/s/hour)
- **LOW:** Normal conditions

## 🚀 Setup Instructions

### Step 1: Start n8n
```bash
cd c:\Users\ajina\Documents\projects\tso-wind-forecast
docker-compose up -d n8n
```

Access n8n at: **http://localhost:5678**
- Username: `admin`
- Password: `changeme123`

### Step 2: Import Workflow
1. Open n8n web interface
2. Click **"+"** → **"Import from File"**
3. Select `wind_forecast_automation.json`
4. Click **"Import"**

### Step 3: Configure Credentials

#### PostgreSQL Database (Optional)
1. Click on **"Save to PostgreSQL"** node
2. Click **"Create New Credential"**
3. Enter:
   - **Host:** `postgres` (Docker) or `localhost:5432` (Local)
   - **Database:** `windforecast`
   - **User:** `windforecast`
   - **Password:** `windforecast123`

#### Email (Optional)
1. Click on **"Send Email Alert"** node
2. Configure SMTP settings or use environment variables in `.env`:
   ```env
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=your-email@gmail.com
   SMTP_PASSWORD=your-app-password
   ```

#### Slack (Optional)
Add to `.env`:
```env
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

#### MS Teams (Optional)
Add to `.env`:
```env
TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/YOUR/WEBHOOK/URL
```

### Step 4: Start Forecast API
```bash
python api_simple.py
```

The API runs on `http://localhost:8000`

### Step 5: Activate Workflow
1. In n8n, open the imported workflow
2. Click **"Active"** toggle in the top right
3. The workflow will now run every 6 hours automatically

## 🔗 API Endpoints

### Internal (from n8n to your API)
```
POST http://host.docker.internal:8000/forecast
Body: {
  "location": "Schleswig-Holstein",
  "hours_ahead": 24
}
```

### Webhook (external trigger)
```
POST http://localhost:5678/webhook/webhook-forecast
```

## 📊 Database Schema

The workflow saves data to PostgreSQL. Create the table:

```sql
CREATE TABLE IF NOT EXISTS forecasts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    location VARCHAR(100),
    avg_wind_speed DECIMAL(5,2),
    max_wind_speed DECIMAL(5,2),
    min_wind_speed DECIMAL(5,2),
    high_wind_hours INTEGER,
    gale_hours INTEGER,
    alert_level VARCHAR(10)
);

CREATE INDEX idx_forecasts_timestamp ON forecasts(timestamp);
CREATE INDEX idx_forecasts_location ON forecasts(location);
```

## 🎯 Customization Options

### Change Schedule
Edit the **"Every 6 Hours"** node:
- Change interval (hourly, daily, weekly)
- Set specific times (e.g., 6 AM and 6 PM)
- Add cron expressions for complex schedules

### Modify Alert Thresholds
Edit the **"Process Forecast Data"** node JavaScript:
```javascript
const hasHighWindAlert = highWindHours > 6;  // Change threshold
const hasRapidChange = maxChange > 3;        // Change sensitivity
```

### Add More Locations
Duplicate the **"Get Wind Forecast"** node and change location parameter.

### Custom Notifications
Add webhooks to other services (Discord, Telegram, etc.) using HTTP Request nodes.

## 🧪 Testing

### Test Manually
1. Click **"Execute Workflow"** button in n8n
2. Monitor execution in real-time
3. Check each node's output

### Test via Webhook
```bash
curl -X POST http://localhost:5678/webhook/webhook-forecast
```

## 📝 Workflow Logic Flow

```
┌─────────────────┐
│  Every 6 Hours  │──┐
│   (Schedule)    │  │
└─────────────────┘  │
                     ├──> ┌──────────────────┐
┌─────────────────┐  │    │ Get Wind Forecast│
│ Webhook Trigger │──┘    │   (API Call)     │
└─────────────────┘       └─────────┬────────┘
                                    │
                          ┌─────────▼────────────┐
                          │ Process Forecast Data│
                          │   (Calculate Alerts) │
                          └──────────┬───────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
          ┌─────────▼────────┐  ┌───▼────┐  ┌───────▼─────────┐
          │   Alert Needed?  │  │Save DB │  │ Webhook Response│
          │   (High/Medium)  │  └────────┘  └─────────────────┘
          └─────────┬────────┘
                    │ (if yes)
          ┌─────────▼────────────┐
          │ Format Alert Message │
          └─────────┬────────────┘
                    │
      ┌─────────────┼─────────────┐
      │             │             │
┌─────▼─────┐ ┌────▼────┐ ┌──────▼──────┐
│   Slack   │ │  Teams  │ │    Email    │
└───────────┘ └─────────┘ └─────────────┘
```

## 🔒 Security Notes

1. **Change default password** in `docker-compose.yml`
2. **Use environment variables** for sensitive data
3. **Enable HTTPS** for production deployments
4. **Limit webhook access** with authentication tokens
5. **Use Docker secrets** for production credentials

## 📚 Additional Resources

- [n8n Documentation](https://docs.n8n.io/)
- [n8n Community](https://community.n8n.io/)
- [Workflow Templates](https://n8n.io/workflows/)

## 🐛 Troubleshooting

### API Connection Failed
- Ensure API is running: `python api_simple.py`
- Check Docker networking: Use `host.docker.internal` instead of `localhost`
- Verify port 8000 is accessible

### Database Connection Failed
- Start PostgreSQL: `docker-compose up -d postgres`
- Check credentials in `.env` file
- Verify database table exists

### No Alerts Received
- Check `.env` file for webhook URLs
- Test webhooks independently
- Verify alert thresholds are being met

### Workflow Not Triggering
- Check workflow is **Active** (toggle in top right)
- Verify schedule settings
- Check n8n logs: `docker logs tennet-n8n`
