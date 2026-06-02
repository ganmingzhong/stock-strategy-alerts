# Stock Strategy Alerts

Phone-friendly alert config app plus the scheduled Slack alert runner.

## Replit Web App

Use this as the normal Replit run command when you want to edit alert parameters from your phone:

```bash
python alert_web/app.py
```

The app starts on port `5001` by default, so it does not conflict with the analysis website if that project is using port `5000`.
Open the Replit web URL, paste the copied alert JSON from the analysis/database website, tap **Fill Fields**, review, then tap **Save Alert**.

## Replit Scheduled Deployment

Add a Replit Secret named:

```text
SLACK_WEBHOOK_URL
```

Scheduled Deployment build command:

```bash
pip install -r requirements.txt
```

Scheduled Deployment run command:

```bash
python scheduler_alerts/run_strategy_alerts.py
```

The scheduler reads:

```text
scheduler_alerts/strategy_alert_config.json
```

## Normal Workflow

1. Use the analysis website to find a strategy.
2. Copy the alert JSON from the database page.
3. Paste it into this alert web app.
4. Save it.
5. The scheduler uses it on the next scheduled run.
Sending the alert whenever there is entry signal trigger for the stock 
