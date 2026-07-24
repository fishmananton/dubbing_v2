to start: 
prefect server start &
sleep 5
python api/serve.py &

uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

export PREFECT_API_URL=http://127.0.0.1:4200/api

when i change code don't forget to redeploy app and restart prefect


in future convert everythin go SystemD 
sudo nano /etc/systemd/system/prefect-server.service
[Unit]
Description=Prefect Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/dubbed
Environment=PREFECT_HOME=/opt/dubbed/.prefect
ExecStart=/opt/dubbed/.venv/bin/prefect server start
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target

sudo nano /etc/systemd/system/dubbing-serve.service
[Unit]
Description=Dubbing Prefect Serve
After=network.target prefect-server.service
Requires=prefect-server.service

[Service]
Type=simple
WorkingDirectory=/opt/dubbed
Environment=PREFECT_API_URL=http://127.0.0.1:4200/api
Environment=PYTHONPATH=/opt/dubbed
ExecStart=/opt/dubbed/.venv/bin/python /opt/dubbed/api/serve.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target

sudo nano /etc/systemd/system/dubbing-api.service
[Unit]
Description=Dubbing FastAPI
After=network.target prefect-server.service
Requires=prefect-server.service

[Service]
Type=simple
WorkingDirectory=/opt/dubbed
Environment=PREFECT_API_URL=http://127.0.0.1:4200/api
Environment=PYTHONPATH=/opt/dubbed
ExecStart=/opt/dubbed/.venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target