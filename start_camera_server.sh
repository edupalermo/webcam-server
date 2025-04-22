#!/bin/bash
cd /home/palermo/webcam-server
source .venv/bin/activate
exec python3 camera_server.py
