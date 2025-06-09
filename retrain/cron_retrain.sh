#!/bin/bash
cd /app
python retrain/retrain.py >> retrain/log.txt 2>&1
