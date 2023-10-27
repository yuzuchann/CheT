#!/bin/bash
while true; do
    python 2.py
    if [ $? -ne 0 ]; then
        echo "Python script failed, waiting for 30 minutes before restart..."
        sleep 1800
    else
        echo "Python script finished successfully."
        break
    fi
done

