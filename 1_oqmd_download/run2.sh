#!/bin/bash
while true; do
    python 2.py
    if [ $? -ne 0 ]; then
        echo "Python script failed, waiting for 10 minutes before restart..."
        sleep 600
    else
        echo "Python script finished successfully."
        sleep 300
    fi
done

