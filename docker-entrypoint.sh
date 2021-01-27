#!/bin/bash

# run service
pipenv run uvicorn server.main:app --host=0.0.0.0 --port=80