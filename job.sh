#!/bin/sh
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -q all.q@jabba
../venv/bin/python bio_transformer.py
