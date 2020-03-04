#!/bin/bash

# This script automatically generates the .zip files to be downloaded in the downloads section: scripts, results and all
# Author: Juan Luis Suárez

zip -r downloads/results.zip results
zip -r downloads/scripts.zip scripts
# zip -r downloads/pyDML-Stats.zip .# No longer: download directly from GitHub's link.