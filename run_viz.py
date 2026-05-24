#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   Smart Datacenter AI — Visualisation Interactive               ║
║   Lancer avec :  python run_viz.py                              ║
╚══════════════════════════════════════════════════════════════════╝
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualization.datacenter_viz import main

if __name__ == "__main__":
    main()
