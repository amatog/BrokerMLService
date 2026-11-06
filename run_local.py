#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from wsgiref.simple_server import make_server
from application import application

if __name__ == "__main__":
    print("Starte lokalen ML-WSGI-Service auf http://127.0.0.1:8000 ...")
    with make_server("127.0.0.1", 8000, application) as server:
        server.serve_forever()
