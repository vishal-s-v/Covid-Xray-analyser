# -*- coding: utf-8 -*-
"""
Created on Sat May 23 18:21:45 2020

@author: Vishal
"""

from flask import *
import sys

app = Flask(__name__)

@app.route('/')
def upload():
    return render_template("cor.html")

if __name__ == '__main__':
    app.run(debug=False)
