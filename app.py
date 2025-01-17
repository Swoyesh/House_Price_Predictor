from flask import Flask, request, jsonify
import pickle
import numpy as np

with open("model.pkl", "rb") as file:
    model = pickle.load(file)