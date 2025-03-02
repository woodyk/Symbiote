#!/usr/bin/env python3
#
# plo-text.py

import plotext as plt

# Generate data
x = range(1, 11)
y = [i**2 for i in x]

# Line Plot
plt.plot(x, y, label="y = x^2")
plt.title("Quadratic Function")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

