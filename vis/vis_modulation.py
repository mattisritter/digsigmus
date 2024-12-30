# ======================================
# Digital Signal Processing
# Jakob Kurz (210262)
# Mattis Tom Ritter (210265)
# Heilbronn University of Applied Sciences
# (C) Jakob Kurz, Mattis Tom Ritter 2024
# ======================================
import matplotlib.pyplot as plt
from math import pi

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../module/'))
from function import Function
from add import add
from modulation import modulate, demodulate, quadrature_modulate, quadrature_demodulate

# Define first function
n = range(0, 32)
ws = pi
f1 = list(n)
f2 = [10] * len(n)
function1 = Function(n, ws=ws, f=f1)
function2 = Function(n, ws=ws, f=f2)
# Modulate the function
w_mod1 = 4
function1_mod = modulate(function1, w_mod1)
w_mod2 = 8
function2_mod = modulate(function2, w_mod2)

function1_demod = demodulate(function1_mod, w_mod1)
function2_demod = demodulate(function2_mod, w_mod2)

fig, axs = plt.subplots(2, 2, figsize=(10, 4))
axs[0,0].plot(function1_mod.t, function1_mod.f)
axs[0,0].plot(function2_mod.t, function2_mod.f)
axs[0,0].plot(function1.t, function1.f, color='darkblue')
axs[0,0].plot(function2.t, function2.f, color='darkred')
axs[0,0].set_title('Modulated functions')
axs[0,0].grid()
axs[1,0].plot(function1_demod.t, function1_demod.f)
axs[1,0].plot(function2_demod.t, function2_demod.f)
axs[1,0].plot(function1.t, function1.f, color='darkblue')
axs[1,0].plot(function2.t, function2.f, color='darkred')
axs[1,0].set_title('Demodulated functions')
axs[1,0].grid()
axs[1,0].set_xlabel('n')
# Quadrature modulation
w_mod = 6
function1_mod, function2_mod = quadrature_modulate(function1, function2, w_mod)
function1_demod, _ = quadrature_demodulate(function1_mod, w_mod)
_, function2_demod = quadrature_demodulate(function2_mod, w_mod)
axs[0,1].plot(function1_mod.t, function1_mod.f)
axs[0,1].plot(function2_mod.t, function2_mod.f)
axs[0,1].plot(function1.t, function1.f, color='darkblue')
axs[0,1].plot(function2.t, function2.f, color='darkred')
axs[0,1].set_title('Quadrature Modulated functions')
axs[0,1].grid()
axs[1,1].plot(function1_demod.t, function1_demod.f)
axs[1,1].plot(function2_demod.t, function2_demod.f)
axs[1,1].plot(function1.t, function1.f, color='darkblue')
axs[1,1].plot(function2.t, function2.f, color='darkred')
axs[1,1].set_title('Quadrature Demodulated functions')
axs[1,1].grid()
axs[1,1].set_xlabel('n')
plt.tight_layout()
plt.show()
