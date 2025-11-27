import numpy as np
import matplotlib.pyplot as plt

knots = np.linspace(0, 1, 21)          # k_0..k_20
x = np.linspace(0, 1, 2001)

def hat(xin, a, b, c):
    f = (np.maximum(0, xin - knots[a]) - np.maximum(0, xin - knots[b])) / (knots[b] - knots[a])
    g = (np.maximum(0, xin - knots[c]) - np.maximum(0, xin - knots[b])) / (knots[c] - knots[b])
    return f + g

# ----- Even hats -----
even_centres = range(2, len(knots)-2, 2)   # 2,4,...,18
h_even = []
for j in even_centres:
    y = hat(x, j-2, j, j+2)
    h_even.append(y)
    #plt.plot(x, y, alpha=0.6, label=f"even @{j}")

# Sum of even hats
S = np.sum(h_even, axis=0)

# (Optional) normalise S into [0,1] to match knot domain
Smin, Smax = S.min(), S.max()
S_norm = (S - Smin) / (Smax - Smin + 1e-12)

# ----- Odd hats on S(x) -----
odd_centres = range(3, len(knots)-3, 2)    # 3,5,...,17
for j in odd_centres:
    # Choose S or S_norm depending on intent:
    y_odd = hat(S_norm, j-2, j, j+2)
    plt.plot(x, y_odd, linewidth=2, label=f"odd-on-S @{j}")

plt.legend(ncol=2)
plt.xlabel("x")
plt.ylabel("value")
plt.title("Hat(Sum(Hats))")
plt.show()