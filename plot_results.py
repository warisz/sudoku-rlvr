import matplotlib.pyplot as plt

blanks = range(1, 16)
base    = [148, 104, 58, 35, 20, 16, 15, 9, 5, 10, 11, 12, 14, 29, 57]
trained = [269, 233, 203, 164, 118, 112, 94, 71, 51, 43, 32, 31, 27, 37, 85]
all_bl  = [287, 279, 263, 246, 230, 201, 188, 136, 127, 95, 77, 65, 83, 98, 156]

pct = lambda c: [100 * x / 288 for x in c]

plt.figure(figsize=(9, 5))
plt.plot(blanks, pct(base), "o-", label="base")
plt.plot(blanks, pct(trained), "o-", label="GRPO on 2-4 blanks")
plt.plot(blanks, pct(all_bl), "o-", label="GRPO on all blanks")
plt.xlabel("blank cells")
plt.ylabel("solve rate (%)")
plt.xticks(list(blanks))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("solve_rate.png", dpi=200)