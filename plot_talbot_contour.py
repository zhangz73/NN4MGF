import math
import matplotlib.pyplot as plt

singularity = -1

def talbot_1(N, t = 1, alpha = 1, s = 0.7556, m = 0.8597, n = 0.3029):
    if N <= 1:
        raise ValueError("N must be >= 2")
    
    theta_lst = []
    z_lst = []
    e_lst = []

    for k in range(-N, N):
        theta = (2*k+1) * math.pi/(2*N)
        z = N/t * (-s + m * (1 + (2 * theta ** 2) / (theta ** 2 - alpha * math.pi ** 2) + n * 1j * theta))
        theta_lst.append(theta)
        z_lst.append(z)
        e_lst.append(abs(math.e ** z))
    return theta_lst, z_lst, e_lst

def talbot_2(N, t = 1, alpha = 1, s = 0.4814, m = 0.6443, n = 0.5653):
    if N <= 1:
        raise ValueError("N must be >= 2")
    
    theta_lst = []
    z_lst = []
    e_lst = []
    
    def cot(x):
        return math.cos(x) / math.sin(x)
    
    def csc(x):
        return 1 / math.sin(x)

    for k in range(-N, N):
        theta = (2*k+1) * math.pi/(2*N)
        z = N/t * (-s + m * (theta * cot(alpha * theta) + n * 1j * theta))
        theta_lst.append(theta)
        z_lst.append(z)
        e_lst.append(abs(math.e ** z))
    return theta_lst, z_lst, e_lst

def plot_talbot_contours(
    N,
    t,
    method,
    param_sets,
    annotate=True,
    annotate_every=1,
    text_size=7,
    point_size=20,
    line_width=1.5,
    real_axis_width=4.0,
):
    """
    param_sets: list of dicts, each like:
        {"alpha": 1, "s": 0.7556, "m": 0.8597, "n": 0.3029, "label": "optional"}
    """
    fig, ax = plt.subplots()

    # First compute all contours so we can pick sensible axis limits
    contours = []
    reals, imags = [], []
    for ps in param_sets:
        alpha = ps.get("alpha", 1)
        s = ps.get("s", 0.7556)
        m = ps.get("m", 0.8597)
        n = ps.get("n", 0.3029)
        label = f"alpha={alpha}, s={s}, m={m}, n={n}"

        if method == 1:
            _, z_lst, e_lst = talbot_1(N=N, t=t, alpha=alpha, s=s, m=m, n=n)
        else:
            _, z_lst, e_lst = talbot_2(N=N, t=t, alpha=alpha, s=s, m=m, n=n)
        contours.append((label, z_lst, e_lst))

        reals.extend([z.real for z in z_lst])
        imags.extend([z.imag for z in z_lst])

    # Real-axis thick black line "before singularity"
    if reals:
        x_min = min(reals + [singularity]) - 0.1 * (max(reals + [singularity]) - min(reals + [singularity]) + 1e-9)
        ax.plot([x_min, singularity], [0, 0], color="black", linewidth=real_axis_width)

    # Optionally mark the singularity point
    ax.scatter([singularity], [0], marker="x")
    ax.text(singularity, 0, f"  singularity={singularity}", va="center")

    # Plot each contour + annotate points with e-values
    for label, z_lst, e_lst in contours:
        xs = [z.real for z in z_lst]
        ys = [z.imag for z in z_lst]

        ax.plot(xs, ys, linewidth=line_width, label=label)
        ax.scatter(xs, ys, s=point_size)

        if annotate:
            for i, (x, y, ev) in enumerate(zip(xs, ys, e_lst)):
                if annotate_every > 1 and (i % annotate_every != 0):
                    continue
                ax.text(
                    x, y,
                    f"{ev:.0e}",
                    fontsize=text_size,
                    ha="left", va="bottom"
                )

    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.axhline(0, linewidth=0.8)  # real axis (thin reference)
    ax.axvline(0, linewidth=0.8)  # imag axis (thin reference)
    ax.set_aspect("equal" if imags else "auto", adjustable="datalim")
    ax.legend()
    ax.set_title(f"Talbot contours (N={N}, t={t})")
    plt.savefig("talbot_contour.png")
    plt.clf()
    plt.close()

#param_sets = [
#    {"alpha": 5, "s": 0.75, "m": 0.86, "n": 0.3},
#    {"alpha": 8, "s": 0.75,    "m": 0.86,    "n": 0.3},
#]

param_sets = [
    {"alpha": 1, "s": 0.4814, "m": 0.6443, "n": 0.5653},
]

plot_talbot_contours(
    N = 10,
    t = 1,
    method = 2,
    param_sets=param_sets,
    annotate=True,
    annotate_every=1,   # set to 2 or 3 if labels get too crowded
)
