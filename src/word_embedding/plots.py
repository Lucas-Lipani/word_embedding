def plot_clean_heatmap(matrix, title, filename, cmap, vmin=0, vmax=1):
    import seaborn as sns, matplotlib.pyplot as plt

    matrix_plot = matrix.astype(float).fillna(-1)

    # ── cálculo dinâmico do figsize ─────────────────────
    rows, cols = matrix_plot.shape
    cell_h, cell_w = 0.6, 1.0  # em polegadas
    height = max(1.5, rows * cell_h + 1)
    width = max(3.0, cols * cell_w + 1)
    fig, ax = plt.subplots(figsize=(width, height))

    sns.heatmap(
        matrix_plot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar=True,
        cbar_kws={"shrink": 0.7},
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        linewidths=0.5,
        square=False,  # <-- evita esticar para quadrado
    )

    # substitui “-1.00” por “N/A”
    for text in ax.texts:
        if text.get_text() == "-1.00":
            text.set_text("N/A")

    ax.set_title(title)
    if "×" in title:  # mesmas regras que você já tinha
        ax.set_xlabel("Tamanho da Janela")
        ax.set_ylabel(
            "Tamanho da Janela"
            if "×" not in title
            else "Tamanho da Janela\nSBM DOC-TERM"
        )
    else:
        ax.set_xlabel("Tamanho da Janela")
        ax.set_ylabel("Tamanho da Janela")

    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(filename)
    plt.close()
