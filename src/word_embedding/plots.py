import seaborn as sns
import matplotlib.pyplot as plt

def plot_clean_heatmap(matrix, title, filename, cmap, vmin=0, vmax=1):
    matrix_plot = matrix.astype(float).fillna(-1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix_plot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar=True,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        linewidths=0.5,
    )

    for text in ax.texts:
        if text.get_text() == "-1.00":
            text.set_text("N/A")

    ax.set_title(title)
    if title in ("ARI: SBM x Word2Vec", "VI: SBM x Word2Vec", "NMI: SBM x Word2Vec"):
        ax.set_xlabel("Janela Word2Vec")
        ax.set_ylabel("Janela SBM")
    else:
        ax.set_xlabel("Tamanho da Janela")
        ax.set_ylabel("Tamanho da Janela")
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(filename)
    plt.close()
