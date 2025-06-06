import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_clean_heatmap(matrix, title, filename, cmap, vmin=0, vmax=1):
    matrix_plot = matrix.astype(float).fillna(-1)

    # Configurações do gráfico
    rows, cols = matrix_plot.shape
    fig, ax = plt.subplots(figsize=(max(3.0, cols), max(1.5, rows)))

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
    )

    for text in ax.texts:
        if text.get_text() == "-1.00":
            text.set_text("N/A")

    ax.set_title(title)
    ax.set_xlabel("Tamanho da Janela")
    ax.set_ylabel("Tamanho da Janela")

    plt.tight_layout()
    plt.gca().invert_yaxis()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
