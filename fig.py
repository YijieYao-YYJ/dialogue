import matplotlib.pyplot as plt


ANNOTATE_METHOD = "GRU"
K_LABELS = ["@1", "@3", "@5", "@10"]


GRU_OFFSETS = [(-10, 10), (10, 10), (-10, -12), (10, -12)]


DATA = {
    "dialogue_history": {
        "BM25": {
            "recall":    [0.0727, 0.1967, 0.2966, 0.4884],
            "precision": [0.6891, 0.6214, 0.5623, 0.4630],
        },
        "Doc2Vec": {
            "recall":    [0.0759, 0.2034, 0.3038, 0.4847],
            "precision": [0.7197, 0.6429, 0.5761, 0.4595],
        },
        "RNN": {
            "recall":    [0.0513, 0.1527, 0.2460, 0.4481],
            "precision": [0.4868, 0.4824, 0.4665, 0.4248],
        },
        "GRU": {
            "recall":    [0.0941, 0.2500, 0.3709, 0.5771],
            "precision": [0.8926, 0.7901, 0.7032, 0.5471],
        },
    },

    "dialogue_history_only_user": {
        "BM25": {
            "recall":    [0.0773, 0.2093, 0.3101, 0.4979],
            "precision": [0.7329, 0.6613, 0.5879, 0.4720],
        },
        "Doc2Vec": {
            "recall":    [0.0804, 0.2135, 0.3133, 0.4864],
            "precision": [0.7623, 0.6747, 0.5941, 0.4611],
        },
        "RNN": {
            "recall":    [0.0557, 0.1685, 0.2705, 0.4818],
            "precision": [0.5282, 0.5324, 0.5130, 0.4568],
        },
        "GRU": {
            "recall":    [0.0962, 0.2548, 0.3765, 0.5847],
            "precision": [0.9124, 0.8051, 0.7138, 0.5543],
        },
    },

    "dialogue_state": {
        "BM25": {
            "recall":    [0.0909, 0.2394, 0.3544, 0.5534],
            "precision": [0.8619, 0.7565, 0.6720, 0.5246],
        },
        "Doc2Vec": {
            "recall":    [0.0935, 0.2500, 0.3677, 0.5589],
            "precision": [0.8866, 0.7899, 0.6972, 0.5298],
        },
        "RNN": {
            "recall":    [0.1050, 0.2720, 0.3988, 0.6066],
            "precision": [0.9952, 0.8595, 0.7562, 0.5750],
        },
        "GRU": {
            "recall":    [0.1055, 0.2731, 0.3999, 0.6080],
            "precision": [1.0000, 0.8629, 0.7582, 0.5764],
        },
    },
}


def annotate_gru(ax, xs, ys):
    """只给 GRU 这条线标注 @1/@3/@5/@10。"""
    for i, (x, y) in enumerate(zip(xs, ys)):
        dx, dy = GRU_OFFSETS[i]
        ax.annotate(
            K_LABELS[i],
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=9,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
        )


def plot_pr_curve(rep_data, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))

    for method, v in rep_data.items():
        xs = v["recall"]
        ys = v["precision"]

        ax.plot(xs, ys, marker="o", linewidth=2, label=method)

        if method == ANNOTATE_METHOD:
            annotate_gru(ax, xs, ys)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    plot_pr_curve(
        DATA["dialogue_history"],
        "Precision–Recall Curve (Dialogue History)",
        "results/fig/pr_dialogue_history.pdf",
    )
    plot_pr_curve(
        DATA["dialogue_history_only_user"],
        "Precision–Recall Curve (Dialogue History – Only User)",
        "results/fig/pr_dialogue_history_only_user.pdf",
    )
    plot_pr_curve(
        DATA["dialogue_state"],
        "Precision–Recall Curve (Dialogue State)",
        "results/fig/pr_dialogue_state.pdf",
    )

    print("   生成完成：")
    print("  - pr_dialogue_history.pdf")
    print("  - pr_dialogue_history_only_user.pdf")
    print("  - pr_dialogue_state.pdf")


if __name__ == "__main__":
    main()





