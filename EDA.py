import matplotlib.pyplot as plt

def plot_feature_distributions(df, features, bins=30):
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
    if len(features) == 1:
        axes = [axes]
    for ax, feature in zip(axes, features):
        sns.histplot(df[feature], bins=bins, kde=True, ax=ax)
        ax.set_title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.show()