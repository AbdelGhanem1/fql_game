import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import umap

def apply_action_chunking(actions, horizon=5):
    """Applies a sliding window to create action chunks for a single shard."""
    N, d_a = actions.shape
    chunked_actions = np.zeros((N - horizon + 1, horizon * d_a))
    for i in range(horizon):
        chunked_actions[:, i * d_a : (i + 1) * d_a] = actions[i : N - horizon + 1 + i]
    return chunked_actions

def load_and_sample_uniformly(dataset_dir, total_samples=100000, chunk_horizon=None):
    """Loads and proportionally samples from every valid shard."""
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
    valid_files = [f for f in all_files if '-val' not in f]
    
    if not valid_files:
        raise FileNotFoundError(f"No valid training .npz files found in {dataset_dir}")
        
    num_files = len(valid_files)
    samples_per_shard = total_samples // num_files
    
    global_actions = []
    for f in valid_files:
        data = np.load(f)
        actions = data['actions']
        if chunk_horizon is not None:
            actions = apply_action_chunking(actions, horizon=chunk_horizon)
        num_available = len(actions)
        take_n = min(samples_per_shard, num_available)
        indices = np.random.choice(num_available, take_n, replace=False)
        global_actions.append(actions[indices])
        
    return np.concatenate(global_actions, axis=0)

def plot_elegant_manifold(ax, data):
    """Fits UMAP and plots an elegant scatter density."""
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(data)
    
    NAVY_BLUE = '#003f5c'
    
    # Using 'o' for both; rasterized=True keeps the PDF size small and fast-loading
    ax.scatter(embedding[:, 0], embedding[:, 1], 
               color=NAVY_BLUE, marker='o', 
               s=2.5, alpha=0.12, edgecolors='none', rasterized=True)
    
    # Completely strip all axes and spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_puzzle', action='store_true')
    parser.add_argument('--samples', type=int, default=100000)
    args = parser.parse_args()

    puzzle_dir = os.path.expanduser("~/abdelghani_work/datasets/puzzle-4x4-100m")
    antmaze_dir = os.path.expanduser("~/abdelghani_work/datasets/antmaze-giant")
    
    # 1x2 grid with minimal whitespace
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    plt.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.99, bottom=0.01)

    # Process Antmaze
    try:
        antmaze_actions = load_and_sample_uniformly(antmaze_dir, total_samples=args.samples)
        plot_elegant_manifold(ax1, antmaze_actions)
    except Exception as e: print(f"Antmaze error: {e}")

    # Process Puzzle
    try:
        horizon = 5 if args.chunk_puzzle else None
        puzzle_actions = load_and_sample_uniformly(puzzle_dir, total_samples=args.samples, chunk_horizon=horizon)
        plot_elegant_manifold(ax2, puzzle_actions)
    except Exception as e: print(f"Puzzle error: {e}")

    output_filename = "global_manifold_raw.pdf"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=400)
    print(f"Final minimalist manifold saved to {output_filename}")

if __name__ == "__main__":
    main()
