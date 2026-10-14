"""RQ1 now evaluates saved pre-grasp point clouds, with no robot or model calls."""
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True, help='New output label; existing results are preserved')
    args = parser.parse_args()
    from analyze_pointcloud_feasibility import analyze
    from plot_pointcloud_feasibility import plot
    analyze(args.label)
    plot(args.label)
