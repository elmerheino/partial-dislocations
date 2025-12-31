import argparse
from src.core.depinning import NoiseCriticalForceSearch

def main():
    """
    Entry point from terminal to use the NoiseVsCriticalForce class.
    """

    parser = argparse.ArgumentParser(description="Run FIRE critical force calculation with NoiseData.")
    parser.add_argument('--N', type=int, required=True, help='System size N (required) usualy N = L')
    parser.add_argument('--L', type=int, required=True, help='System length L (required) usually L=N')

    parser.add_argument('--partial', action='store_true', help='Give this flag to simulate partial dislocations. Otherwise the script simulates just a single dislocation')
    parser.add_argument('--d0', type=int, required=True, default=None, help='Separation of the partials only relevant if --partial is given')

    parser.add_argument('--cores', type=int, required=True, help='Number of cores (required)')
    parser.add_argument('--seed', type=int, required=True, help='Random seed (required)')

    parser.add_argument('--points', type=int, required=True, help='Number of points in deltaR (required)')
    parser.add_argument('--rmin', type=float, required=True, help='log10(min deltaR) (required)')
    parser.add_argument('--rmax', type=float, required=True, help='log10(max deltaR) (required)')

    parser.add_argument('--taupoints', type=int, default=20, help='The number of external forces to test.')
    parser.add_argument('--taumin', type=float, default=0, help="Multiple of deltaR to use as the lower end of tau search interval. Default 0.")
    parser.add_argument('--taumax', type=float, default=0, help="Multiple of deltaR to use as the upper end of tau search interval. Default 1.3")

    parser.add_argument('--save_folder', type=str, required=True, help='Folder where simulations results are saved (required)')
    
    args = parser.parse_args()

    noise_data = NoiseCriticalForceSearch(
        N=args.N,
        L=args.L,
        cores=args.cores,
        folder_name=f"{args.save_folder}/backups",
        seed=args.seed,
        d0=args.d0
    )
    if args.partial:
        noise_data.run_search_partial(args.rmin, args.rmax, args.points, args.save_folder, args.taumin, args.taumax, args.taupoints)
    else:
        noise_data.run_search_perfect(args.rmin, args.rmax, args.points, args.save_folder, args.taumin, args.taumax, args.taupoints)

if __name__ == "__main__":
    main()