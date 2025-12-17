import argparse
from src.core.depinning import NoiseVsCriticalForce

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

    parser.add_argument('--save_folder', type=str, required=True, help='Folder where simulations results are saved (required)')
    parser.add_argument('--taupoints', type=int, required=True, help='The number of external forces to test.')
    
    args = parser.parse_args()

    noise_data = NoiseVsCriticalForce(
        N=args.N,
        L=args.L,
        cores=args.cores,
        folder_name=f"{args.save_folder}/backups",
        seed=args.seed,
        d0=args.d0,
        tau_points=args.taupoints
    )
    if args.partial:
        noise_data.do_all_steps_partial(args.rmin, args.rmax, args.points, args.save_folder)
    else:
        noise_data.do_all_steps(args.rmin, args.rmax, args.points, args.save_folder)

if __name__ == "__main__":
    main()