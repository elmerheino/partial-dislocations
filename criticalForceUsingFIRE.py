import argparse
from src.core.depinning import NoiseData

def main():
    parser = argparse.ArgumentParser(description="Run FIRE critical force calculation with NoiseData.")
    parser.add_argument('--N', type=int, required=True, help='System size N (required)')
    parser.add_argument('--L', type=int, required=True, help='System length L (required)')
    parser.add_argument('--d0', type=int, required=True, default=None, help='separation of the partials only relevan if --partial is given')
    parser.add_argument('--cores', type=int, required=True, help='Number of cores (required)')
    parser.add_argument('--folder_name', type=str, required=True, help='Folder name for output (required)')
    parser.add_argument('--seed', type=int, required=True, help='Random seed (required)')
    parser.add_argument('--time', type=float, required=True, help='Simulation time (required)')
    parser.add_argument('--dt', type=float, required=True, help='Timestep (required)')
    parser.add_argument('--points', type=int, required=True, help='Number of points in deltaR (required)')
    parser.add_argument('--rmin', type=float, required=True, help='log10(min deltaR) (required)')
    parser.add_argument('--rmax', type=float, required=True, help='log10(max deltaR) (required)')
    parser.add_argument('--save_folder', type=str, required=True, help='Folder to save results (required)')
    parser.add_argument('--partial', action='store_true', help='Enable partial dislocations')
    
    args = parser.parse_args()

    noise_data = NoiseData(
        N=args.N,
        L=args.L,
        cores=args.cores,
        folder_name=args.folder_name,
        seed=args.seed,
        time=args.time,
        dt=args.dt,
        points=args.points,
        d0=args.d0
    )
    if args.partial:
        noise_data.do_all_steps_partial(args.rmin, args.rmax, args.points, args.save_folder)
    else:
        noise_data.do_all_steps(args.rmin, args.rmax, args.points, args.save_folder)

if __name__ == "__main__":
    main()