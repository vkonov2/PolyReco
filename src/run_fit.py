from polyreco.io import load_initial_model, load_contours
from polyreco.contour import contour_to_support
from polyreco.solve import solve_cutting_plane
from polyreco.export import export_all

def main():
    poly = load_initial_model("data/InitialModel")
    contours = load_contours(prefix="data/shadow/merge2d-cont", first=0, last=399)
    supports = [contour_to_support(c) for c in contours]

    res = solve_cutting_plane(
        poly, supports,
        tol=1e-5,
        max_added_per_iter=200_000,
        max_outer_iters=50,
        log_to_console=True,
        log_file="highs.log",
        reuse_basis=False,
    )

    print("FINAL:", res.status, res.obj, "iters:", res.outer_iters, "added:", res.total_added_rows)
    export_all("results", poly, res)

if __name__ == "__main__":
    main()
