from pathlib import Path

import pandas as pd
import vedo
from brainrender import Scene, settings
from brainrender.actors import Points
from PIL import Image, ImageDraw, ImageFont
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist


SNR_PATH = Path("analysis_tables/snr_proj_location_table.csv")
ENDPOINT_THRESHOLD = 1
N_CLUSTERS = 3
PROJ_COLORS = {
    "orb": "#5fa3c7",
    "rsp_orb": "#d7b227",
    "rsp": "#c95b96",
}
PROJ_LABELS = {
    "orb": "ECL5a ORB",
    "rsp_orb": "ECL5a RSC&ORB",
    "rsp": "ECL5a RSC",
}
CLUSTER_COLORS = {
    1: "#c95b96",
    2: "#d7b227",
    3: "#5fa3c7",
}
CLUSTER_LABELS = {
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
}


def get_endpoint_columns(df: pd.DataFrame) -> list[str]:
    """Return the SNR endpoint columns."""
    return [col for col in df.columns if col.endswith("_endpoint")]


def get_binary_endpoint_matrix(df: pd.DataFrame, threshold: int = ENDPOINT_THRESHOLD) -> tuple[pd.DataFrame, list[str]]:
    """Return the binarized endpoint matrix used for SNR clustering."""
    endpoint_cols = get_endpoint_columns(df)
    return (df[endpoint_cols] >= threshold).astype(int), endpoint_cols


def get_cluster_order(linkage_matrix, cluster_labels: pd.Series) -> list[int]:
    """Order clusters by their first appearance in the dendrogram."""
    leaves = dendrogram(linkage_matrix, no_plot=True)["leaves"]
    return cluster_labels.iloc[leaves].drop_duplicates().tolist()


def add_hierarchical_cluster_label(
    df: pd.DataFrame,
    n_clusters: int = N_CLUSTERS,
    threshold: int = ENDPOINT_THRESHOLD,
    label_col: str = "projection_cluster",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], object, list[int]]:
    """Add the shared hierarchical SNR cluster label."""
    x_binary, endpoint_cols = get_binary_endpoint_matrix(df, threshold=threshold)
    linkage_matrix = linkage(pdist(x_binary.to_numpy(), metric="euclidean"), method="ward")
    raw_labels = pd.Series(fcluster(linkage_matrix, t=n_clusters, criterion="maxclust"), index=df.index)
    cluster_order = get_cluster_order(linkage_matrix, raw_labels)
    relabel = {label: i + 1 for i, label in enumerate(cluster_order)}

    out = df.copy()
    out[label_col] = raw_labels.map(relabel).astype(int)
    cluster_order = list(range(1, n_clusters + 1))
    return out, x_binary, endpoint_cols, linkage_matrix, cluster_order


def load_snr_data(
    n_clusters: int = N_CLUSTERS,
    threshold: int = ENDPOINT_THRESHOLD,
    with_details: bool = False,
) -> pd.DataFrame:
    """Load the prepared SNR table and add the shared cluster label."""
    df = pd.read_csv(SNR_PATH)
    df, x_binary, endpoint_cols, linkage_matrix, cluster_order = add_hierarchical_cluster_label(
        df,
        n_clusters=n_clusters,
        threshold=threshold,
    )
    if with_details:
        return df, x_binary, endpoint_cols, linkage_matrix, cluster_order
    return df


def add_legend(
    image_path: Path,
    order,
    colors: dict,
    legend_labels: dict,
    x0: int = 2480,
    y0: int = 1780,
    dy: int = 150,
) -> None:
    """Add a simple legend to a saved brainrender screenshot."""
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 72)
    except OSError:
        font = ImageFont.load_default()

    r = 24
    for i, label in enumerate(order):
        y = y0 + i * dy
        draw.ellipse((x0, y - r, x0 + 2 * r, y + r), fill=colors[label], outline=None)
        draw.text((x0 + 90, y - 42), legend_labels[label], fill=colors[label], font=font)

    image.save(image_path)


def save_brainrender_plot(
    df: pd.DataFrame,
    group_col: str,
    output_path: Path,
    order,
    colors: dict,
    legend_labels: dict,
) -> Path:
    """Render SNR soma locations in brainrender for one grouping column."""
    settings.OFFSCREEN = True
    settings.SHOW_AXES = False
    settings.SCREENSHOT_SCALE = 2
    vedo.settings.default_backend = "vtk"

    scene = Scene(atlas_name="allen_mouse_25um", title="")
    scene.add_brain_region("root", alpha=0.06, color="lightgrey")

    for label in order:
        coords = df.loc[df[group_col] == label, ["x", "y", "z"]].to_numpy()
        scene.add(Points(coords, radius=55, colors=colors[label], alpha=0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render(interactive=False, camera="frontal", zoom=1.6)
    scene.screenshot(name=str(output_path))
    add_legend(output_path, order=order, colors=colors, legend_labels=legend_labels)
    return output_path
