from pathlib import Path

import numpy as np
import pandas as pd
import vedo
from brainrender import Scene, settings
from brainrender.actors import Points
from PIL import Image, ImageChops, ImageDraw, ImageFont
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler


SNR_PATH = Path("analysis_tables/snr_proj_location_table.csv")
ENDPOINT_THRESHOLD = 1
N_CLUSTERS = 3
CLUSTER_TRANSFORM = "binary"
DEFAULT_CLUSTER_COLUMN = "projection_cluster_binary"
EXCLUDED_CLUSTER_COMMENTS = {"bad tracing", "outside of ECL5a"}
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


def get_endpoint_matrix(
    df: pd.DataFrame,
    threshold: int = ENDPOINT_THRESHOLD,
    transform: str = CLUSTER_TRANSFORM,
) -> tuple[pd.DataFrame, list[str]]:
    """Return the endpoint matrix used for clustering."""
    if transform == "binary":
        return get_binary_endpoint_matrix(df, threshold=threshold)

    endpoint_cols = get_endpoint_columns(df)
    x_log = pd.DataFrame(
        StandardScaler().fit_transform(np.log1p(df[endpoint_cols])),
        columns=endpoint_cols,
        index=df.index,
    )
    return x_log, endpoint_cols


def get_cluster_order(linkage_matrix, cluster_labels: pd.Series) -> list[int]:
    """Order clusters by their first appearance in the dendrogram."""
    leaves = dendrogram(linkage_matrix, no_plot=True)["leaves"]
    return cluster_labels.iloc[leaves].drop_duplicates().tolist()


def add_hierarchical_cluster_label(
    df: pd.DataFrame,
    n_clusters: int = N_CLUSTERS,
    threshold: int = ENDPOINT_THRESHOLD,
    transform: str = CLUSTER_TRANSFORM,
    label_col: str = "projection_cluster",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], object, list[int]]:
    """Add the shared hierarchical SNR cluster label."""
    x_features, endpoint_cols = get_endpoint_matrix(df, threshold=threshold, transform=transform)
    linkage_matrix = linkage(pdist(x_features.to_numpy(), metric="euclidean"), method="ward")
    raw_labels = pd.Series(fcluster(linkage_matrix, t=n_clusters, criterion="maxclust"), index=df.index)
    cluster_order = get_cluster_order(linkage_matrix, raw_labels)
    relabel = {label: i + 1 for i, label in enumerate(cluster_order)}

    out = df.copy()
    out[label_col] = raw_labels.map(relabel).astype(int)
    cluster_order = list(range(1, n_clusters + 1))
    return out, x_features, endpoint_cols, linkage_matrix, cluster_order


def load_snr_data(
    n_clusters: int = N_CLUSTERS,
    threshold: int = ENDPOINT_THRESHOLD,
    transform: str = CLUSTER_TRANSFORM,
    with_details: bool = False,
) -> pd.DataFrame:
    """Load the prepared SNR table, filter excluded comments, and add both cluster labels."""
    df = pd.read_csv(SNR_PATH)

    keep = df[
        ~df["comment"].fillna("NA").isin(EXCLUDED_CLUSTER_COMMENTS)
        & df["x"].notna()
        & df["y"].notna()
        & df["z"].notna()
    ].copy()

    keep, x_binary, endpoint_cols, binary_linkage_matrix, binary_cluster_order = add_hierarchical_cluster_label(
        keep,
        n_clusters=n_clusters,
        threshold=threshold,
        transform="binary",
        label_col="projection_cluster_binary",
    )
    keep, x_log, _, log_linkage_matrix, log_cluster_order = add_hierarchical_cluster_label(
        keep,
        n_clusters=n_clusters,
        threshold=threshold,
        transform="log",
        label_col="projection_cluster_log",
    )

    if with_details:
        if transform == "log":
            return keep, x_log, endpoint_cols, log_linkage_matrix, log_cluster_order
        return keep, x_binary, endpoint_cols, binary_linkage_matrix, binary_cluster_order
        
    return keep


def add_legend(
    image_path: Path,
    order,
    colors: dict,
    legend_labels: dict,
    y0: int = 2220,
    gap: int = 120,
) -> None:
    """Add a simple horizontal legend to a saved brainrender screenshot."""
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 72)
    except OSError:
        font = ImageFont.load_default()

    text_widths = []
    total_width = 0
    r = 24
    for label in order:
        bbox = draw.textbbox((0, 0), legend_labels[label], font=font)
        text_width = bbox[2] - bbox[0]
        text_widths.append(text_width)
        total_width += (2 * r) + 42 + text_width

    total_width += gap * (len(order) - 1)
    x = int((image.size[0] - total_width) / 2)

    for label, text_width in zip(order, text_widths):
        draw.ellipse((x, y0 - r, x + 2 * r, y0 + r), fill=colors[label], outline=None)
        draw.text((x + 90, y0 - 42), legend_labels[label], fill=colors[label], font=font)
        x += (2 * r) + 42 + text_width + gap

    image.save(image_path)


def center_brainrender_image(image_path: Path, bottom_margin: int = 260) -> None:
    """Shift the rendered brain horizontally so it sits centrally on the canvas."""
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size

    analysis = image.crop((0, 0, width, height - bottom_margin))
    bbox = ImageChops.difference(analysis, Image.new("RGBA", analysis.size, (255, 255, 255, 255))).getbbox()
    if bbox is None:
        image.save(image_path)
        return

    left, _, right, _ = bbox
    content_center = (left + right) / 2
    canvas_center = width / 2
    shift = int(round(canvas_center - content_center))

    centered = Image.new("RGBA", image.size, (255, 255, 255, 255))
    centered.alpha_composite(image, (shift, 0))
    centered.save(image_path)


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
    center_brainrender_image(output_path)
    add_legend(output_path, order=order, colors=colors, legend_labels=legend_labels)
    return output_path
