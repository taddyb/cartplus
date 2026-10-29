"""Benchmark models of the paper (regional HBV, dHBV, stateful LSTM, EA-LSTM)."""
import os


def data_files(dataset):
    """(npz, held-out list) recorded in run.json: GB "" (fixed file); US the env overrides CAMELS_US_NPZ671 / CAMELS_US_UNGAUGED of
    cart.data.load_us ("" = the defaults); AUS the files cart.data.load_aus reads (env CART_AUS_NPZ / CART_AUS_UNGAUGED or their
    defaults)."""
    if dataset == "us":
        return os.environ.get("CAMELS_US_NPZ671", ""), os.environ.get("CAMELS_US_UNGAUGED", "")
    if dataset == "aus":
        from cart import data as D
        return (os.environ.get("CART_AUS_NPZ", str(D.DATA / "camels_aus_v2.npz")),
                os.environ.get("CART_AUS_UNGAUGED", str(D.PKG / "data_lists" / "aus_ungauged_10pct.txt")))
    return "", ""
