try:
    from importlib.metadata import version

    __version__ = version("meegsim")
except Exception:  # noqa: BLE001
    __version__ = "0.0.0"
