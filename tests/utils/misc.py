import os


def running_on_ci():
    return os.environ.get("BUILD_ENV", "local") == "ci"
