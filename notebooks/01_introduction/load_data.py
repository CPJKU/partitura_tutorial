#!/usr/bin/env python

import os
import json
import tarfile
import io
from urllib.request import urlopen
import urllib

from IPython.display import display
import ipywidgets as widgets
import appdirs

TIMEOUT = 2
REPO_NAME = "vienna4x22_rematched"
DATASET_BRANCH = "master"
OWNER = "OFAI"
DATASET_URL = "https://api.github.com/repos/{}/{}/tarball/{}".format(
    OWNER, REPO_NAME, DATASET_BRANCH
)


TMP_DIR = appdirs.user_cache_dir("partitura_tutorial")
CFG_FILE = os.path.join(TMP_DIR, "cache.json")
CFG = None
# DATASET_DIR will be set to the path of our data
DATASET_DIR = None


def load_cfg():
    global CFG
    if os.path.exists(CFG_FILE):
        with open(CFG_FILE) as f:
            CFG = json.load(f)
    else:
        CFG = {"last_dataset_dir": None}


def save_cfg():
    with open(CFG_FILE, "w") as f:
        json.dump(CFG, f)


def get_datasetdir():
    """Get the SHA of the latest commit and return the corresponding
    datast directory path.

    """
    commit_url = "https://api.github.com/repos/{}/{}/commits/{}".format(
        OWNER, REPO_NAME, DATASET_BRANCH
    )
    try:

        with urlopen(commit_url, timeout=TIMEOUT) as response:
            commit = json.load(response)
        repo_dirname = "{}-{}-{}".format(OWNER, REPO_NAME, commit["sha"][:7])
        return os.path.join(TMP_DIR, repo_dirname)

    except urllib.error.URLError as e:
        # warnings.warn('{} (url: {})'.format(e, commit_url))
        return CFG.get("last_dataset_dir", None)
    except Exception as e:
        # warnings.warn('{} (url: {})'.format(e, commit_url))
        return CFG.get("last_dataset_dir", None)


def init_dataset():
    global DATASET_DIR, PIECES, PERFORMERS, SCORE_PERFORMANCE_PAIRS

    load_cfg()

    status = widgets.Output()
    display(status)
    status.clear_output()

    DATASET_DIR = get_datasetdir()

    if DATASET_DIR is None:
        status.append_stdout("No internet connection?\n")

    elif os.path.exists(DATASET_DIR):

        status.append_stdout("Vienna 4x22 Corpus already downloaded.\n")
        status.append_stdout("Data is in {}".format(DATASET_DIR))

    else:
        status.append_stdout("Downloading Vienna 4x22 Corpus...")
        try:
            try:
                urldata = urlopen(DATASET_URL).read()
            except urllib.error.URLError as e:
                # warnings.warn('{} (url: {})'.format(e, DATASET_URL))
                status.append_stdout("error. No internet connection?\n")
                return

            with tarfile.open(fileobj=io.BytesIO(urldata)) as archive:
                folder = next(iter(archive.getnames()), None)
                archive.extractall(TMP_DIR)
                if folder:
                    DATASET_DIR = os.path.join(TMP_DIR, folder)
                    CFG["last_dataset_dir"] = DATASET_DIR
                    save_cfg()
                # assert DATASET_DIR == os.path.join(TMP_DIR, folder)

        except Exception as e:
            status.append_stdout("\nError: {}".format(e))
            return None
        status.append_stdout("done\nData is in {}".format(DATASET_DIR))

    return DATASET_DIR


if __name__ == "__main__":
    init_dataset()
