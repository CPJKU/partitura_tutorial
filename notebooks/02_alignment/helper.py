#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper methods for plotting and visualizing alignments
"""
import numpy as np
import os

from shutil import make_archive

from matplotlib import lines, pyplot as plt
from sklearn.datasets import make_blobs

import partitura as pt

from partitura.score import ScoreLike
from partitura.performance import PerformanceLike
from partitura.utils.misc import PathLike

from partitura.io.exportparangonada import alignment_dicts_to_array

from scipy.sparse import csc_matrix

from typing import Union, Optional, List, Tuple

# Define random state for reproducibility
RNG = np.random.RandomState(1984)


def generate_example_sequences(
    lenX: int = 100,
    centers: int = 3,
    n_features: int = 5,
    maxreps: int = 4,
    minreps: int = 1,
    noise_scale: float = 0.01,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Generates example pairs of related sequences. Sequence X are samples of
    an K-dimensional space around a specified number of centroids.
    Sequence Y is a non-constant "time-streched" version of X with some
    noise added.

    Parameters
    ----------
    lenX : int
        Number of elements in the X sequence
    centers: int
        Number of different centers ("classes") that the elements
        of the sequences represent
    n_features: int
        Dimensionality of the features ($K$) in the notation of the
        Notebook
    noise_scale: float
        Scale of the noise

    Returns
    -------
    X : np.ndarray
        Sequence X (a matrix where each row represents
        an element of the sequence)
    Y: np.ndarray
        Sequence Y
    ground_truth_path: np.ndarray
        Alignment between X and Y where the first column represents the indices
        in X and the second column represents the corresponding index in Y.
    """

    X, _ = make_blobs(n_samples=lenX, centers=centers, n_features=n_features)
    # Time stretching X! each element in sequence X is
    # repeated a random number of times
    # and then we add some noise to spice things up :)

    if minreps == maxreps:
        n_reps = np.ones(len(X), dtype=int) * minreps
    else:
        n_reps = RNG.randint(minreps, maxreps, len(X))
    y_idxs = [rp * [i] for i, rp in enumerate(n_reps)]
    y_idxs = np.array([el for reps in y_idxs for el in reps], dtype=int)
    # Add a bias, so that Y has a different "scaling" than X
    Y = X[y_idxs]
    # add some noise
    Y += noise_scale * RNG.randn(*Y.shape)
    ground_truth_path = np.column_stack((y_idxs, np.arange(len(Y))))
    return X, Y, ground_truth_path


def plot_alignment(
    X: np.ndarray,
    Y: np.ndarray,
    alignment_path: np.ndarray,
) -> None:
    """
    Visualize alignment between two sequences.

    Parameters
    ----------
    X : np.ndarray
        Reference sequence (a matrix where each row represents an element of
        the sequence)
    Y : np.ndarray
        The sequence we want to align to X.
    alignment_path : np.ndarray
        A 2D array where each row corresponds to the indices in array X and its
        corresponding element in X.
    """
    vmax = max(max(abs(X.max()), abs(X.min())), max(abs(Y.max()), abs(Y.min())))
    fig, axes = plt.subplots(2, sharex=True)
    axes[0].imshow(
        X.T,
        cmap="gray",
        origin="lower",
        aspect="equal",
        interpolation=None,
        vmax=vmax,
        vmin=-vmax,
    )
    axes[0].set_ylabel(r"$\mathbf{X}$")
    axes[1].imshow(
        Y.T,
        cmap="gray",
        origin="lower",
        aspect="equal",
        interpolation=None,
        vmax=vmax,
        vmin=-vmax,
    )
    axes[1].set_ylabel(r"$\mathbf{Y}$")
    axes[0].set_xlim((-1, max(len(X), len(Y)) + 1))
    axes[1].set_xlim((-1, max(len(X), len(Y)) + 1))
    axes[0].set_ylim((-1, X.shape[1] + 1))
    axes[1].set_ylim((-1, Y.shape[1] + 1))

    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["bottom"].set_visible(False)
    axes[0].spines["left"].set_visible(False)
    axes[0].get_xaxis().set_ticks([])
    axes[0].get_yaxis().set_ticks([])
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["bottom"].set_visible(False)
    axes[1].spines["left"].set_visible(False)
    axes[1].get_xaxis().set_ticks([])
    axes[1].get_yaxis().set_ticks([])

    for ref_idx, perf_idx in alignment_path:
        # Add line from one subplot to the other
        xyA = [ref_idx, 0]
        axes[0].plot(*xyA)
        xyB = [perf_idx, Y.shape[1] - 0.75]
        axes[1].plot(*xyB)
        transFigure = fig.transFigure.inverted()
        coord1 = transFigure.transform(axes[0].transData.transform(xyA))
        coord2 = transFigure.transform(axes[1].transData.transform(xyB))
        line = lines.Line2D(
            (coord1[0], coord2[0]),  # xdata
            (coord1[1], coord2[1]),  # ydata
            transform=fig.transFigure,
            color="red",
            linewidth=0.5,
        )
        fig.lines.append(line)

    plt.show()


def compute_pitch_class_pianoroll(
    note_info: Union[
        pt.score.ScoreLike,
        pt.performance.PerformanceLike,
        np.ndarray,
        csc_matrix,
    ],
    normalize: bool = True,
    time_unit: str = "auto",
    time_div: int = "auto",
    onset_only: bool = False,
    note_separation: bool = False,
    time_margin: int = 0,
    remove_silence: bool = True,
    end_time: Optional[float] = None,
    binary: bool = False,
) -> np.ndarray:
    """
    Compute a pitch class piano roll.

    Parameters
    ----------

    """
    pianoroll = None
    if isinstance(note_info, csc_matrix):
        pianoroll = note_info

    if pianoroll is None:

        pianoroll = pt.utils.compute_pianoroll(
            note_info=note_info,
            time_unit=time_unit,
            time_div=time_div,
            onset_only=onset_only,
            note_separation=note_separation,
            pitch_margin=-1,
            time_margin=time_margin,
            return_idxs=False,
            piano_range=False,
            remove_drums=True,
            remove_silence=remove_silence,
            end_time=end_time,
        )

    pc_pianoroll = np.zeros((12, pianoroll.shape[1]), dtype=float)
    for i in range(int(np.ceil(128 / 12))):
        pr_slice = pianoroll[i * 12 : (i + 1) * 12, :].toarray().astype(float)
        pc_pianoroll[: pr_slice.shape[0], :] += pr_slice

    if binary:
        # only show active pitch classes
        pc_pianoroll[pc_pianoroll > 0] = 1

    if normalize:
        norm_term = pc_pianoroll.sum(0)
        # avoid dividing by 0 if a slice is empty
        norm_term[np.isclose(norm_term, 0)] = 1
        pc_pianoroll /= norm_term

    return pc_pianoroll


def compute_pitch_class_distribution_windowed(piano_roll, time_div, win_size):
    n_windows = int(np.ceil(piano_roll.shape[1] / (time_div * win_size)))

    window_size = win_size * time_div

    observations = np.zeros((n_windows, 12))
    for win in range(n_windows):
        idx = slice(win * window_size, (win + 1) * window_size)
        segment = piano_roll[:, idx].sum(1)
        dist = np.zeros(12)
        pitch_idxs = np.where(segment != 0)[0]
        for pix in pitch_idxs:
            dist[pix % 12] += segment[pix]
        dist /= dist.sum()
        observations[win] = dist

    return observations


def evaluate_alignment_notewise(
    prediction: List[dict],
    ground_truth: List[dict],
    types: List[str] = ["match", "deletion", "insertion"],
) -> (float, float, float):
    """
    Evaluate Alignments.

    This methods evaluates note-level alignments by computing the
    precision, recall and F-score.

    Parameters
    ----------
    prediction: List of dicts
        List of dictionaries containing the predicted alignments
    ground_truth:
        List of dictionaries containing the ground truth alignments
    types: List of strings
        List of alignment types to consider for evaluation
        (e.g ['match', 'deletion', 'insertion']

    Returns
    -------
    precision: float
       The precision
    recall: float
        The recall
    f_score: float
       The F score
    """

    sanitize_alignment(prediction)
    sanitize_alignment(ground_truth)
    pred_filtered = list(filter(lambda x: x["label"] in types, prediction))
    gt_filtered = list(filter(lambda x: x["label"] in types, ground_truth))

    filtered_correct = [pred for pred in pred_filtered if pred in gt_filtered]

    n_pred_filtered = len(pred_filtered)
    n_gt_filtered = len(gt_filtered)
    n_correct = len(filtered_correct)

    if n_pred_filtered > 0 or n_gt_filtered > 0:
        precision = n_correct / n_pred_filtered if n_pred_filtered > 0 else 0.0
        recall = n_correct / n_gt_filtered if n_gt_filtered > 0 else 0
        f_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
    else:
        # no prediction and no ground truth for a
        # given type -> correct alignment
        precision, recall, f_score = 1.0, 1.0, 1.0

    return precision, recall, f_score


def sanitize_alignment(alignment: List[dict]) -> None:
    """
    Ensure that note ids are strings in alignments.
    These method changes alignments in-place.

    Parameters
    ----------
    alignment : List[dict]
        List of dictionaries containing an alignment.
    """
    for note in alignment:

        score_id = note.get("score_id", None)

        if score_id is not None:
            note["score_id"] = str(score_id)
        perf_id = note.get("performance_id", None)

        if perf_id is not None:
            note["performance_id"] = str(perf_id)


def greedy_note_alignment(
    warping_path: np.ndarray,
    idx1: np.ndarray,
    note_array1: np.ndarray,
    idx2: np.ndarray,
    note_array2: np.ndarray,
) -> List[dict]:
    """
    Greedily find and store possible note alignments

    Parameters
    ----------
    warping_path : numpy ndarray
        alignment sequence idx in stacked columns
    idx1: numpy ndarray
        pitch, start, and end coordinates of all notes in note_array1
    note_array1: numpy structured array
        note_array of sequence 1 (the score)
    idx2: numpy ndarray
        pitch, start, and end coordinates of all notes in note_array2
    note_array2: numpy structured array
        note_array of sequence 2 (the performance)

    Returns
    ----------
    note_alignment : list
        list of note alignment dictionaries

    """
    note_alignment = []
    used_notes1 = list()
    used_notes2 = list()

    coord_info1 = idx1
    if idx1.shape[1] == 3:
        # Assume that the first column contains the correct MIDI pitch
        coord_info1 = np.column_stack((idx1, idx1[:, 0]))

    coord_info2 = idx2

    if idx2.shape[1] == 3:
        # Assume that the first column contains the correct MIDI pitch
        coord_info2 = np.column_stack((idx2, idx2[:, 0]))

    # loop over all notes in sequence 1
    for note1, coord1 in zip(note_array1, coord_info1):
        note1_id = note1["id"]
        pc1, s1, e1, pitch1 = coord1

        # find the coordinates of the note in the warping_path

        idx_in_warping_path = np.all(
            [warping_path[:, 0] >= s1, warping_path[:, 0] <= e1], axis=0
        )
        # print(idx_in_warping_path, idx_in_warping_path.shape)
        range_in_sequence2 = warping_path[idx_in_warping_path, 1]
        max2 = np.max(range_in_sequence2)
        min2 = np.min(range_in_sequence2)

        # loop over all notes in sequence 2 and pick the notes with same pitch
        # and position
        for note2, coord2 in zip(note_array2, coord_info2):
            note2_id = note2["id"]
            pc2, s2, e2, pitch2 = coord2
            if note2_id not in used_notes2:
                if pitch2 == pitch1 and s2 <= max2 and e2 >= min2:

                    note_alignment.append(
                        {
                            "label": "match",
                            "score_id": note1_id,
                            "performance_id": str(note2_id),
                        }
                    )
                    used_notes2.append(str(note2_id))
                    used_notes1.append(note1_id)

        # check if a note has been found for the sequence 1 note,
        # otherwise add it as deletion
        if note1_id not in used_notes1:
            note_alignment.append({"label": "deletion", "score_id": note1_id})
            used_notes1.append(note1_id)

    # check again for all notes in sequence 2, if not used,
    # add them as insertions
    for note2 in note_array2:
        note2_id = note2["id"]
        if note2_id not in used_notes2:
            note_alignment.append(
                {
                    "label": "insertion",
                    "performance_id": str(note2_id),
                }
            )
            used_notes2.append(str(note2_id))

    return note_alignment


def dummy_linear_alignment(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    A Dummy linear alignment
    """

    alignment_times_score = np.arange(X.shape[0])
    alignment_times_perf = np.floor(
        np.arange(X.shape[0]) * Y.shape[0] / X.shape[0],
    )

    linear_alignment = np.column_stack(
        [
            alignment_times_score,
            alignment_times_perf,
        ],
    ).astype(int)

    return linear_alignment


def save_parangonada_csv(
    alignment: List[dict],
    performance_data: Union[PerformanceLike, np.ndarray],
    score_data: Union[ScoreLike, np.ndarray],
    outdir: Optional[PathLike] = None,
    zalign: Optional[List[dict]] = None,
    feature: Optional[List[dict]] = None,
) -> Optional[Tuple[np.ndarray]]:
    """
    Save an alignment for visualization with parangonda.

    Parameters
    ----------
    alignment : list
        A list of note alignment dictionaries.
    performance_data : Performance, PerformedPart, structured ndarray
        The performance information
    score_data : ScoreLike
        The musical score. A :class:`partitura.score.Score` object,
        a :class:`partitura.score.Part`, a :class:`partitura.score.PartGroup` or
        a list of these.
    outdir : PathLike
        A directory to save the files into.
    ppart : PerformedPart, structured ndarray
        A PerformedPart or its note_array.
    zalign : list, optional
        A second list of note alignment dictionaries.
    feature : list, optional
        A list of expressive feature dictionaries.

    Returns
    -------
    perf_note_array : np.ndarray
        The performance note array. Only returned if `outdir` is None.
    score_note_array: np.ndarray
        The note array from the score. Only returned if `outdir` is None.
    alignarray: np.ndarray
    zalignarray: np.ndarray
    featurearray: np.ndarray
    """

    score_note_array = pt.utils.ensure_notearray(score_data)

    perf_note_array = pt.utils.ensure_notearray(performance_data)

    ffields = [
        ("velocity", "<f4"),
        ("timing", "<f4"),
        ("articulation", "<f4"),
        ("id", "U256"),
    ]

    farray = []
    notes = list(score_note_array["id"])
    if feature is not None:
        # veloctiy, timing, articulation, note
        for no, i in enumerate(list(feature["id"])):
            farray.append(
                (
                    feature["velocity"][no],
                    feature["timing"][no],
                    feature["articulation"][no],
                    i,
                )
            )
    else:
        for no, i in enumerate(notes):
            farray.append((0, 0, 0, i))

    featurearray = np.array(farray, dtype=ffields)
    alignarray = alignment_dicts_to_array(alignment)

    if zalign is not None:
        zalignarray = alignment_dicts_to_array(zalign)
    else:  # if no zalign is available, save the same alignment twice
        zalignarray = alignment_dicts_to_array(alignment)

    if outdir is not None:
        np.savetxt(
            os.path.join(outdir, "ppart.csv"),
            # outdir + os.path.sep + "perf_note_array.csv",
            perf_note_array[
                [
                    "onset_sec",
                    "duration_sec",
                    "pitch",
                    "velocity",
                    "track",
                    "channel",
                    "id",
                ]
            ],
            fmt="%.20s",
            delimiter=",",
            header=",".join(
                [
                    "onset_sec",
                    "duration_sec",
                    "pitch",
                    "velocity",
                    "track",
                    "channel",
                    "id",
                ]
            ),
            comments="",
        )
        np.savetxt(
            os.path.join(outdir, "part.csv"),
            # outdir + os.path.sep + "score_note_array.csv",
            score_note_array,
            fmt="%.20s",
            delimiter=",",
            header=",".join(score_note_array.dtype.names),
            comments="",
        )
        np.savetxt(
            os.path.join(outdir, "align.csv"),
            # outdir + os.path.sep + "align.csv",
            alignarray,
            fmt="%.20s",
            delimiter=",",
            header=",".join(alignarray.dtype.names),
            comments="",
        )
        np.savetxt(
            os.path.join(outdir, "zalign.csv"),
            # outdir + os.path.sep + "zalign.csv",
            zalignarray,
            fmt="%.20s",
            delimiter=",",
            header=",".join(zalignarray.dtype.names),
            comments="",
        )
        np.savetxt(
            os.path.join(outdir, "feature.csv"),
            # outdir + os.path.sep + "feature.csv",
            featurearray,
            fmt="%.20s",
            delimiter=",",
            header=",".join(featurearray.dtype.names),
            comments="",
        )

        # Zip all files
        make_archive(outdir, "zip", outdir)
    else:
        return (
            perf_note_array,
            score_note_array,
            alignarray,
            zalignarray,
            featurearray,
        )
