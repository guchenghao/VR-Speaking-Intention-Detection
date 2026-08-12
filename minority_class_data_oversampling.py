import numpy as np
from scipy.interpolate import CubicSpline


def jitter(x, sigma=0.03):
    """Add point-wise Gaussian noise."""
    return x + np.random.normal(
        loc=0.0,
        scale=sigma,
        size=x.shape
    )


def scaling(x, sigma=0.1):
    """
    Apply time-point-wise magnitude scaling.

    The same scaling factor is applied to all channels at each
    time step, while the factor varies across time.
    """
    factor = np.random.normal(
        loc=1.0,
        scale=sigma,
        size=(x.shape[0], 1)
    )
    return np.multiply(x, factor)


def time_warp(x, sigma=0.2, knot=4):
    """
    Apply smooth nonlinear temporal warping.

    The same temporal warping function is applied to all channels
    within a modality.
    """
    x = np.asarray(x)

    time_steps = x.shape[0]
    channels = x.shape[1]

    orig_steps = np.arange(time_steps)

    # Generate random temporal scaling factors at spline knots.
    random_warps = np.random.normal(
        loc=1.0,
        scale=sigma,
        size=(knot + 2,)
    )

    warp_steps = np.linspace(
        0,
        time_steps - 1,
        knot + 2
    )

    # Construct a smooth warping function.
    time_warp_func = CubicSpline(
        warp_steps,
        random_warps
    )

    warps = time_warp_func(orig_steps)

    # Convert local temporal scaling into warped time coordinates.
    warped_steps = np.cumsum(warps)

    # Normalize to the original temporal range.
    warped_steps = (
        (warped_steps - warped_steps.min())
        / (warped_steps.max() - warped_steps.min())
    )
    warped_steps *= (time_steps - 1)

    ret = np.zeros_like(x)

    # Preserve the internal synchronization of channels
    # belonging to the same modality.
    for ch in range(channels):
        ret[:, ch] = np.interp(
            orig_steps,
            warped_steps,
            x[:, ch]
        )

    return ret


def augment_data(
    ecg_data,
    acc_data,
    labels,
    person_ids,
    target_count=None
):
    """
    Perform participant-wise class balancing using data augmentation.

    For each participant, the minority class is augmented to match
    the participant's majority class. If both classes are already
    balanced, no augmentation is performed.

    Participants containing only one class are left unchanged because
    samples of the missing class cannot be generated from existing data.
    """
    labels = np.asarray(labels)
    person_ids = np.asarray(person_ids)

    augmented_ecg = []
    augmented_acc = []
    augmented_labels = []
    augmented_person_ids = []

    for person_id in np.unique(person_ids):

        person_indices = np.where(person_ids == person_id)[0]
        person_labels = labels[person_indices]

        unique, counts = np.unique(
            person_labels,
            return_counts=True
        )

        class_counts = dict(zip(unique, counts))

        print(
            f"Person {person_id} original distribution: "
            f"{class_counts}"
        )

        # Keep all original samples.
        for idx in person_indices:
            augmented_ecg.append(ecg_data[idx])
            augmented_acc.append(acc_data[idx])
            augmented_labels.append(labels[idx])
            augmented_person_ids.append(person_ids[idx])

        # A missing class cannot be synthesized from existing samples.
        if len(unique) < 2:
            print(
                f"Person {person_id} contains only class "
                f"{unique.tolist()}; augmentation skipped."
            )
            continue

        # By default, balance both classes to the participant-specific
        # majority-class sample count.
        person_target_count = (
            max(counts)
            if target_count is None
            else target_count
        )

        for class_label in unique:

            current_count = class_counts[class_label]

            if current_count >= person_target_count:
                continue

            need_count = (
                person_target_count - current_count
            )

            class_indices = person_indices[
                person_labels == class_label
            ]

            print(
                f"Person {person_id}, class {class_label}: "
                f"generating {need_count} augmented samples."
            )

            for _ in range(need_count):

                # Randomly select an existing sample from this
                # participant and class.
                idx = np.random.choice(class_indices)

                ecg_sample = ecg_data[idx].copy()
                acc_sample = acc_data[idx].copy()

                aug_method = np.random.choice(
                    [
                        "jitter",
                        "scaling",
                        "time_warp",
                        "combined"
                    ]
                )

                if aug_method == "jitter":
                    ecg_aug = jitter(ecg_sample)
                    acc_aug = jitter(acc_sample)

                elif aug_method == "scaling":
                    ecg_aug = scaling(ecg_sample)
                    acc_aug = scaling(acc_sample)

                elif aug_method == "time_warp":
                    ecg_aug = time_warp(ecg_sample)
                    acc_aug = time_warp(acc_sample)

                else:
                    # Combined magnitude perturbation.
                    ecg_aug = jitter(scaling(ecg_sample))
                    acc_aug = jitter(scaling(acc_sample))

                augmented_ecg.append(ecg_aug)
                augmented_acc.append(acc_aug)
                augmented_labels.append(class_label)
                augmented_person_ids.append(person_id)

    return (
        np.array(augmented_ecg),
        np.array(augmented_acc),
        np.array(augmented_labels),
        np.array(augmented_person_ids)
    )
