# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Cosmos3 generation-mode selection."""


def generation_mode(has_seed_image: bool) -> tuple[str, int]:
    """Return the native model mode and its opening conditioning-frame count."""
    return ("image2video", 1) if has_seed_image else ("text2video", 0)
