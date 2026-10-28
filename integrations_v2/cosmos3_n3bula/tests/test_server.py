# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

import pytest
from cosmos3_n3bula.server import parse_args, settings_from_args

pytestmark = pytest.mark.ci_cpu


def test_server_defaults_match_the_fast_single_camera_profile():
    settings = settings_from_args(parse_args([]))

    assert settings.device == "cuda:0"
    assert settings.fps == 30.0
    assert settings.accelerated_rms_norm
    assert settings.compile_denoise == "default"
    assert settings.compile_control_encode
    assert not settings.compile_decoder


def test_eager_flags_translate_into_disabled_compilation():
    args = parse_args(
        [
            "--compile-denoise",
            "off",
            "--no-accelerated-rms-norm",
            "--no-compile-control-encode",
            "--attention-backend",
            "cudnn",
        ]
    )

    settings = settings_from_args(args)

    assert settings.compile_denoise is None
    assert not settings.accelerated_rms_norm
    assert not settings.compile_control_encode
    assert settings.attention_backend == "cudnn"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--compile-denoise", "invalid"],
        ["--attention-backend", "invalid"],
    ],
)
def test_unknown_runtime_modes_are_rejected_by_cli(arguments):
    with pytest.raises(SystemExit) as raised:
        parse_args(arguments)

    assert raised.value.code == 2