#!/usr/bin/env python

import pathlib
from utils.imaging_utils import load_config
from core.session.mock_neurofeedback_session import NeuroFeedbackSession


if __name__ == "__main__":
    imaging_config_path = str(pathlib.Path(
        '/home') / 'pb' / 'PycharmProjects' / 'WideFlow' / 'wideflow' / 'Imaging' / 'imaging_configurations' / 'neurofeedback_3424_config.json')

    session_config = load_config(imaging_config_path)
    session_pipeline = NeuroFeedbackSession(session_config)
    session_pipeline.session_preparation()
    session_pipeline.run_session_pipeline()

