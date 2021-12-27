#!/usr/bin/env python

import pathlib
from utils.imaging_utils import load_config
from wideflow.core.session.mock_neurofeedback_session import PostAnalysisNeuroFeedbackSession


if __name__ == "__main__":
    # imaging_config_path = str(pathlib.Path(
    #     '/home') / 'pb' / 'PycharmProjects' / 'WideFlow' / 'wideflow' / 'Imaging' / 'imaging_configurations' / 'neurofeedback_session_config.json')
    imaging_config_path = '/data/Rotem/WideFlow prj/2680/20211208_neurofeedback/2680_12_08_2021__17_30_03.json'
    session_config = load_config(imaging_config_path)
    session_pipeline = PostAnalysisNeuroFeedbackSession(session_config)
    session_pipeline.session_preparation()
    session_pipeline.run_session_pipeline()

