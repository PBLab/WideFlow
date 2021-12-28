#!/usr/bin/env python

from utils.imaging_utils import load_config
from wideflow.core.session.neurofeedback_session import NeuroFeedbackSession


if __name__ == "__main__":
    # imaging_config_path = '/data/Rotem/WideFlow prj/2680/20211208_neurofeedback/2680_12_08_2021__17_30_03.json'
    # imaging_config_path = '/home/pb/PycharmProjects/WideFlow/wideflow/Imaging/imaging_configurations/neurofeedback_config_for_changes.json'
    imaging_config_path = '/home/pb/PycharmProjects/WideFlow/wideflow/Imaging/imaging_configurations/training_config_for_changes.json'
    session_config = load_config(imaging_config_path)
    session_pipeline = NeuroFeedbackSession(session_config)
    session_pipeline.session_preparation()
    session_pipeline.run_session_pipeline()

