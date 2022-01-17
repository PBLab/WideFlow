#!/usr/bin/env python

from utils.imaging_utils import load_config
from wideflow.core.session.neurofeedback_session import NeuroFeedbackSession
from wideflow.core.session.mock_neurofeedback_session import PostAnalysisNeuroFeedbackSession
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wide Field Real-Time Analysis and Neurofeedback')

    parser.add_argument('--path', type=str, help='full path to session configuration JSON file')
    parser.add_argument('--session', type=str, help='session pipeline class name')

    args = parser.parse_args()

    session_config = load_config(args.path)
    if args.pipeline == 'NeuroFeedbackSession':
        session_pipeline = NeuroFeedbackSession(session_config)

    elif args.pipeline == 'PostAnalysisNeuroFeedbackSession':
        session_pipeline = PostAnalysisNeuroFeedbackSession(session_config)

    else:
        raise NameError(f'{args.pipeline} session class was not found')

    session_pipeline.session_preparation()
    session_pipeline.run_session_pipeline()


