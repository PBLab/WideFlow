#!/usr/bin/env python

from utils.load_config import load_config
import argparse
import os


if __name__ == "__main__":
    # nice_val = os.nice(0)
    # os.nice(nice_val-10)
    parser = argparse.ArgumentParser(description='Wide Field Real-Time Analysis and Neurofeedback')

    parser.add_argument('-c', '--config_path', type=str, help='full path to session configuration JSON file')
    parser.add_argument('-s', '--session', type=str, help='session pipeline class name')

    args = parser.parse_args()

    session_config = load_config(args.config_path)
    if args.session == 'NeuroFeedbackSession':
        from core.session.neurofeedback_session import NeuroFeedbackSession
        session_pipeline = NeuroFeedbackSession(session_config)

    elif args.session == 'PostAnalysisNeuroFeedbackSession':
        from core.session.mock_neurofeedback_session import PostAnalysisNeuroFeedbackSession
        session_pipeline = PostAnalysisNeuroFeedbackSession(session_config)

    else:
        raise NameError(f'{args.session} session class was not found')

    session_pipeline.session_preparation()
    session_pipeline.run_session_pipeline()


