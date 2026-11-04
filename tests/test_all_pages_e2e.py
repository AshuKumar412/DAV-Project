"""
End-to-End Simulation Test for all 6 Dashboard Pages in app.py.
Verifies no ValueError truth-value errors or runtime exceptions occur
in Demo Mode, Custom Upload Mode, Empty Dataset, and Missing Events.
"""

import sys
from pathlib import Path
import unittest.mock as mock
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import app


def mock_columns_func(spec, *args, **kwargs):
    count = spec if isinstance(spec, int) else len(spec)
    return [mock.MagicMock() for _ in range(count)]


def mock_selectbox_func(label, options, *args, **kwargs):
    opts = list(options)
    return opts[0] if opts else None


def run_e2e_page_tests():
    print("Beginning E2E Simulation of All 6 Pages...")

    # Mock streamlit functions that render UI so we can execute page functions directly
    with mock.patch('streamlit.markdown'), \
         mock.patch('streamlit.title'), \
         mock.patch('streamlit.warning'), \
         mock.patch('streamlit.info'), \
         mock.patch('streamlit.success'), \
         mock.patch('streamlit.error'), \
         mock.patch('streamlit.dataframe'), \
         mock.patch('streamlit.plotly_chart'), \
         mock.patch('streamlit.columns', side_effect=mock_columns_func), \
         mock.patch('streamlit.selectbox', side_effect=mock_selectbox_func), \
         mock.patch('streamlit.multiselect', return_value=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']), \
         mock.patch('streamlit.slider', return_value=0.1), \
         mock.patch('streamlit.button', return_value=False), \
         mock.patch('streamlit.download_button', return_value=False), \
         mock.patch('streamlit.tabs', return_value=[mock.MagicMock(), mock.MagicMock(), mock.MagicMock()]), \
         mock.patch('streamlit.expander', return_value=mock.MagicMock(__enter__=mock.MagicMock(), __exit__=mock.MagicMock())):

        # -------------------------------------------------------------
        # SCENARIO 1: Demo Dataset Mode (Fresh state, no events yet)
        # -------------------------------------------------------------
        print("\n--- Test Scenario 1: Demo Mode (Fresh Boot) ---")
        app.init_session_state()
        demo_df = pd.read_csv(project_root / 'data' / 'sample_network_traffic.csv')
        app.st.session_state['data'] = demo_df
        app.st.session_state['demo_mode'] = True
        app.st.session_state['target_col'] = 'Label'
        app.st.session_state['security_events'] = []
        app.st.session_state['predictions'] = None

        print("Testing Page 1: Security Overview...")
        app.render_security_overview()

        print("Testing Page 2: Traffic Analysis...")
        app.render_traffic_analysis()

        print("Testing Page 3: Intrusion Detection...")
        app.render_intrusion_detection()

        print("Testing Page 4: Anomaly Detection...")
        app.render_anomaly_detection()

        print("Testing Page 5: Model Performance...")
        app.render_model_performance()

        print("Testing Page 6: Security Events (Empty events)...")
        app.render_security_events()
        print("Scenario 1 Passed without errors!")

        # -------------------------------------------------------------
        # SCENARIO 2: With Predictions & Generated Events
        # -------------------------------------------------------------
        print("\n--- Test Scenario 2: With Active Predictions & Events ---")
        preds = pd.DataFrame({
            'predicted_label': ['BENIGN', 'DoS', 'Bot', 'PortScan', 'BENIGN'],
            'confidence': [0.99, 0.88, 0.94, 0.75, 0.98],
            'is_malicious': [False, True, True, True, False]
        })
        app.st.session_state['predictions'] = preds
        events = app.generate_security_events(preds, model_name="Random Forest")
        app.st.session_state['security_events'] = events

        print("Testing Page 1: Security Overview with active events...")
        app.render_security_overview()

        print("Testing Page 3: Intrusion Detection with predictions...")
        app.render_intrusion_detection()

        print("Testing Page 6: Security Events with DataFrame events...")
        app.render_security_events()
        print("Scenario 2 Passed without errors!")

        # -------------------------------------------------------------
        # SCENARIO 3: Custom CSV Upload Mode (Empty DataFrame)
        # -------------------------------------------------------------
        print("\n--- Test Scenario 3: Custom CSV Upload Mode (Empty Data) ---")
        app.st.session_state['data'] = pd.DataFrame()
        app.st.session_state['demo_mode'] = False
        app.st.session_state['target_col'] = None
        app.st.session_state['security_events'] = pd.DataFrame()
        app.st.session_state['predictions'] = pd.DataFrame()

        print("Testing Page 1...")
        app.render_security_overview()

        print("Testing Page 2...")
        app.render_traffic_analysis()

        print("Testing Page 3...")
        app.render_intrusion_detection()

        print("Testing Page 4...")
        app.render_anomaly_detection()

        print("Testing Page 5...")
        app.render_model_performance()

        print("Testing Page 6...")
        app.render_security_events()
        print("Scenario 3 Passed without errors!")

        # -------------------------------------------------------------
        # SCENARIO 4: Custom Unlabeled Dataset
        # -------------------------------------------------------------
        print("\n--- Test Scenario 4: Custom Unlabeled Dataset ---")
        unlabeled_df = demo_df.drop(columns=['Label'])
        app.st.session_state['data'] = unlabeled_df
        app.st.session_state['demo_mode'] = False
        app.st.session_state['target_col'] = None
        app.st.session_state['security_events'] = []
        app.st.session_state['predictions'] = None

        print("Testing Page 1...")
        app.render_security_overview()

        print("Testing Page 2...")
        app.render_traffic_analysis()

        print("Testing Page 4...")
        app.render_anomaly_detection()

        print("Testing Page 6...")
        app.render_security_events()
        print("Scenario 4 Passed without errors!")

    print("\n=======================================================")
    print("ALL 6 PAGES TESTED ACROSS ALL SCENARIOS: 100% SUCCESS!")
    print("=======================================================")


if __name__ == '__main__':
    run_e2e_page_tests()
