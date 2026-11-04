"""
Demo Dataset Generator for Network Intrusion Detection System
=============================================================

Generates a realistic synthetic network traffic dataset formatted similarly
to CICIDS2017 with labeled attack categories and benign traffic.
Clearly marked as DEMO DATA.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_ids_data(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic network traffic dataset with realistic IDS features.

    Distribution:
    - ~65% BENIGN
    - ~12% DoS
    - ~8% PortScan
    - ~6% DDoS
    - ~5% Bot
    - ~4% Brute Force

    Parameters
    ----------
    n_samples : int, default=1500
        Number of network flow samples to generate.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Network traffic DataFrame with features and 'Label' column.
    """
    np.random.seed(random_state)

    labels_dist = {
        'BENIGN': int(n_samples * 0.65),
        'DoS': int(n_samples * 0.12),
        'PortScan': int(n_samples * 0.08),
        'DDoS': int(n_samples * 0.06),
        'Bot': int(n_samples * 0.05),
        'Brute Force': n_samples - (
            int(n_samples * 0.65) +
            int(n_samples * 0.12) +
            int(n_samples * 0.08) +
            int(n_samples * 0.06) +
            int(n_samples * 0.05)
        )
    }

    records = []

    for label, count in labels_dist.items():
        for _ in range(count):
            if label == 'BENIGN':
                flow_duration = int(np.random.exponential(scale=500000)) + 100
                tot_fwd_pkts = int(np.random.gamma(shape=3, scale=4)) + 1
                tot_bwd_pkts = int(np.random.gamma(shape=3, scale=3)) + 1
                pkt_len_mean = np.random.normal(loc=450, scale=120)
                pkt_len_min = max(0, int(np.random.normal(loc=40, scale=10)))
                pkt_len_max = max(pkt_len_min + 100, int(np.random.normal(loc=1400, scale=200)))
                pkt_len_std = np.random.uniform(50, 300)
                flow_bytes_s = np.random.uniform(1000, 500000)
                flow_pkts_s = np.random.uniform(5, 500)
                syn_flag = 1 if np.random.rand() < 0.15 else 0
                ack_flag = 1 if np.random.rand() < 0.85 else 0
                dest_port = np.random.choice([80, 443, 53, 8080, 21, 25, 110, 8443, 993])
                protocol = np.random.choice([6, 17], p=[0.75, 0.25])
                fwd_len_mean = pkt_len_mean * np.random.uniform(0.8, 1.2)
                bwd_len_mean = pkt_len_mean * np.random.uniform(0.9, 1.3)
                flow_iat_mean = np.random.exponential(scale=10000)

            elif label == 'DoS':
                flow_duration = int(np.random.exponential(scale=100000)) + 50
                tot_fwd_pkts = int(np.random.gamma(shape=12, scale=15)) + 20
                tot_bwd_pkts = int(np.random.gamma(shape=2, scale=2))
                pkt_len_mean = np.random.normal(loc=800, scale=150)
                pkt_len_min = int(np.random.normal(loc=64, scale=10))
                pkt_len_max = int(np.random.normal(loc=1460, scale=50))
                pkt_len_std = np.random.uniform(100, 400)
                flow_bytes_s = np.random.uniform(2000000, 15000000)
                flow_pkts_s = np.random.uniform(2000, 15000)
                syn_flag = 1 if np.random.rand() < 0.70 else 0
                ack_flag = 0 if np.random.rand() < 0.70 else 1
                dest_port = np.random.choice([80, 443, 8080])
                protocol = 6
                fwd_len_mean = np.random.normal(loc=900, scale=100)
                bwd_len_mean = np.random.normal(loc=200, scale=80)
                flow_iat_mean = np.random.exponential(scale=500)

            elif label == 'DDoS':
                flow_duration = int(np.random.exponential(scale=50000)) + 20
                tot_fwd_pkts = int(np.random.gamma(shape=20, scale=20)) + 50
                tot_bwd_pkts = int(np.random.gamma(shape=1, scale=1))
                pkt_len_mean = np.random.normal(loc=1100, scale=180)
                pkt_len_min = 64
                pkt_len_max = 1500
                pkt_len_std = np.random.uniform(150, 450)
                flow_bytes_s = np.random.uniform(10000000, 40000000)
                flow_pkts_s = np.random.uniform(8000, 35000)
                syn_flag = 1 if np.random.rand() < 0.85 else 0
                ack_flag = 0
                dest_port = np.random.choice([80, 443, 53])
                protocol = np.random.choice([6, 17], p=[0.6, 0.4])
                fwd_len_mean = np.random.normal(loc=1200, scale=80)
                bwd_len_mean = np.random.normal(loc=100, scale=50)
                flow_iat_mean = np.random.exponential(scale=100)

            elif label == 'PortScan':
                flow_duration = int(np.random.exponential(scale=10000)) + 5
                tot_fwd_pkts = int(np.random.choice([1, 2, 3]))
                tot_bwd_pkts = int(np.random.choice([0, 1]))
                pkt_len_mean = np.random.normal(loc=44, scale=6)
                pkt_len_min = 40
                pkt_len_max = int(np.random.normal(loc=60, scale=10))
                pkt_len_std = np.random.uniform(0, 15)
                flow_bytes_s = np.random.uniform(100, 10000)
                flow_pkts_s = np.random.uniform(10, 500)
                syn_flag = 1
                ack_flag = 0
                dest_port = int(np.random.choice(range(1, 65535)))
                protocol = 6
                fwd_len_mean = 44.0
                bwd_len_mean = 0.0 if tot_bwd_pkts == 0 else 40.0
                flow_iat_mean = np.random.exponential(scale=2000)

            elif label == 'Bot':
                flow_duration = int(np.random.exponential(scale=800000)) + 5000
                tot_fwd_pkts = int(np.random.gamma(shape=5, scale=5)) + 4
                tot_bwd_pkts = int(np.random.gamma(shape=4, scale=4)) + 3
                pkt_len_mean = np.random.normal(loc=350, scale=80)
                pkt_len_min = 32
                pkt_len_max = int(np.random.normal(loc=950, scale=120))
                pkt_len_std = np.random.uniform(60, 200)
                flow_bytes_s = np.random.uniform(5000, 200000)
                flow_pkts_s = np.random.uniform(10, 300)
                syn_flag = 0
                ack_flag = 1
                dest_port = np.random.choice([6667, 8088, irc_port := 7000, 9050, 4444, 80])
                protocol = 6
                fwd_len_mean = np.random.normal(loc=320, scale=60)
                bwd_len_mean = np.random.normal(loc=380, scale=70)
                flow_iat_mean = np.random.normal(loc=50000, scale=5000)

            else:  # Brute Force
                flow_duration = int(np.random.exponential(scale=300000)) + 1000
                tot_fwd_pkts = int(np.random.gamma(shape=8, scale=6)) + 10
                tot_bwd_pkts = int(np.random.gamma(shape=6, scale=5)) + 8
                pkt_len_mean = np.random.normal(loc=260, scale=50)
                pkt_len_min = 40
                pkt_len_max = int(np.random.normal(loc=700, scale=80))
                pkt_len_std = np.random.uniform(40, 150)
                flow_bytes_s = np.random.uniform(20000, 400000)
                flow_pkts_s = np.random.uniform(50, 600)
                syn_flag = 1 if np.random.rand() < 0.4 else 0
                ack_flag = 1
                dest_port = np.random.choice([22, 3389, 21, 23, 1433])
                protocol = 6
                fwd_len_mean = np.random.normal(loc=240, scale=40)
                bwd_len_mean = np.random.normal(loc=280, scale=50)
                flow_iat_mean = np.random.exponential(scale=8000)

            records.append({
                'Flow Duration': max(1, flow_duration),
                'Total Fwd Packets': max(1, tot_fwd_pkts),
                'Total Backward Packets': max(0, tot_bwd_pkts),
                'Flow Bytes/s': round(max(0.0, float(flow_bytes_s)), 2),
                'Flow Packets/s': round(max(0.1, float(flow_pkts_s)), 2),
                'Packet Length Mean': round(max(0.0, float(pkt_len_mean)), 2),
                'Packet Length Max': max(1, int(pkt_len_max)),
                'Packet Length Min': max(0, int(pkt_len_min)),
                'Packet Length Std': round(max(0.0, float(pkt_len_std)), 2),
                'SYN Flag Count': int(syn_flag),
                'ACK Flag Count': int(ack_flag),
                'Destination Port': int(dest_port),
                'Protocol': int(protocol),
                'Fwd Packet Length Mean': round(max(0.0, float(fwd_len_mean)), 2),
                'Bwd Packet Length Mean': round(max(0.0, float(bwd_len_mean)), 2),
                'Flow IAT Mean': round(max(0.0, float(flow_iat_mean)), 2),
                'Label': label
            })

    df = pd.DataFrame(records)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def main():
    """Generate and save sample dataset."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    out_file = data_dir / 'sample_network_traffic.csv'
    df = generate_synthetic_ids_data(n_samples=1500, random_state=42)
    df.to_csv(out_file, index=False)
    print(f"Generated {len(df)} sample records at {out_file}")
    print("\nClass Distribution:")
    print(df['Label'].value_counts())


if __name__ == '__main__':
    main()
