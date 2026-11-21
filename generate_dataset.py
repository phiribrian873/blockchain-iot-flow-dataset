"""
Synthetic Blockchain-IoT network-flow dataset generator.
NMAI513 End Term Examination - Machine Learning for Blockchain-Enabled IoT Networks.

Every record is one network flow between an IoT device and a blockchain gateway/service.
The generator is seeded so that the dataset is fully reproducible.
Benign and attack flows are drawn from class-conditional distributions that overlap,
so that the classification problem is realistic rather than trivially separable.
"""
import numpy as np
import pandas as pd

SEED = 2026
N = 12000
rng = np.random.default_rng(SEED)

DEVICE_TYPES = ["sensor", "gateway", "actuator", "camera", "wearable"]
PROTOCOLS = ["TCP", "UDP", "MQTT", "CoAP", "ICMP"]
ATTACK_TYPES = ["DoS_Flood", "Port_Scan", "Auth_Spoofing", "Data_Exfiltration"]

# Class mix: roughly 12% malicious, with the minority attack types also rarer
CLASS_PROBS = {
    "Benign": 0.880,
    "DoS_Flood": 0.040,
    "Port_Scan": 0.030,
    "Auth_Spoofing": 0.030,
    "Data_Exfiltration": 0.020,
}

def sample_benign(n):
    dev = rng.choice(DEVICE_TYPES, size=n, p=[0.35, 0.15, 0.20, 0.15, 0.15])
    proto = rng.choice(PROTOCOLS, size=n, p=[0.30, 0.15, 0.30, 0.20, 0.05])
    # device-dependent packet sizes (cameras send large frames)
    base_size = pd.Series(dev).map({"sensor": 120, "gateway": 600, "actuator": 180,
                                    "camera": 1100, "wearable": 250}).values
    pkt_size_mean = np.clip(rng.normal(base_size, base_size * 0.25), 40, 1500)
    pkt_size_std = np.clip(rng.normal(pkt_size_mean * 0.20, 25), 0, 600)
    flow_dur = rng.lognormal(mean=4.2, sigma=0.7, size=n)            # ms
    pkts_per_s = rng.gamma(shape=2.0, scale=12.0, size=n)
    inter_arrival = np.clip(1000.0 / np.maximum(pkts_per_s, 0.5) + rng.normal(0, 5, n), 0.1, None)
    bytes_per_s = pkts_per_s * pkt_size_mean * rng.uniform(0.9, 1.1, n)
    dst_port = rng.choice([80, 443, 1883, 8883, 5683, 8080, 22], size=n,
                          p=[0.15, 0.30, 0.20, 0.15, 0.10, 0.05, 0.05])
    tx_latency = rng.lognormal(mean=5.6, sigma=0.45, size=n)         # ms, blockchain tx confirmation
    tx_per_min = rng.poisson(lam=3.0, size=n).astype(float)
    auth_fail = rng.poisson(lam=0.15, size=n).astype(float)
    encrypted = rng.binomial(1, 0.88, size=n)
    return dict(device_type=dev, protocol=proto, pkt_size_mean=pkt_size_mean,
                pkt_size_std=pkt_size_std, flow_duration_ms=flow_dur, pkts_per_s=pkts_per_s,
                inter_arrival_ms=inter_arrival, bytes_per_s=bytes_per_s, dst_port=dst_port,
                tx_confirm_latency_ms=tx_latency, tx_per_min=tx_per_min,
                auth_failures=auth_fail, encrypted=encrypted)

def apply_attack(d, kind, idx):
    m = len(idx)
    if kind == "DoS_Flood":
        d["pkts_per_s"][idx] = rng.gamma(4.0, 60.0, m)
        d["bytes_per_s"][idx] = d["pkts_per_s"][idx] * rng.uniform(40, 200, m)
        d["pkt_size_mean"][idx] = rng.normal(90, 30, m).clip(40, 1500)
        d["flow_duration_ms"][idx] = rng.lognormal(3.0, 0.8, m)
        d["inter_arrival_ms"][idx] = 1000.0 / np.maximum(d["pkts_per_s"][idx], 0.5)
        d["tx_confirm_latency_ms"][idx] *= rng.uniform(1.4, 3.0, m)  # network congestion slows confirmations
        d["protocol"][idx] = rng.choice(["UDP", "TCP", "ICMP"], m, p=[0.6, 0.3, 0.1])
        d["encrypted"][idx] = rng.binomial(1, 0.25, m)
    elif kind == "Port_Scan":
        d["dst_port"][idx] = rng.integers(1, 65535, m)
        d["flow_duration_ms"][idx] = rng.lognormal(1.5, 0.5, m)
        d["pkts_per_s"][idx] = rng.gamma(2.0, 8.0, m)
        d["pkt_size_mean"][idx] = rng.normal(60, 8, m).clip(40, 200)
        d["protocol"][idx] = rng.choice(["TCP", "UDP"], m, p=[0.85, 0.15])
        d["encrypted"][idx] = rng.binomial(1, 0.05, m)
        d["bytes_per_s"][idx] = d["pkts_per_s"][idx] * d["pkt_size_mean"][idx]
    elif kind == "Auth_Spoofing":
        d["auth_failures"][idx] = rng.poisson(3.0, m).astype(float)
        d["tx_per_min"][idx] = rng.poisson(9.0, m).astype(float)
        d["tx_confirm_latency_ms"][idx] *= rng.uniform(1.1, 1.8, m)
        d["protocol"][idx] = rng.choice(["MQTT", "CoAP", "TCP"], m, p=[0.5, 0.3, 0.2])
        d["encrypted"][idx] = rng.binomial(1, 0.55, m)
    elif kind == "Data_Exfiltration":
        d["pkt_size_mean"][idx] = rng.normal(900, 250, m).clip(200, 1500)
        d["flow_duration_ms"][idx] = rng.lognormal(7.0, 0.6, m)     # long-lived flows
        d["bytes_per_s"][idx] = d["pkts_per_s"][idx] * d["pkt_size_mean"][idx] * rng.uniform(1.5, 3.0, m)
        d["dst_port"][idx] = rng.choice([443, 8080, 22], m, p=[0.6, 0.3, 0.1])
        d["protocol"][idx] = rng.choice(["TCP", "MQTT"], m, p=[0.8, 0.2])
        d["encrypted"][idx] = rng.binomial(1, 0.7, m)
        d["tx_per_min"][idx] = rng.poisson(1.5, m).astype(float)
    return d

def main():
    labels = rng.choice(list(CLASS_PROBS.keys()), size=N, p=list(CLASS_PROBS.values()))
    d = sample_benign(N)
    for kind in ATTACK_TYPES:
        idx = np.where(labels == kind)[0]
        if len(idx):
            d = apply_attack(d, kind, idx)

    df = pd.DataFrame(d)
    df["pkt_size_mean"] = df["pkt_size_mean"].round(2)
    df["pkt_size_std"] = df["pkt_size_std"].round(2)
    df["flow_duration_ms"] = df["flow_duration_ms"].round(3)
    df["pkts_per_s"] = df["pkts_per_s"].round(3)
    df["inter_arrival_ms"] = df["inter_arrival_ms"].round(3)
    df["bytes_per_s"] = df["bytes_per_s"].round(2)
    df["tx_confirm_latency_ms"] = df["tx_confirm_latency_ms"].round(3)
    df["dst_port"] = df["dst_port"].astype(int)
    df["tx_per_min"] = df["tx_per_min"].astype(int)
    df["auth_failures"] = df["auth_failures"].astype(int)
    df["encrypted"] = df["encrypted"].astype(int)

    # Realistic data-quality issues: missing values in the blockchain-side measurement
    miss_idx = rng.choice(N, size=int(0.04 * N), replace=False)
    df.loc[miss_idx[: len(miss_idx) // 2], "tx_confirm_latency_ms"] = np.nan
    df.loc[miss_idx[len(miss_idx) // 2:], "flow_duration_ms"] = np.nan
    # Sensor glitches: a few extreme values (outliers)
    out_idx = rng.choice(N, size=40, replace=False)
    df.loc[out_idx, "bytes_per_s"] = df.loc[out_idx, "bytes_per_s"] * 50

    df["label"] = np.where(labels == "Benign", "Benign", "Malicious")
    df["attack_type"] = labels
    df.insert(0, "flow_id", [f"F{i:06d}" for i in range(N)])
    df.to_csv("blockchain_iot_flows.csv", index=False)
    print(df.shape)
    print(df["attack_type"].value_counts())
    print(df.isna().sum()[df.isna().sum() > 0])

if __name__ == "__main__":
    main()
