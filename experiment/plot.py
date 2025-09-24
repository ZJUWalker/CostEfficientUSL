import os
import json
from usl.utils.usl_gantt_plot import plot_gantt_per_batch, merge_cs_json
import time


def plot(dir='log/profile'):
    client_fp = os.path.join(dir, 'client.json')
    server_fp = os.path.join(dir, 'server.json')
    with open(client_fp, 'r') as f:
        client_data = json.load(f)
    with open(server_fp, 'r') as f:
        server_data = json.load(f)
    merged_data = merge_cs_json(server_data, client_data, save=True, save_fp=os.path.join(dir, 'merged.json'))
    plot_gantt_per_batch(merged_data, fp=f'log/img/gantt_{time.time()}.png')


if __name__ == '__main__':
    plot()
