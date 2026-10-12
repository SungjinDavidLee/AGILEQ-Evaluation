"""
Splits eval.py's AVI into one MP4 per episode and writes a keyframe sheet.

  python make_videos.py --run_name iter5
"""
import os
import argparse

import cv2
import numpy as np
import pandas as pd

DEMO = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_name', required=True)
    ap.add_argument('--town', default='town04')
    args = ap.parse_args()
    vdir = os.path.join(DEMO, 'videos', args.run_name)
    avi = os.path.join(vdir, f'eval_{args.town}', 'best_model_eval.avi')
    steps = pd.read_csv(os.path.join(DEMO, 'logs', args.run_name, 'steps.csv'))
    steps = steps[(steps.episode >= 1) & (steps.action_none == 0)]
    counts = steps.groupby('episode').size().to_dict()

    cap = cv2.VideoCapture(avi)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'{avi}: {frames_total} frames, per-episode steps {counts}')
    keyframes = []
    first = True
    for ep in sorted(counts):
        n = counts[ep] + (1 if first else 0)
        first = False
        rows = steps[steps.episode == ep].reset_index(drop=True)
        writer = None
        picks = set()
        f_rows = rows.index[rows.path_mode == 'F'].tolist()
        trig = rows.index[rows.trigger_active == 1].tolist()
        # keyframes: first trigger, first F, last F, last frame
        for idx in [trig[0] if trig else None, f_rows[0] if f_rows else None, f_rows[-1] if f_rows else None, len(rows) - 1]:
            if idx is not None:
                picks.add(idx + (1 if ep == min(counts) else 0))
        for i in range(n):
            ok, fr = cap.read()
            if not ok:
                break
            if writer is None:
                h, w = fr.shape[:2]
                writer = cv2.VideoWriter(os.path.join(vdir, f'episode_{ep}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            writer.write(fr)
            if i in picks:
                lab = f'ep{ep} step{i}'
                small = cv2.resize(fr, (fr.shape[1] // 2, fr.shape[0] // 2))
                cv2.putText(small, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                keyframes.append((ep, small))
        if writer is not None:
            writer.release()
        print(f'  episode {ep}: {n} frames -> episode_{ep}.mp4')
    cap.release()
    if keyframes:
        by_ep = {}
        for ep, im in keyframes:
            by_ep.setdefault(ep, []).append(im)
        w = max(len(v) for v in by_ep.values())
        rows = []
        for ep in sorted(by_ep):
            ims = by_ep[ep] + [np.zeros_like(by_ep[ep][0])] * (w - len(by_ep[ep]))
            rows.append(np.hstack(ims))
        cv2.imwrite(os.path.join(vdir, 'keyframes.png'), np.vstack(rows))


if __name__ == '__main__':
    main()
