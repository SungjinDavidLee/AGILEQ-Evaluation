"""
Runs eval.py with BypassRouteEnv swapped in for CarlaRouteEnv.

  python run_bypass_demo.py --run_name iter5 --config configs/iter5.json
  python run_bypass_demo.py --run_name main --config configs/main_module.json
  python run_bypass_demo.py --run_name off --config configs/iter5.json --disable_bypass

Logs go to logs/<run_name>/, the eval video to videos/<run_name>/eval_<town>/.
"""
import os
import sys
import json
import runpy
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.abspath(os.path.join(HERE, '..'))
MODEL = os.path.join(EVAL_DIR, 'ieee_results', 'AGILE-Q-C2', 'v1', 'last_model.zip')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_name', required=True)
    ap.add_argument('--config', default=os.path.join(HERE, 'configs', 'iter5.json'))
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--episodes', type=int, default=None)
    ap.add_argument('--disable_bypass', action='store_true')
    ap.add_argument('--no_video', action='store_true')
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    args = ap.parse_args()

    cfg = json.load(open(os.path.abspath(args.config)))
    if args.episodes is not None:
        cfg['n_episodes'] = args.episodes
    if args.disable_bypass:
        cfg['enable_bypass'] = False

    log_dir = os.path.join(HERE, 'logs', args.run_name)
    video_dir = os.path.join(HERE, 'videos', args.run_name)
    work = os.path.join(HERE, 'logs', '_workdir')
    for d in (log_dir, video_dir, work, os.path.join(work, 'results', 'terminal_reason')):
        os.makedirs(d, exist_ok=True)
    for name in ['BEV', 'best_bev_model.pth', 'best_traffic_model.pth']:
        dst = os.path.join(work, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(EVAL_DIR, name), dst)
    # eval.py writes its video next to the model file
    model_link = os.path.join(video_dir, 'best_model.zip')
    if os.path.lexists(model_link):
        os.remove(model_link)
    os.symlink(os.path.abspath(args.model), model_link)
    with open(os.path.join(log_dir, 'effective_config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    os.chdir(work)
    sys.path.insert(0, EVAL_DIR)
    sys.path.insert(0, HERE)

    import config as eval_config
    if cfg.get('c2_state'):
        eval_config.CONFIGS['crossq']['state'] = list(cfg['c2_state'])

    from carla_env.bypass.plan_bypass import PlanBypass
    main = cfg.get('main_module', {})
    PlanBypass.ENABLED = bool(main.get('enabled', False))
    PlanBypass.OVERRIDE_ACTIONS = bool(main.get('override_actions', False))
    for key, attr in (('trigger_points', 'TRIGGER_POINTS'), ('sample_m', 'SAMPLE_M'), ('method', 'METHOD')):
        if key in main:
            setattr(PlanBypass, attr, main[key])
    if PlanBypass.ENABLED:
        cfg['enable_bypass'] = False

    import carla_env.envs.carla_FixedRoute_env as fr
    import bypass_env
    bypass_env.OT_CFG = cfg
    bypass_env.LOG_DIR = log_dir
    fr.CarlaRouteEnv = bypass_env.BypassRouteEnv

    sc = cfg['scenario']
    argv = ['eval.py', '--model', model_link, '--config', 'crossq', '--town', sc['town'],
            '--start', ','.join(str(v) for v in sc['start_xyzyaw']), '--dest', ','.join(str(v) for v in sc['dest_xyz']),
            '--fps', '15', '--host', args.host, '--port', str(args.port)]
    if args.no_video:
        argv.append('--no_record_video')
    sys.argv = argv
    print('[bypass_demo] running', ' '.join(argv), flush=True)
    runpy.run_path(os.path.join(EVAL_DIR, 'eval.py'), run_name='__main__')


if __name__ == '__main__':
    main()
