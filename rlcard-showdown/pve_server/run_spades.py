import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from spades_env import SpadesSessionManager

app = Flask(__name__)
CORS(app)

sessions = SpadesSessionManager()


@app.route('/reset', methods=['POST'])
def reset():
    body = request.get_json(force=True, silent=True) or {}
    seed = body.get('seed')
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            return jsonify({'error': 'seed must be an integer'}), 400

    human_player = body.get('human_player', 0)
    try:
        human_player = int(human_player)
    except (TypeError, ValueError):
        return jsonify({'error': 'human_player must be an integer in [0, 3]'}), 400
    if human_player < 0 or human_player > 3:
        return jsonify({'error': 'human_player must be an integer in [0, 3]'}), 400

    ai_checkpoint = body.get('ai_checkpoint')
    ai_checkpoint_team0 = body.get('ai_checkpoint_team0')
    ai_checkpoint_team1 = body.get('ai_checkpoint_team1')
    enable_blind_nil = bool(body.get('game_enable_blind_nil', True))

    has_all = bool(ai_checkpoint)
    has_team0 = bool(ai_checkpoint_team0)
    has_team1 = bool(ai_checkpoint_team1)
    if not has_all and not (has_team0 and has_team1):
        return jsonify({'error': 'Provide ai_checkpoint, or provide both ai_checkpoint_team0 and ai_checkpoint_team1.'}), 400

    # Resolve relative checkpoint path from repo root (find folder containing experiments/)
    def resolve_checkpoint_path(path):
        if not path:
            return None
        if os.path.isabs(path):
            return os.path.abspath(path)
        search_dir = os.path.abspath(os.path.dirname(__file__))
        repo_root = None
        for _ in range(6):
            if os.path.exists(os.path.join(search_dir, 'experiments')):
                repo_root = search_dir
                break
            search_dir = os.path.dirname(search_dir)
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        return os.path.abspath(os.path.join(repo_root, path))

    ai_checkpoint = resolve_checkpoint_path(ai_checkpoint)
    ai_checkpoint_team0 = resolve_checkpoint_path(ai_checkpoint_team0)
    ai_checkpoint_team1 = resolve_checkpoint_path(ai_checkpoint_team1)

    if ai_checkpoint:
        if not os.path.exists(ai_checkpoint):
            return jsonify({'error': f'ai_checkpoint not found: {ai_checkpoint}'}), 400
    if ai_checkpoint_team0:
        if not os.path.exists(ai_checkpoint_team0):
            return jsonify({'error': f'ai_checkpoint_team0 not found: {ai_checkpoint_team0}'}), 400
    if ai_checkpoint_team1:
        if not os.path.exists(ai_checkpoint_team1):
            return jsonify({'error': f'ai_checkpoint_team1 not found: {ai_checkpoint_team1}'}), 400

    try:
        game_id, session = sessions.create_session(
            seed=seed,
            human_player=human_player,
            ai_checkpoint=ai_checkpoint,
            ai_checkpoint_team0=ai_checkpoint_team0,
            ai_checkpoint_team1=ai_checkpoint_team1,
            enable_blind_nil=enable_blind_nil,
        )
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    payload = session.reset()
    payload['game_id'] = game_id
    return jsonify(payload)


@app.route('/step', methods=['POST'])
def step():
    body = request.get_json(force=True, silent=True) or {}
    game_id = body.get('game_id')
    action = body.get('action')
    if game_id is None:
        return jsonify({'error': 'game_id is required'}), 400
    if action is None:
        return jsonify({'error': 'action is required'}), 400

    try:
        action = int(action)
    except (TypeError, ValueError):
        return jsonify({'error': 'action must be an integer'}), 400

    session = sessions.get(game_id)
    if session is None:
        return jsonify({'error': 'invalid game_id'}), 404

    try:
        payload = session.step(action)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 409

    payload['game_id'] = game_id
    return jsonify(payload)


@app.route('/state', methods=['GET'])
def state():
    game_id = request.args.get('game_id')
    if not game_id:
        return jsonify({'error': 'game_id is required'}), 400
    session = sessions.get(game_id)
    if session is None:
        return jsonify({'error': 'invalid game_id'}), 404
    payload = session._build_response()
    payload['game_id'] = game_id
    return jsonify(payload)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
