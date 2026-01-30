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
    human_player = body.get('human_player', 0)
    game_id, session = sessions.create_session(seed=seed, human_player=human_player)
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

    payload = session.step(action)
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
