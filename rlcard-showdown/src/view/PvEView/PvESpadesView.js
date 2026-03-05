import Button from '@material-ui/core/Button';
import axios from 'axios';
import React, { useCallback, useEffect, useState } from 'react';
import '../../assets/gameview.scss';
import { SpadesGameBoard } from '../../components/GameBoard';
import { spadesActionIdToLabel, spadesCardToActionId, sortSpadesCards } from '../../utils';
import { spadesDemoUrl } from '../../utils/config';

const defaultPlayerInfo = [
    { id: 0, index: 0, agentInfo: { name: 'You' } },
    { id: 1, index: 1, agentInfo: { name: 'AI-1' } },
    { id: 2, index: 2, agentInfo: { name: 'AI-2' } },
    { id: 3, index: 3, agentInfo: { name: 'AI-3' } },
];

function PvESpadesView() {
    const [gameId, setGameId] = useState(null);
    const [trickLog, setTrickLog] = useState([]);
    const [errorMessage, setErrorMessage] = useState('');
    const [gameState, setGameState] = useState({
        phase: 'bidding',
        obs: {
            hand: [],
            bids: [null, null, null, null],
            bid_types: [null, null, null, null],
            tricks_won: [0, 0, 0, 0],
            spades_broken: 0,
            current_trick: [null, null, null, null],
        },
        legal_actions: [],
        current_player: 0,
        terminal: false,
        result: null,
        hand_sizes: [13, 13, 13, 13],
        trick: null,
    });

    const [checkpointPath, setCheckpointPath] = useState('experiments/spades_selfplay_dqn_0.5_12h/checkpoint_dqn.pt');
    const [checkpointTeam0, setCheckpointTeam0] = useState('');
    const [checkpointTeam1, setCheckpointTeam1] = useState('');
    const [enableBlindNil, setEnableBlindNil] = useState(true);
    const hasAllSeatsCheckpoint = checkpointPath.trim().length > 0;
    const hasTeamCheckpoints = checkpointTeam0.trim().length > 0 && checkpointTeam1.trim().length > 0;
    const isCheckpointValid = hasAllSeatsCheckpoint || hasTeamCheckpoints;

    const appendTrickFromResponse = useCallback((respData) => {
        if (!respData?.trick) {
            return;
        }
        const trick = respData.trick;
        const lead = trick.lead;
        const cards = trick.cards || [];
        const map = {};
        if (lead !== null && lead !== undefined) {
            cards.forEach((card, idx) => {
                map[(lead + idx) % 4] = card;
            });
        }
        const display = [0, 1, 2, 3].map((pid) => `P${pid}:${map[pid] || '_'}`).join(' ');
        setTrickLog((prev) => {
            if (prev.length > 0 && prev[0].id === trick.trick_id) {
                return prev;
            }
            return [{
                id: trick.trick_id,
                display,
                winner: trick.winner,
            }, ...prev];
        });
    }, []);

    const resetGame = useCallback(async () => {
        if (!isCheckpointValid) {
            setErrorMessage('Checkpoint path is required: fill AI Checkpoint or both Team0/Team1 checkpoints.');
            return;
        }
        try {
            setErrorMessage('');
            const res = await axios.post(`${spadesDemoUrl}/reset`, {
                game: 'spades',
                human_player: 0,
                ai_checkpoint: hasAllSeatsCheckpoint ? checkpointPath.trim() : null,
                ai_checkpoint_team0: checkpointTeam0.trim() ? checkpointTeam0.trim() : null,
                ai_checkpoint_team1: checkpointTeam1.trim() ? checkpointTeam1.trim() : null,
                game_enable_blind_nil: enableBlindNil,
            });
            setGameId(res.data.game_id);
            setGameState(res.data);
            setTrickLog([]);
            appendTrickFromResponse(res.data);
        } catch (err) {
            const message = err?.response?.data?.error || 'Failed to start game. Check checkpoint path and backend logs.';
            setErrorMessage(message);
        }
    }, [
        appendTrickFromResponse,
        checkpointPath,
        checkpointTeam0,
        checkpointTeam1,
        enableBlindNil,
        hasAllSeatsCheckpoint,
        isCheckpointValid,
    ]);

    useEffect(() => {
        resetGame();
    }, [resetGame]);

    const stepGame = async (actionId) => {
        if (!gameId) return;
        try {
            setErrorMessage('');
            const res = await axios.post(`${spadesDemoUrl}/step`, {
                game_id: gameId,
                action: actionId,
            });
            setGameState(res.data);
            appendTrickFromResponse(res.data);
        } catch (err) {
            const message = err?.response?.data?.error || 'Failed to play action.';
            setErrorMessage(message);
        }
    };

    const legalActionSet = new Set(
        (gameState.legal_actions || [])
            .map((id) => spadesActionIdToLabel(id))
            .filter((label) => label && label.length === 2),
    );

    const legalCards = sortSpadesCards(
        (gameState.legal_actions || [])
            .map((id) => spadesActionIdToLabel(id))
            .filter((label) => label && label.length === 2),
    );

    const lastTrick = gameState.trick
        ? {
              ...gameState.trick,
              display: (() => {
                  const lead = gameState.trick.lead;
                  const cards = gameState.trick.cards || [];
                  const map = {};
                  if (lead !== null && lead !== undefined) {
                      cards.forEach((card, idx) => {
                          map[(lead + idx) % 4] = card;
                      });
                  }
                  return [0, 1, 2, 3].map((pid) => `P${pid}:${map[pid] || '_'}`).join(' ');
              })(),
          }
        : null;

    const teamScores = gameState.result?.team_scores || [0, 0];
    const leadingTeam = teamScores[0] === teamScores[1] ? null : (teamScores[0] > teamScores[1] ? 0 : 1);

    const opponentHands = (gameState.hand_sizes || [0, 0, 0, 0]).map((size, idx) => {
        if (idx === 0) return gameState.obs.hand;
        return Array.from({ length: size }).fill('XX');
    });

    const biddingActions = (gameState.legal_actions || [])
        .map((id) => ({ id, label: spadesActionIdToLabel(id) }))
        .filter((item) => item.label && (item.label === 'pass' || item.label === 'blind_nil' || item.label === 'nil' || item.label.startsWith('bid_')));

    return (
        <div className="spades-pve-root">
            <div className="spades-action-buttons">
                <div className="spades-checkpoint">
                    <label className="spades-checkpoint-label">AI Checkpoint (All Seats)</label>
                    <input
                        type="text"
                        value={checkpointPath}
                        onChange={(e) => setCheckpointPath(e.target.value)}
                        className="spades-checkpoint-input"
                    />
                    <div className={`spades-checkpoint-help ${!isCheckpointValid ? 'is-invalid' : ''}`}>
                        {isCheckpointValid
                            ? 'Example: experiments/spades_selfplay_dqn_0.5_12h/checkpoint_dqn.pt'
                            : 'Fill AI Checkpoint, or both Team0 and Team1 checkpoints.'}
                    </div>
                </div>
                <div className="spades-checkpoint">
                    <label className="spades-checkpoint-label">Team0 Checkpoint (P0,P2)</label>
                    <input
                        type="text"
                        value={checkpointTeam0}
                        onChange={(e) => setCheckpointTeam0(e.target.value)}
                        className="spades-checkpoint-input"
                    />
                    <div className="spades-checkpoint-help">
                        Optional override for Team0 (auto-detects DQN/NFSP/DMC/CFR).
                    </div>
                </div>
                <div className="spades-checkpoint">
                    <label className="spades-checkpoint-label">Team1 Checkpoint (P1,P3)</label>
                    <input
                        type="text"
                        value={checkpointTeam1}
                        onChange={(e) => setCheckpointTeam1(e.target.value)}
                        className="spades-checkpoint-input"
                    />
                    <div className="spades-checkpoint-help">
                        Optional override for Team1 (auto-detects DQN/NFSP/DMC/CFR).
                    </div>
                </div>
                <div className="spades-checkpoint">
                    <label className="spades-checkpoint-label">Blind-Nil Phase</label>
                    <div className="spades-checkpoint-help">
                        <input
                            type="checkbox"
                            checked={enableBlindNil}
                            onChange={(e) => setEnableBlindNil(e.target.checked)}
                        />
                        <span style={{ marginLeft: '8px' }}>
                            Enable blind-nil phase before bidding
                        </span>
                    </div>
                </div>
                <Button variant="contained" color="primary" onClick={resetGame} disabled={!isCheckpointValid}>
                    Start / Reset
                </Button>
            </div>

            {errorMessage ? (
                <div className="spades-action-buttons">
                    <div className="spades-checkpoint-help is-invalid">{errorMessage}</div>
                </div>
            ) : null}

            <div className="spades-panels-row">
                <div className="spades-panel spades-panel--fixed">
                    <div className="spades-panel-title">Game Status</div>
                    <div className="spades-panel-line">
                        <span className="spades-panel-label">Phase</span>
                        <span className="spades-panel-value">{gameState.phase}</span>
                    </div>
                    <div className="spades-panel-line">
                        <span className="spades-panel-label">Bids</span>
                        <span className="spades-panel-value">
                            {(gameState.obs.bid_types || []).length
                                ? gameState.obs.bid_types.map((t, i) => {
                                      const b = gameState.obs.bids?.[i];
                                      if (b === null || b === undefined) return '-';
                                      if (t === 'blind_nil') return 'Blind Nil';
                                      if (t === 'nil') return 'Nil';
                                      return b;
                                  }).join(', ')
                                : '-'}
                        </span>
                    </div>
                    <div className="spades-panel-line">
                        <span className="spades-panel-label">Tricks</span>
                        <span className="spades-panel-value">{(gameState.obs.tricks_won || []).join(', ')}</span>
                    </div>
                    <div className="spades-panel-line">
                        <span className="spades-panel-label">Spades Broken</span>
                        <span className="spades-panel-value">{gameState.obs.spades_broken ? 'Yes' : 'No'}</span>
                    </div>
                </div>
                <div className="spades-panel spades-panel--fixed">
                    <div className="spades-panel-title">Last Trick & Legal</div>
                    <div className="spades-panel-line">
                        <span className="spades-panel-label">Legal Cards</span>
                        <span className="spades-panel-value">{legalCards.length ? legalCards.join(' ') : '-'}</span>
                    </div>
                    <div className="spades-panel-line">
                        <span className="spades-panel-label">Last Trick</span>
                        <span className="spades-panel-value">{lastTrick ? `${lastTrick.display} | Winner: P${lastTrick.winner}` : '-'}</span>
                    </div>
                </div>
            </div>

            <SpadesGameBoard
                playerInfo={defaultPlayerInfo}
                hands={opponentHands}
                bids={gameState.obs.bids}
                bidTypes={gameState.obs.bid_types}
                tricksWon={gameState.obs.tricks_won}
                handSizes={gameState.hand_sizes}
                currentTrick={gameState.obs.current_trick}
                spadesBroken={gameState.obs.spades_broken}
                currentPlayer={gameState.current_player}
                phase={gameState.phase}
                mainPlayerId={0}
                gamePlayable={!gameState.terminal}
                hideOpponentHands={true}
                legalActionSet={legalActionSet}
                legalCards={legalCards}
                lastTrick={lastTrick}
                onCardClick={(card) => {
                    const actionId = spadesCardToActionId(card);
                    if (actionId !== undefined) stepGame(actionId);
                }}
            />

            {gameState.phase === 'bidding' && !gameState.terminal ? (
                <div className="spades-action-buttons">
                    <div className="spades-bid-hint">Select your bid for this round.</div>
                    {biddingActions.map((action) => (
                        <Button
                            key={`bid-${action.id}`}
                            variant="contained"
                            color="primary"
                            onClick={() => stepGame(action.id)}
                        >
                            {action.label === 'pass'
                                ? 'Pass'
                                : action.label === 'blind_nil'
                                ? 'Blind Nil'
                                : action.label === 'nil'
                                ? 'Nil'
                                : action.label.replace('bid_', 'Bid ')}
                        </Button>
                    ))}
                </div>
            ) : null}

            {gameState.terminal && gameState.result ? (
                <div className="spades-action-buttons">
                    <div className="spades-summary">
                        <div className="spades-summary-scores">
                            <div className={`spades-summary-team ${leadingTeam === 0 ? 'is-leading' : ''}`}>
                                <span className="spades-summary-label">Team0</span>
                                <span className="spades-summary-score">{teamScores[0]}</span>
                            </div>
                            <div className={`spades-summary-team ${leadingTeam === 1 ? 'is-leading' : ''}`}>
                                <span className="spades-summary-label">Team1</span>
                                <span className="spades-summary-score">{teamScores[1]}</span>
                            </div>
                        </div>
                        <div className="spades-summary-grid">
                            {[0, 1, 2, 3].map((pid) => (
                                <div key={`summary-${pid}`}>
                                    <span className="spades-summary-player">P{pid}</span>
                                    <span className="spades-summary-detail">Bid: {gameState.result.bids?.[pid]}</span>
                                    <span className="spades-summary-detail">Tricks: {gameState.result.tricks_won?.[pid]}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                    <Button variant="contained" color="secondary" onClick={resetGame}>Restart</Button>
                </div>
            ) : null}

            <div className="spades-action-buttons">
                <div className="spades-log">
                    <div className="spades-log-title">
                        <strong>Trick Log</strong>
                        <span className="spades-log-count">{trickLog.length}</span>
                    </div>
                    {trickLog.length > 0 ? (
                        <ul>
                            {trickLog.map((item) => (
                                <li key={`trick-log-${item.id}`}>
                                    <div className="spades-log-row">
                                        <span className="spades-log-badge">Trick {item.id}</span>
                                        <span className="spades-log-cards">{item.display}</span>
                                        <span className="spades-log-winner">Winner: P{item.winner}</span>
                                    </div>
                                </li>
                            ))}
                        </ul>
                    ) : (
                        <div className="spades-log-empty">No trick finished yet.</div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default PvESpadesView;
