import Button from '@material-ui/core/Button';
import axios from 'axios';
import React, { useEffect, useState } from 'react';
import '../../assets/gameview.scss';
import { SpadesGameBoard } from '../../components/GameBoard';
import { spadesActionIdToLabel, spadesCardToActionId } from '../../utils';
import { spadesDemoUrl } from '../../utils/config';

const defaultPlayerInfo = [
    { id: 0, index: 0, agentInfo: { name: 'You' } },
    { id: 1, index: 1, agentInfo: { name: 'AI-1' } },
    { id: 2, index: 2, agentInfo: { name: 'AI-2' } },
    { id: 3, index: 3, agentInfo: { name: 'AI-3' } },
];

function PvESpadesView() {
    const [gameId, setGameId] = useState(null);
    const [gameState, setGameState] = useState({
        phase: 'bidding',
        obs: {
            hand: [],
            bids: [null, null, null, null],
            tricks_won: [0, 0, 0, 0],
            spades_broken: 0,
            current_trick: [null, null, null, null],
        },
        legal_actions: [],
        current_player: 0,
        terminal: false,
        result: null,
        hand_sizes: [13, 13, 13, 13],
    });

    const resetGame = async () => {
        const res = await axios.post(`${spadesDemoUrl}/reset`, {
            game: 'spades',
            human_player: 0,
        });
        setGameId(res.data.game_id);
        setGameState(res.data);
    };

    useEffect(() => {
        resetGame();
    }, []);

    const stepGame = async (actionId) => {
        if (!gameId) return;
        const res = await axios.post(`${spadesDemoUrl}/step`, {
            game_id: gameId,
            action: actionId,
        });
        setGameState(res.data);
    };

    const legalActionSet = new Set(
        (gameState.legal_actions || [])
            .map((id) => spadesActionIdToLabel(id))
            .filter((label) => label && label.length === 2),
    );

    const opponentHands = (gameState.hand_sizes || [0, 0, 0, 0]).map((size, idx) => {
        if (idx === 0) return gameState.obs.hand;
        return Array.from({ length: size }).fill('XX');
    });

    const biddingActions = (gameState.legal_actions || [])
        .map((id) => ({ id, label: spadesActionIdToLabel(id) }))
        .filter((item) => item.label && (item.label === 'pass' || item.label === 'blind_nil' || item.label === 'nil' || item.label.startsWith('bid_')));

    return (
        <div>
            <SpadesGameBoard
                playerInfo={defaultPlayerInfo}
                hands={opponentHands}
                bids={gameState.obs.bids}
                tricksWon={gameState.obs.tricks_won}
                currentTrick={gameState.obs.current_trick}
                spadesBroken={gameState.obs.spades_broken}
                currentPlayer={gameState.current_player}
                phase={gameState.phase}
                mainPlayerId={0}
                gamePlayable={!gameState.terminal}
                hideOpponentHands={true}
                legalActionSet={legalActionSet}
                onCardClick={(card) => {
                    const actionId = spadesCardToActionId(card);
                    if (actionId !== undefined) stepGame(actionId);
                }}
            />

            {gameState.phase === 'bidding' && !gameState.terminal ? (
                <div className="spades-action-buttons">
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
                    <span>{`Team0: ${gameState.result.team_scores[0]} | Team1: ${gameState.result.team_scores[1]}`}</span>
                    <Button variant="contained" color="secondary" onClick={resetGame}>Restart</Button>
                </div>
            ) : null}
        </div>
    );
}

export default PvESpadesView;
