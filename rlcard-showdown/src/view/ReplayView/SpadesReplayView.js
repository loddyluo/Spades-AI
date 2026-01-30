import Button from '@material-ui/core/Button';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogTitle from '@material-ui/core/DialogTitle';
import LinearProgress from '@material-ui/core/LinearProgress';
import Paper from '@material-ui/core/Paper';
import Slider from '@material-ui/core/Slider';
import PauseCircleOutlineRoundedIcon from '@material-ui/icons/PauseCircleOutlineRounded';
import PlayArrowRoundedIcon from '@material-ui/icons/PlayArrowRounded';
import ReplayRoundedIcon from '@material-ui/icons/ReplayRounded';
import axios from 'axios';
import { Layout, Loading, Message } from 'element-react';
import qs from 'query-string';
import React from 'react';
import '../../assets/gameview.scss';
import { SpadesGameBoard } from '../../components/GameBoard';
import { deepCopy, spadesDeckIndex } from '../../utils';
import { apiUrl } from '../../utils/config';

class SpadesReplayView extends React.Component {
    constructor(props) {
        super(props);

        const mainViewerId = 0;
        this.initConsiderationTime = 2000;
        this.considerationTimeDeduction = 200;
        this.gameStateTimeout = null;
        this.moveHistory = [];
        this.gameStateHistory = [];
        this.result = null;
        this.initGameState = {
            gameStatus: 'ready',
            playerInfo: [],
            hands: [[], [], [], []],
            bids: [null, null, null, null],
            tricksWon: [0, 0, 0, 0],
            currentTrick: [null, null, null, null],
            spadesBroken: 0,
            mainViewerId: mainViewerId,
            currentPlayer: null,
            phase: 'bidding',
            turn: 0,
            considerationTime: this.initConsiderationTime,
            completedPercent: 0,
        };

        this.state = {
            gameInfo: this.initGameState,
            gameSpeed: 0,
            gameEndDialog: false,
            gameEndDialogText: '',
            fullScreenLoading: false,
        };
    }

    generateNewState() {
        let gameInfo = deepCopy(this.state.gameInfo);
        if (gameInfo.turn >= this.moveHistory.length) return gameInfo;

        const move = this.moveHistory[gameInfo.turn];
        const playerIdx = move.current_player;

        if (move.phase === 'play' && move.action_str && spadesDeckIndex[move.action_str] !== undefined) {
            const remained = gameInfo.hands[playerIdx].filter((card) => card !== move.action_str);
            gameInfo.hands[playerIdx] = remained.length === gameInfo.hands[playerIdx].length
                ? gameInfo.hands[playerIdx]
                : remained;
        }

        if (move.obs) {
            gameInfo.bids = move.obs.bids || gameInfo.bids;
            gameInfo.tricksWon = move.obs.tricks_won || gameInfo.tricksWon;
            gameInfo.spadesBroken = move.obs.spades_broken || 0;
            gameInfo.currentTrick = move.obs.current_trick || [null, null, null, null];
        }

        gameInfo.currentPlayer = playerIdx;
        gameInfo.phase = move.phase;
        gameInfo.turn += 1;
        gameInfo.considerationTime = this.initConsiderationTime;
        gameInfo.completedPercent = Math.min(100, gameInfo.completedPercent + 100.0 / this.moveHistory.length);

        if (gameInfo.turn === this.moveHistory.length) {
            gameInfo.gameStatus = 'over';
            this.setState({ gameInfo: gameInfo });
            const resultText = this.result
                ? `Team0: ${this.result.team_scores[0]} | Team1: ${this.result.team_scores[1]}`
                : 'Game Over';
            setTimeout(() => {
                this.setState({ gameEndDialog: true, gameEndDialogText: resultText });
            }, 200);
            return gameInfo;
        }

        if (gameInfo.turn === this.gameStateHistory.length) {
            this.gameStateHistory.push(gameInfo);
        }
        return gameInfo;
    }

    gameStateTimer() {
        this.gameStateTimeout = setTimeout(() => {
            let currentConsiderationTime = this.state.gameInfo.considerationTime;
            if (currentConsiderationTime > 0) {
                currentConsiderationTime -= this.considerationTimeDeduction * Math.pow(2, this.state.gameSpeed);
                currentConsiderationTime = currentConsiderationTime < 0 ? 0 : currentConsiderationTime;
                let gameInfo = deepCopy(this.state.gameInfo);
                gameInfo.considerationTime = currentConsiderationTime;
                this.setState({ gameInfo: gameInfo });
                this.gameStateTimer();
            } else {
                let gameInfo = this.generateNewState();
                if (gameInfo.gameStatus === 'over') return;
                gameInfo.gameStatus = 'playing';
                this.setState({ gameInfo: gameInfo });
            }
        }, this.considerationTimeDeduction);
    }

    startReplay() {
        const { name, agent0, agent1, index } = qs.parse(window.location.search);
        const requestUrl = `${apiUrl}/tournament/replay?name=${name}&agent0=${agent0}&agent1=${agent1}&index=${index}`;
        this.setState({ fullScreenLoading: true });
        axios
            .get(requestUrl)
            .then((res) => {
                res = res.data;
                if (typeof res === 'string') res = JSON.parse(res.replaceAll("'", '"').replaceAll('None', 'null'));

                if (!res.states || res.states.length === 0) {
                    Message({ message: 'Empty replay data', type: 'error', showClose: true });
                    this.setState({ fullScreenLoading: false });
                    return;
                }

                this.moveHistory = res.states;
                this.result = res.result || null;

                let gameInfo = deepCopy(this.initGameState);
                gameInfo.gameStatus = 'playing';
                gameInfo.playerInfo = res.playerInfo || [];
                gameInfo.hands = res.initialHands || [[], [], [], []];
                gameInfo.currentPlayer = res.states[0].current_player;
                gameInfo.phase = res.states[0].phase;
                if (res.states[0].obs) {
                    gameInfo.bids = res.states[0].obs.bids || gameInfo.bids;
                    gameInfo.tricksWon = res.states[0].obs.tricks_won || gameInfo.tricksWon;
                    gameInfo.currentTrick = res.states[0].obs.current_trick || gameInfo.currentTrick;
                }

                if (this.gameStateHistory.length === 0) this.gameStateHistory.push(gameInfo);

                this.setState({ gameInfo: gameInfo, fullScreenLoading: false }, () => {
                    if (this.gameStateTimeout) {
                        window.clearTimeout(this.gameStateTimeout);
                        this.gameStateTimeout = null;
                    }
                    this.gameStateTimer();
                });
            })
            .catch(() => {
                this.setState({ fullScreenLoading: false });
                Message({ message: 'Error in getting replay data', type: 'error', showClose: true });
            });
    }

    pauseReplay() {
        if (this.gameStateTimeout) {
            window.clearTimeout(this.gameStateTimeout);
            this.gameStateTimeout = null;
        }
        let gameInfo = deepCopy(this.state.gameInfo);
        gameInfo.gameStatus = 'paused';
        this.setState({ gameInfo: gameInfo });
    }

    resumeReplay() {
        this.gameStateTimer();
        let gameInfo = deepCopy(this.state.gameInfo);
        gameInfo.gameStatus = 'playing';
        this.setState({ gameInfo: gameInfo });
    }

    changeGameSpeed(newVal) {
        this.setState({ gameSpeed: newVal });
    }

    gameStatusButton(status) {
        switch (status) {
            case 'ready':
                return (
                    <Button
                        className={'status-button'}
                        variant={'contained'}
                        startIcon={<PlayArrowRoundedIcon />}
                        color="primary"
                        onClick={() => this.startReplay()}
                    >
                        Start
                    </Button>
                );
            case 'playing':
                return (
                    <Button
                        className={'status-button'}
                        variant={'contained'}
                        startIcon={<PauseCircleOutlineRoundedIcon />}
                        color="secondary"
                        onClick={() => this.pauseReplay()}
                    >
                        Pause
                    </Button>
                );
            case 'paused':
                return (
                    <Button
                        className={'status-button'}
                        variant={'contained'}
                        startIcon={<PlayArrowRoundedIcon />}
                        color="primary"
                        onClick={() => this.resumeReplay()}
                    >
                        Resume
                    </Button>
                );
            case 'over':
                return (
                    <Button
                        className={'status-button'}
                        variant={'contained'}
                        startIcon={<ReplayRoundedIcon />}
                        color="primary"
                        onClick={() => this.startReplay()}
                    >
                        Restart
                    </Button>
                );
            default:
                return null;
        }
    }

    render() {
        return (
            <Layout.Row className={'gameview'}>
                <Loading fullscreen={true} loading={this.state.fullScreenLoading} />
                <Layout.Col span={24} className={'gameboard'}>
                    <SpadesGameBoard
                        playerInfo={this.state.gameInfo.playerInfo}
                        hands={this.state.gameInfo.hands}
                        bids={this.state.gameInfo.bids}
                        tricksWon={this.state.gameInfo.tricksWon}
                        currentTrick={this.state.gameInfo.currentTrick}
                        spadesBroken={this.state.gameInfo.spadesBroken}
                        currentPlayer={this.state.gameInfo.currentPlayer}
                        phase={this.state.gameInfo.phase}
                        mainPlayerId={this.state.gameInfo.mainViewerId}
                        gamePlayable={false}
                        hideOpponentHands={false}
                    />
                </Layout.Col>
                <Layout.Col span={24} className={'control'}>
                    <Paper elevation={3} className={'control-panel'}>
                        <div className={'progress-bar'}>
                            <LinearProgress variant="determinate" value={this.state.gameInfo.completedPercent} />
                        </div>
                        <div className={'control-panel-row'}>
                            {this.gameStatusButton(this.state.gameInfo.gameStatus)}
                            <div className={'game-speed'}>
                                <span>Speed</span>
                                <Slider
                                    defaultValue={0}
                                    step={1}
                                    min={0}
                                    max={2}
                                    valueLabelDisplay="auto"
                                    onChange={(e, val) => this.changeGameSpeed(val)}
                                />
                            </div>
                        </div>
                    </Paper>
                </Layout.Col>

                <Dialog open={this.state.gameEndDialog} aria-labelledby="form-dialog-title">
                    <DialogTitle id="form-dialog-title">Game Over</DialogTitle>
                    <DialogContent>
                        <DialogContentText>{this.state.gameEndDialogText}</DialogContentText>
                    </DialogContent>
                    <DialogActions>
                        <Button
                            onClick={() => {
                                this.setState({ gameEndDialog: false });
                            }}
                            color="primary"
                            variant="contained"
                        >
                            Close
                        </Button>
                    </DialogActions>
                </Dialog>
            </Layout.Row>
        );
    }
}

export default SpadesReplayView;
