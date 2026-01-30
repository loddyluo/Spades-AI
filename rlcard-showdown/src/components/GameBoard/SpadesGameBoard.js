import Avatar from '@material-ui/core/Avatar';
import Chip from '@material-ui/core/Chip';
import React from 'react';
import '../../assets/spades.scss';
import PlaceHolderPlayer from '../../assets/images/Portrait/Player.png';
import { computeHandCardsWidth, translateCardData } from '../../utils';

class SpadesGameBoard extends React.Component {
    computePlayerPortrait(playerId, playerIdx) {
        if (this.props.playerInfo.length > 0) {
            const chipTitle =
                this.props.playerInfo[playerIdx].agentInfo && this.props.playerInfo[playerIdx].agentInfo.name
                    ? ''
                    : 'ID';
            const chipLabel =
                this.props.playerInfo[playerIdx].agentInfo && this.props.playerInfo[playerIdx].agentInfo.name
                    ? this.props.playerInfo[playerIdx].agentInfo.name
                    : playerId;
            return (
                <div>
                    <img src={PlaceHolderPlayer} alt={'Player'} height="60%" width="60%" />
                    <Chip
                        style={{ maxWidth: '135px' }}
                        avatar={chipTitle ? <Avatar>{chipTitle}</Avatar> : undefined}
                        label={chipLabel}
                        color="primary"
                    />
                </div>
            );
        }
        return (
            <div>
                <img src={PlaceHolderPlayer} alt={'Player'} height="60%" width="60%" />
                <Chip avatar={<Avatar>ID</Avatar>} label={playerId} color="primary" />
            </div>
        );
    }

    renderHand(cards, cardSelectable, legalActionSet, onCardClick, hideCards) {
        if (hideCards) {
            return (
                <div className={'playingCards unselectable loose'}>
                    <ul className="hand" style={{ width: computeHandCardsWidth(cards.length, 10) }}>
                        {cards.map((card, idx) => (
                            <li key={`handCard-back-${idx}`}>
                                <span className="card back">*</span>
                            </li>
                        ))}
                    </ul>
                </div>
            );
        }
        return (
            <div className={`playingCards loose ${cardSelectable ? 'selectable' : 'unselectable'}`}>
                <ul className="hand" style={{ width: computeHandCardsWidth(cards.length, 10) }}>
                    {cards.map((card) => {
                        const [rankClass, suitClass, rankText, suitText] = translateCardData(card);
                        const isLegal = legalActionSet ? legalActionSet.has(card) : false;
                        return (
                            <li key={`handCard-${card}`}>
                                <label
                                    className={`card ${rankClass} ${suitClass} ${cardSelectable && isLegal ? 'selected' : ''}`}
                                    onClick={() => {
                                        if (cardSelectable && isLegal && onCardClick) onCardClick(card);
                                    }}
                                >
                                    <span className="rank">{rankText}</span>
                                    <span className="suit">{suitText}</span>
                                </label>
                            </li>
                        );
                    })}
                </ul>
            </div>
        );
    }

    renderTrickCard(card) {
        if (!card) {
            return <div className={'non-card'}>...</div>;
        }
        const [rankClass, suitClass, rankText, suitText] = translateCardData(card);
        return (
            <div className={'playingCards unselectable'}>
                <div className={`card ${rankClass} ${suitClass}`}>
                    <span className="rank">{rankText}</span>
                    <span className="suit">{suitText}</span>
                </div>
            </div>
        );
    }

    renderSeat(positionClass, playerIdx) {
        const player = this.props.playerInfo[playerIdx];
        const playerId = player ? player.id : playerIdx;
        const hand = this.props.hands[playerIdx] || [];
        const bids = this.props.bids || [];
        const tricks = this.props.tricksWon || [];
        const hideCards = this.props.hideOpponentHands && playerIdx !== this.props.mainPlayerId;
        return (
            <div className={`spades-seat ${positionClass}`}>
                {this.computePlayerPortrait(playerId, playerIdx)}
                <div className="spades-bid">Bid: {bids[playerIdx] !== null && bids[playerIdx] !== undefined ? bids[playerIdx] : '-'}</div>
                <div className="spades-trick-count">Tricks: {tricks[playerIdx] !== null && tricks[playerIdx] !== undefined ? tricks[playerIdx] : '-'}</div>
                <div className="spades-hand">
                    {this.renderHand(
                        hand,
                        this.props.gamePlayable && playerIdx === this.props.mainPlayerId && this.props.phase === 'play',
                        this.props.legalActionSet,
                        this.props.onCardClick,
                        hideCards,
                    )}
                </div>
            </div>
        );
    }

    render() {
        const mainPlayerId = this.props.mainPlayerId || 0;
        const topIdx = (mainPlayerId + 2) % 4;
        const leftIdx = (mainPlayerId + 3) % 4;
        const rightIdx = (mainPlayerId + 1) % 4;
        const bottomIdx = mainPlayerId;
        const currentTrick = this.props.currentTrick || [null, null, null, null];

        return (
            <div className="spades-wrapper">
                <div className="spades-table">
                    {this.renderSeat('top', topIdx)}
                    {this.renderSeat('left', leftIdx)}
                    {this.renderSeat('right', rightIdx)}
                    {this.renderSeat('bottom', bottomIdx)}
                    <div className="spades-center">
                        <div className="trick">
                            {currentTrick.map((card, idx) => (
                                <div key={`trick-${idx}`}>{this.renderTrickCard(card)}</div>
                            ))}
                        </div>
                        <div className="spades-info">
                            <div>Phase: {this.props.phase}</div>
                            <div>Current Player: {this.props.currentPlayer}</div>
                            <div>Spades Broken: {this.props.spadesBroken ? 'Yes' : 'No'}</div>
                        </div>
                    </div>
                </div>
            </div>
        );
    }
}

export default SpadesGameBoard;
