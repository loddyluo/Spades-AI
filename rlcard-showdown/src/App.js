import React from 'react';
import { BrowserRouter as Router, Redirect, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import LeaderBoard from './view/LeaderBoard';
import { PvEDoudizhuDemoView, PvESpadesView } from './view/PvEView';
import { DoudizhuReplayView, LeducHoldemReplayView, SpadesReplayView } from './view/ReplayView';

const navbarSubtitleMap = {
    '/leaderboard': '',
    '/replay/doudizhu': 'Doudizhu',
    '/replay/leduc-holdem': "Leduc Hold'em",
    '/replay/spades': 'Spades',
    '/pve/doudizhu-demo': 'Doudizhu PvE Demo',
    '/pve/spades': 'Spades PvE',
};

function App() {
    // todo: add 404 page
    return (
        <Router>
            <Navbar subtitleMap={navbarSubtitleMap} />
            <div style={{ marginTop: '75px' }}>
                <Route exact path="/">
                    <Redirect to="/leaderboard?type=game&name=leduc-holdem" />
                    {/* <Redirect to="/pve/doudizhu-demo" /> */}
                </Route>
                <Route path="/leaderboard" component={LeaderBoard} />
                <Route path="/replay/doudizhu" component={DoudizhuReplayView} />
                <Route path="/replay/leduc-holdem" component={LeducHoldemReplayView} />
                <Route path="/replay/spades" component={SpadesReplayView} />
                <Route path="/pve/doudizhu-demo" component={PvEDoudizhuDemoView} />
                <Route path="/pve/spades" component={PvESpadesView} />
            </div>
        </Router>
    );
}

export default App;
