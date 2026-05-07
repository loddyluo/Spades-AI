#include <cstdint>
#include <vector>
#include <algorithm>
#include <limits>

extern "C" {

struct NativeState {
    int32_t num_players;
    uint64_t hand_bits[4];
    int32_t hand_counts[4];
    int32_t table_pids[4];
    int32_t table_suits[4];
    int32_t table_ranks[4];
    int32_t table_count;
    int32_t turn;
    int32_t trick_leader;
    int32_t spades_broken;
    int32_t tricks_played;
    int32_t tricks_won[4];
    int32_t max_bid[4];
    int32_t teams[4];
};

// 根节点动作Q值导出结果：最多13个合法动作。
struct RootQResult {
    int32_t count;
    int32_t current_player;
    int32_t optimize_for_team;
    int32_t best_action;
    double value;
    int32_t actions[13];
    double q_values[13];
};

static inline int32_t card_suit(int32_t card_id) { return card_id / 13; }
static inline int32_t card_rank(int32_t card_id) { return (card_id % 13) + 2; }
static inline uint64_t card_bit(int32_t card_id) { return 1ULL << card_id; }

static inline int32_t popcount64(uint64_t x) {
#if defined(__GNUG__)
    return __builtin_popcountll(x);
#else
    int32_t c = 0;
    while (x) { x &= (x - 1); ++c; }
    return c;
#endif
}

static int32_t trick_winner_from_state(const NativeState* s) {
    if (s->table_count == 0) return s->trick_leader;

    int32_t best_pid = s->table_pids[0];
    int32_t best_rank = -1;
    bool has_spade = false;

    for (int32_t i = 0; i < s->table_count; ++i) {
        if (s->table_suits[i] == 0) {
            if (!has_spade || s->table_ranks[i] > best_rank) {
                has_spade = true;
                best_rank = s->table_ranks[i];
                best_pid = s->table_pids[i];
            }
        }
    }
    if (has_spade) return best_pid;

    const int32_t lead_suit = s->table_suits[0];
    best_rank = -1;
    for (int32_t i = 0; i < s->table_count; ++i) {
        if (s->table_suits[i] == lead_suit && s->table_ranks[i] > best_rank) {
            best_rank = s->table_ranks[i];
            best_pid = s->table_pids[i];
        }
    }
    return best_pid;
}

static inline bool hand_has_suit(uint64_t hand_bits, int32_t suit) {
    uint64_t mask = ((1ULL << 13) - 1ULL) << (suit * 13);
    return (hand_bits & mask) != 0;
}

static void legal_actions(const NativeState* s, int32_t player_id, std::vector<int32_t>& actions) {
    actions.clear();
    uint64_t hand = s->hand_bits[player_id];
    if (hand == 0) return;

    const int32_t table_count = s->table_count;
    if (table_count == 0) {
        // 领牌：黑桃未破时不能首攻黑桃，除非只剩黑桃。
        if (!s->spades_broken) {
            uint64_t non_spade_mask = hand & ~(((1ULL << 13) - 1ULL) << 0);
            non_spade_mask |= hand & ~(((1ULL << 13) - 1ULL) << 0); // fallback clarity
            bool has_non_spade = false;
            for (int32_t suit = 1; suit < 4; ++suit) {
                if (hand_has_suit(hand, suit)) { has_non_spade = true; break; }
            }
            if (has_non_spade) {
                for (int32_t cid = 0; cid < 52; ++cid) {
                    if ((hand & card_bit(cid)) && card_suit(cid) != 0) actions.push_back(cid);
                }
                return;
            }
        }
        for (int32_t cid = 0; cid < 52; ++cid) {
            if (hand & card_bit(cid)) actions.push_back(cid);
        }
        return;
    }

    const int32_t lead_suit = s->table_suits[0];
    bool has_lead = hand_has_suit(hand, lead_suit);
    if (has_lead) {
        for (int32_t cid = 0; cid < 52; ++cid) {
            if ((hand & card_bit(cid)) && card_suit(cid) == lead_suit) actions.push_back(cid);
        }
        return;
    }

    for (int32_t cid = 0; cid < 52; ++cid) {
        if (hand & card_bit(cid)) actions.push_back(cid);
    }
}

static double evaluate_score_diff(const NativeState* s) {
    int32_t team_scores[2] = {0, 0};
    for (int32_t team_id = 0; team_id < 2; ++team_id) {
        int members[2];
        int idx = 0;
        for (int32_t pid = 0; pid < 4; ++pid) {
            if (s->teams[pid] == team_id) members[idx++] = pid;
        }

        int32_t team_bid = 0;
        int32_t team_tricks = 0;
        double score = 0.0;
        for (int k = 0; k < 2; ++k) {
            int32_t pid = members[k];
            team_tricks += s->tricks_won[pid];
            int32_t bid = s->max_bid[pid];
            if (bid == 14) {
                score += (s->tricks_won[pid] == 0) ? 100.0 : -100.0;
            } else if (bid == 0) {
                score += (s->tricks_won[pid] == 0) ? 50.0 : -50.0;
            } else {
                team_bid += bid;
            }
        }

        if (team_bid > 0) {
            if (team_tricks >= team_bid) {
                int32_t over = team_tricks - team_bid;
                score += team_bid * 10 - over * 9;
            } else {
                score -= team_bid * 10;
            }
        }

        team_scores[team_id] = static_cast<int32_t>(score);
    }

    return static_cast<double>(team_scores[0] - team_scores[1]);
}

static inline uint64_t hash_state(const NativeState* s) {
    uint64_t h = 0;
    for (int i = 0; i < 4; ++i) {
        h ^= s->hand_bits[i];
        h = (h << 13) | (h >> (64 - 13));
    }
    for (int i = 0; i < s->table_count; ++i) {
        uint64_t x = (static_cast<uint64_t>(s->table_pids[i]) << 16) | static_cast<uint64_t>(s->table_suits[i] * 13 + (s->table_ranks[i] - 2));
        h ^= x;
        h = (h << 7) | (h >> (64 - 7));
    }
    h ^= static_cast<uint64_t>(s->turn);
    h = (h << 11) | (h >> (64 - 11));
    h ^= static_cast<uint64_t>(s->trick_leader) << 8;
    h = (h << 11) | (h >> (64 - 11));
    h ^= static_cast<uint64_t>(s->tricks_played) << 16;
    h = (h << 11) | (h >> (64 - 11));
    h ^= static_cast<uint64_t>(s->spades_broken) << 24;
    h = (h << 11) | (h >> (64 - 11));
    for (int i = 0; i < 4; ++i) {
        h ^= static_cast<uint64_t>(s->tricks_won[i]) << (i * 4);
        h = (h << 13) | (h >> (64 - 13));
    }
    return h;
}

static inline uint64_t verify_state(const NativeState* s) {
    uint64_t v = 0;
    for (int i = 0; i < 4; ++i) {
        v ^= s->hand_bits[i];
        v = (v << 17) | (v >> (64 - 17));
    }
    for (int i = 0; i < 4; ++i) {
        v ^= static_cast<uint64_t>(s->tricks_won[i]);
        v = (v << 19) | (v >> (64 - 19));
    }
    v ^= static_cast<uint64_t>(s->tricks_played);
    v = (v << 23) | (v >> (64 - 23));
    v ^= static_cast<uint64_t>(s->turn);
    return v;
}

struct TTEntry {
    uint64_t key;
    uint64_t verify;
    double value;
    int32_t flag;
    bool used;
};

static std::vector<TTEntry> g_tt;
static constexpr int TT_SIZE = 1 << 20;
static constexpr int32_t TT_EXACT = 0;
static constexpr int32_t TT_LOWER_BOUND = 1;
static constexpr int32_t TT_UPPER_BOUND = 2;

static inline TTEntry& tt_slot(uint64_t key) {
    return g_tt[key & (TT_SIZE - 1)];
}

static void clear_tt() {
    if (g_tt.empty()) {
        g_tt.resize(TT_SIZE);
    }
    for (int i = 0; i < TT_SIZE; ++i) {
        g_tt[i].used = false;
        g_tt[i].key = 0;
        g_tt[i].verify = 0;
        g_tt[i].value = 0.0;
        g_tt[i].flag = TT_EXACT;
    }
}

static double minimax(NativeState* s, double alpha, double beta);

static double solve_child_value(NativeState* s, int32_t player_id, int32_t action_cid, double alpha, double beta) {
    uint64_t bit = card_bit(action_cid);
    s->hand_bits[player_id] &= ~bit;
    s->table_pids[s->table_count] = player_id;
    s->table_suits[s->table_count] = card_suit(action_cid);
    s->table_ranks[s->table_count] = card_rank(action_cid);
    s->table_count += 1;
    s->turn = (player_id + 1) % s->num_players;
    s->spades_broken = s->spades_broken || (card_suit(action_cid) == 0);

    if (s->table_count == 4) {
        int32_t winner = trick_winner_from_state(s);
        s->tricks_won[winner] += 1;
        s->tricks_played += 1;
        s->table_count = 0;
        s->trick_leader = winner;
        s->turn = winner;
    }

    double value = minimax(s, alpha, beta);

    // 回滚
    if (s->table_count == 0) {
        // 刚刚完成一墩
        int32_t winner = s->trick_leader;
        // 注意：tricks_won 已在递归返回前修改，这里无法从当前状态直接恢复，因此调用方负责使用快照回滚。
    }
    return value;
}

static double minimax(NativeState* s, double alpha, double beta) {
    if (s->tricks_played >= 13) return evaluate_score_diff(s);

    uint64_t key = hash_state(s);
    TTEntry& slot = tt_slot(key);
    if (slot.used && slot.key == key && slot.verify == verify_state(s)) {
        if (slot.flag == TT_EXACT) return slot.value;
        if (slot.flag == TT_LOWER_BOUND && slot.value >= beta) return slot.value;
        if (slot.flag == TT_UPPER_BOUND && slot.value <= alpha) return slot.value;
    }

    int32_t current_player = s->turn;
    std::vector<int32_t> actions;
    legal_actions(s, current_player, actions);
    if (actions.empty()) return evaluate_score_diff(s);

    // 简单动作排序：黑桃和高点优先
    std::sort(actions.begin(), actions.end(), [](int32_t a, int32_t b) {
        int32_t sa = card_suit(a) == 0 ? 100 : 0;
        int32_t sb = card_suit(b) == 0 ? 100 : 0;
        int32_t ra = card_rank(a);
        int32_t rb = card_rank(b);
        return (sa + ra) > (sb + rb);
    });

    int32_t current_team = s->teams[current_player];
    bool pruned = false;
    double value;

    if (current_team == 0) {
        value = -std::numeric_limits<double>::infinity();
        for (int32_t action : actions) {
            NativeState child = *s;
            double child_value = solve_child_value(&child, current_player, action, alpha, beta);
            value = std::max(value, child_value);
            alpha = std::max(alpha, value);
            if (value >= beta) { pruned = true; break; }
        }
        slot.flag = pruned ? TT_LOWER_BOUND : TT_EXACT;
    } else {
        value = std::numeric_limits<double>::infinity();
        for (int32_t action : actions) {
            NativeState child = *s;
            double child_value = solve_child_value(&child, current_player, action, alpha, beta);
            value = std::min(value, child_value);
            beta = std::min(beta, value);
            if (value <= alpha) { pruned = true; break; }
        }
        slot.flag = pruned ? TT_UPPER_BOUND : TT_EXACT;
    }

    slot.key = key;
    slot.verify = verify_state(s);
    slot.value = value;
    slot.used = true;
    return value;
}

// 入口：只求当前状态的最优值。
// 返回值：当前状态最优得分差（队伍0 - 队伍1）。
double solve_native(const NativeState* input) {
    clear_tt();
    NativeState s = *input;
    return minimax(&s, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
}

// 入口：计算根节点每个合法动作的Q值，并返回最优动作。
void solve_native_with_q(const NativeState* input, RootQResult* out_result) {
    NativeState s = *input;

    out_result->current_player = s.turn;
    out_result->optimize_for_team = s.teams[s.turn];

    std::vector<int32_t> actions;
    legal_actions(&s, s.turn, actions);
    out_result->count = static_cast<int32_t>(actions.size());

    if (actions.empty()) {
        clear_tt();
        out_result->value = minimax(&s, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
        out_result->best_action = -1;
        return;
    }

    std::sort(actions.begin(), actions.end(), [](int32_t a, int32_t b) {
        int32_t sa = card_suit(a) == 0 ? 100 : 0;
        int32_t sb = card_suit(b) == 0 ? 100 : 0;
        int32_t ra = card_rank(a);
        int32_t rb = card_rank(b);
        return (sa + ra) > (sb + rb);
    });

    bool maximize = (out_result->optimize_for_team == 0);
    double best_value = maximize ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    int32_t best_action = actions[0];

    for (size_t i = 0; i < actions.size(); ++i) {
        NativeState child = s;
        double q_value;
        clear_tt();
        q_value = solve_child_value(&child, s.turn, actions[i], -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());

        out_result->actions[i] = actions[i];
        out_result->q_values[i] = q_value;

        if (maximize) {
            if (q_value > best_value) {
                best_value = q_value;
                best_action = actions[i];
            }
        } else {
            if (q_value < best_value) {
                best_value = q_value;
                best_action = actions[i];
            }
        }
    }

    out_result->best_action = best_action;
    out_result->value = best_value;
}

}
