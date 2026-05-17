#include <cstdint>
#include <vector>
#include <algorithm>
#include <limits>
#include <thread>
#include <cstring>

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

// 等价牌过滤：同花色中连续的牌（中间没有其他玩家的牌）只保留最高的。
// 例如手持 K♥Q♥ 且没有人持有 K♥Q♥ 之间的牌，则 Q♥ 等价于 K♥，只搜 K♥。
static void filter_equivalent(const NativeState* s, int32_t player_id, std::vector<int32_t>& actions) {
    if (actions.size() <= 1) return;

    const uint64_t my_hand = s->hand_bits[player_id];

    // 所有其他玩家的手牌 + 桌面上的牌，合并成"非我方持有"的位图
    uint64_t others = 0;
    for (int32_t p = 0; p < 4; ++p) {
        if (p != player_id) others |= s->hand_bits[p];
    }
    // 桌面牌也算outstanding（它们影响当前墩的胜负判定）
    for (int32_t i = 0; i < s->table_count; ++i) {
        int32_t cid = s->table_suits[i] * 13 + (s->table_ranks[i] - 2);
        others |= (1ULL << cid);
    }

    // 对每个花色，找到我方持有的rank集合和others持有的rank集合
    // 在我方rank中，从高到低扫描，如果相邻两张之间没有others的rank，低的那张等价，标记移除
    uint64_t remove_mask = 0;
    for (int32_t suit = 0; suit < 4; ++suit) {
        int32_t base = suit * 13;
        // 提取该花色的13位rank掩码（bit 0 = rank 2, bit 12 = rank Ace）
        uint32_t my_ranks = static_cast<uint32_t>((my_hand >> base) & 0x1FFFULL);
        uint32_t ot_ranks = static_cast<uint32_t>((others >> base) & 0x1FFFULL);

        if (my_ranks == 0) continue;

        // 从最高位(Ace=bit12)向下扫描
        int32_t prev_my_bit = -1;  // 上一张我方的rank bit位置
        for (int32_t bit = 12; bit >= 0; --bit) {
            if (!(my_ranks & (1U << bit))) continue;
            if (prev_my_bit < 0) {
                // 第一张（最高的），保留
                prev_my_bit = bit;
                continue;
            }
            // 检查 prev_my_bit 和 bit 之间是否有 others 的牌
            // 区间: (bit, prev_my_bit) 开区间
            bool has_gap = false;
            for (int32_t g = bit + 1; g < prev_my_bit; ++g) {
                if (ot_ranks & (1U << g)) { has_gap = true; break; }
            }
            if (!has_gap) {
                // bit 这张牌等价于 prev_my_bit，标记移除
                remove_mask |= (1ULL << (base + bit));
            } else {
                // 有间隔，这张牌不等价，成为新的"上一张"
                prev_my_bit = bit;
            }
        }
    }

    if (remove_mask == 0) return;

    // 过滤 actions
    size_t write = 0;
    for (size_t read = 0; read < actions.size(); ++read) {
        if (!(remove_mask & (1ULL << actions[read]))) {
            actions[write++] = actions[read];
        }
    }
    actions.resize(write);
}

static void legal_actions(const NativeState* s, int32_t player_id, std::vector<int32_t>& actions) {
    actions.clear();
    const uint64_t hand = s->hand_bits[player_id];
    if (hand == 0) return;

    const int32_t table_count = s->table_count;
    if (table_count == 0) {
        // 领牌：黑桃未破时不能首攻黑桃，除非只剩黑桃。
        if (!s->spades_broken) {
            bool has_non_spade = false;
            for (int32_t suit = 1; suit < 4; ++suit) {
                if (hand_has_suit(hand, suit)) { has_non_spade = true; break; }
            }
            if (has_non_spade) {
                uint64_t bits = hand;
                while (bits) {
                    int32_t cid = __builtin_ctzll(bits);
                    bits &= (bits - 1ULL);
                    if (card_suit(cid) != 0) actions.push_back(cid);
                }
                filter_equivalent(s, player_id, actions);
                return;
            }
        }
        uint64_t bits = hand;
        while (bits) {
            int32_t cid = __builtin_ctzll(bits);
            bits &= (bits - 1ULL);
            actions.push_back(cid);
        }
        filter_equivalent(s, player_id, actions);
        return;
    }

    const int32_t lead_suit = s->table_suits[0];
    bool has_lead = hand_has_suit(hand, lead_suit);
    if (has_lead) {
        uint64_t bits = hand;
        while (bits) {
            int32_t cid = __builtin_ctzll(bits);
            bits &= (bits - 1ULL);
            if (card_suit(cid) == lead_suit) actions.push_back(cid);
        }
        filter_equivalent(s, player_id, actions);
        return;
    }

    uint64_t bits = hand;
    while (bits) {
        int32_t cid = __builtin_ctzll(bits);
        bits &= (bits - 1ULL);
        actions.push_back(cid);
    }
    filter_equivalent(s, player_id, actions);
}

static double evaluate_score_diff(const NativeState* s) {
    double team_scores[2] = {0.0, 0.0};
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

        team_scores[team_id] = score;
    }

    return team_scores[0] - team_scores[1];
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
    v = (v << 7) | (v >> (64 - 7));
    v ^= static_cast<uint64_t>(s->trick_leader) << 3;
    v = (v << 11) | (v >> (64 - 11));
    v ^= static_cast<uint64_t>(s->spades_broken) << 5;
    // 桌面牌也参与 verify
    for (int i = 0; i < s->table_count; ++i) {
        v ^= static_cast<uint64_t>(s->table_suits[i] * 13 + s->table_ranks[i]) << 9;
        v = (v << 13) | (v >> (64 - 13));
        v ^= static_cast<uint64_t>(s->table_pids[i]) << 2;
        v = (v << 7) | (v >> (64 - 7));
    }
    return v;
}

struct TTEntry {
    uint64_t key;
    uint64_t verify;
    double value;
    int32_t flag;
    int32_t best_action;  // 该节点搜索到的最佳动作，用于move ordering提示
    bool used;
};

static std::vector<TTEntry> g_tt;
static constexpr int TT_SIZE = 1 << 21;  // 2M entries
static constexpr int32_t TT_EXACT = 0;
static constexpr int32_t TT_LOWER_BOUND = 1;
static constexpr int32_t TT_UPPER_BOUND = 2;

// 求解上下文：包含 TT 和 killer 表。多线程时每线程一份。
struct SolverCtx {
    std::vector<TTEntry> tt;
    int32_t killer[56];

    void init() {
        tt.resize(TT_SIZE);
        clear();
    }
    void clear() {
        for (int i = 0; i < TT_SIZE; ++i) {
            tt[i].used = false; tt[i].key = 0; tt[i].verify = 0;
            tt[i].value = 0.0; tt[i].flag = TT_EXACT; tt[i].best_action = -1;
        }
        for (int i = 0; i < 56; ++i) killer[i] = -1;
    }
    TTEntry& slot(uint64_t key) { return tt[key & (TT_SIZE - 1)]; }
};

// 全局默认上下文（单线程 solve_native 使用）
static SolverCtx g_ctx;

static inline TTEntry& tt_slot(uint64_t key) {
    return g_ctx.tt[key & (TT_SIZE - 1)];
}

static void clear_tt() {
    g_ctx.init();
}

static void clear_killers() {
    for (int i = 0; i < 56; ++i) g_ctx.killer[i] = -1;
}

// 将指定动作移到 actions 最前面（如果存在）
static inline void promote_action(std::vector<int32_t>& actions, int32_t target) {
    if (target < 0) return;
    for (size_t i = 1; i < actions.size(); ++i) {
        if (actions[i] == target) {
            // swap to front
            int32_t tmp = actions[0];
            actions[0] = actions[i];
            actions[i] = tmp;
            return;
        }
    }
}

static double minimax(NativeState* s, double alpha, double beta);

// 增量出牌：保存被修改的最小字段集，原地修改状态，返回回滚所需的快照。
struct UndoInfo {
    uint64_t hand_bit;       // 被移除的牌的 bit
    int32_t player_id;
    int32_t old_turn;
    int32_t old_table_count;
    int32_t old_spades_broken;
    // 如果完成了一墩，还需要回滚 tricks_won、tricks_played、trick_leader
    // 以及保存被覆盖的 table 数据（下一墩的 make_move 会写 table_pids[0] 等）
    bool trick_completed;
    int32_t old_tricks_played;
    int32_t old_trick_leader;
    int32_t trick_winner;
    // 保存完成的墩的 4 张桌面牌（回滚后需要恢复，否则后续 hash 会出错）
    int32_t saved_table_pids[4];
    int32_t saved_table_suits[4];
    int32_t saved_table_ranks[4];
};

static inline UndoInfo make_move(NativeState* s, int32_t player_id, int32_t action_cid) {
    UndoInfo u;
    u.hand_bit = card_bit(action_cid);
    u.player_id = player_id;
    u.old_turn = s->turn;
    u.old_table_count = s->table_count;
    u.old_spades_broken = s->spades_broken;
    u.old_tricks_played = s->tricks_played;
    u.old_trick_leader = s->trick_leader;
    u.trick_completed = false;
    u.trick_winner = -1;

    s->hand_bits[player_id] &= ~u.hand_bit;
    s->table_pids[s->table_count] = player_id;
    s->table_suits[s->table_count] = card_suit(action_cid);
    s->table_ranks[s->table_count] = card_rank(action_cid);
    s->table_count += 1;
    s->turn = (player_id + 1) % s->num_players;
    s->spades_broken = s->spades_broken || (card_suit(action_cid) == 0);

    if (s->table_count == 4) {
        u.trick_completed = true;
        // 保存 4 张桌面牌数据（下一墩的 make_move 会覆盖它们）
        for (int k = 0; k < 4; ++k) {
            u.saved_table_pids[k] = s->table_pids[k];
            u.saved_table_suits[k] = s->table_suits[k];
            u.saved_table_ranks[k] = s->table_ranks[k];
        }
        int32_t winner = trick_winner_from_state(s);
        u.trick_winner = winner;
        s->tricks_won[winner] += 1;
        s->tricks_played += 1;
        s->table_count = 0;
        s->trick_leader = winner;
        s->turn = winner;
    }
    return u;
}

static inline void unmake_move(NativeState* s, const UndoInfo& u) {
    if (u.trick_completed) {
        s->tricks_won[u.trick_winner] -= 1;
        s->tricks_played = u.old_tricks_played;
        s->trick_leader = u.old_trick_leader;
        // 恢复桌面牌数据
        for (int k = 0; k < 4; ++k) {
            s->table_pids[k] = u.saved_table_pids[k];
            s->table_suits[k] = u.saved_table_suits[k];
            s->table_ranks[k] = u.saved_table_ranks[k];
        }
    }
    s->table_count = u.old_table_count;
    s->turn = u.old_turn;
    s->spades_broken = u.old_spades_broken;
    s->hand_bits[u.player_id] |= u.hand_bit;
}

// ============================================================================
// Quick Tricks 剪枝
//
// 在墩边界（table_count==0）分析每个花色的"确定赢墩"，快速估算得分上下界。
// 如果上下界已满足 alpha-beta 窗口，直接返回，跳过完整搜索。
//
// 只计算保守的、100%确定的赢墩：
// 1. 某花色的 master card（场上最大牌）持有者一定赢该墩
// 2. 某花色只有一方有牌 → 全部赢
// 3. 王牌(黑桃)最高牌的持有者在切牌时一定赢
// ============================================================================

static inline uint32_t suit_ranks(uint64_t hand, int32_t suit) {
    return static_cast<uint32_t>((hand >> (suit * 13)) & 0x1FFFULL);
}

static inline int32_t highest_bit(uint32_t x) {
    // 返回最高位的位置 (0-12)，x==0 返回 -1
    if (x == 0) return -1;
    int32_t b = 0;
    if (x & 0xFF00) { b += 8; x >>= 8; }
    if (x & 0x00F0) { b += 4; x >>= 4; }
    if (x & 0x000C) { b += 2; x >>= 2; }
    if (x & 0x0002) { b += 1; }
    return b;
}

// 计算当前局面下，team0 的确定赢墩下界和上界。
// 返回值: quick_tricks_bound(s, &lower, &upper) -> bool (是否有意义的bound)
// lower: team0 至少能再赢这么多墩
// upper: team0 至多能再赢这么多墩
// 注意：这里的 lower/upper 是"剩余墩数"中 team0 能赢的范围
static bool quick_tricks_bound(const NativeState* s, int32_t* team0_min, int32_t* team0_max) {
    // 只在墩边界分析
    if (s->table_count != 0) return false;

    int32_t remaining_tricks = 13 - s->tricks_played;
    if (remaining_tricks <= 1) return false;

    // 合并各队手牌
    uint64_t team_hand[2] = {0, 0};
    for (int p = 0; p < 4; ++p) {
        team_hand[s->teams[p]] |= s->hand_bits[p];
    }

    // 各玩家持有的黑桃
    uint32_t sp0 = suit_ranks(team_hand[0], 0);
    uint32_t sp1 = suit_ranks(team_hand[1], 0);
    bool team0_has_trump = (sp0 != 0);
    bool team1_has_trump = (sp1 != 0);

    int32_t sure_team0 = 0;
    int32_t sure_team1 = 0;

    // 分析黑桃(王牌)花色：只有一方垄断所有黑桃时才确定赢
    // 双方都有黑桃时不做任何假设（因为出牌顺序和花色领牌权的交互太复杂）
    if (sp0 != 0 && sp1 == 0) {
        // team0 垄断所有黑桃
        sure_team0 += popcount64(sp0);
    } else if (sp1 != 0 && sp0 == 0) {
        // team1 垄断所有黑桃
        sure_team1 += popcount64(sp1);
    }
    // 双方都有黑桃 → 不计任何确定赢墩（保守）

    // 分析非黑桃花色：只有"单方垄断且对手无王牌"时才能确定赢
    for (int32_t suit = 1; suit < 4; ++suit) {
        uint32_t r0 = suit_ranks(team_hand[0], suit);
        uint32_t r1 = suit_ranks(team_hand[1], suit);

        if (r0 == 0 && r1 == 0) continue;

        // 只有一方有牌 + 对手无王牌 → 全赢
        if (r1 == 0 && r0 != 0 && !team1_has_trump) {
            sure_team0 += popcount64(r0);
        } else if (r0 == 0 && r1 != 0 && !team0_has_trump) {
            sure_team1 += popcount64(r1);
        }
        // 双方都有 → 不做任何假设（保守）
    }

    if (sure_team0 > remaining_tricks) sure_team0 = remaining_tricks;
    if (sure_team1 > remaining_tricks) sure_team1 = remaining_tricks;

    *team0_min = sure_team0;
    *team0_max = remaining_tricks - sure_team1;

    // 只在 bound 有意义时返回 true
    return (sure_team0 > 0 || sure_team1 > 0);
}

// 根据 team0 赢墩范围，计算得分差的上下界。
// 枚举 team0_extra 从 min 到 max，对每个值枚举所有可能的墩分配方式，
// 取最小和最大得分差。
static void score_bounds_from_tricks(
    const NativeState* s,
    int32_t team0_extra_min,
    int32_t team0_extra_max,
    int32_t remaining_tricks,
    double* score_lower,
    double* score_upper
) {
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();

    // 找到每个队伍的成员
    int32_t t0[2], t1[2];
    int t0n = 0, t1n = 0;
    for (int p = 0; p < 4; ++p) {
        if (s->teams[p] == 0) t0[t0n++] = p;
        else t1[t1n++] = p;
    }

    // 控制枚举量：如果范围太大，放弃（避免 quick tricks 本身变成瓶颈）
    int32_t range = team0_extra_max - team0_extra_min + 1;
    if (range > 6 || remaining_tricks > 8) {
        *score_lower = -std::numeric_limits<double>::infinity();
        *score_upper = std::numeric_limits<double>::infinity();
        return;
    }

    for (int32_t t0extra = team0_extra_min; t0extra <= team0_extra_max; ++t0extra) {
        int32_t t1extra = remaining_tricks - t0extra;

        // 枚举 team0 内部的分配：成员A 拿 a 墩，成员B 拿 t0extra-a 墩
        for (int32_t a0 = 0; a0 <= t0extra; ++a0) {
            int32_t a1 = t0extra - a0;
            // 枚举 team1 内部的分配
            for (int32_t b0 = 0; b0 <= t1extra; ++b0) {
                int32_t b1 = t1extra - b0;

                // 临时修改 tricks_won
                int32_t saved[4];
                for (int i = 0; i < 4; ++i) saved[i] = s->tricks_won[i];
                const_cast<NativeState*>(s)->tricks_won[t0[0]] += a0;
                const_cast<NativeState*>(s)->tricks_won[t0[1]] += a1;
                const_cast<NativeState*>(s)->tricks_won[t1[0]] += b0;
                const_cast<NativeState*>(s)->tricks_won[t1[1]] += b1;

                double v = evaluate_score_diff(s);
                if (v < lo) lo = v;
                if (v > hi) hi = v;

                for (int i = 0; i < 4; ++i) const_cast<NativeState*>(s)->tricks_won[i] = saved[i];
            }
        }
    }

    *score_lower = lo;
    *score_upper = hi;
}

static double minimax(NativeState* s, double alpha, double beta) {
    if (s->tricks_played >= 13) return evaluate_score_diff(s);

    uint64_t key = hash_state(s);
    TTEntry& slot = tt_slot(key);
    int32_t tt_best = -1;
    if (slot.used && slot.key == key && slot.verify == verify_state(s)) {
        if (slot.flag == TT_EXACT) return slot.value;
        if (slot.flag == TT_LOWER_BOUND && slot.value >= beta) return slot.value;
        if (slot.flag == TT_UPPER_BOUND && slot.value <= alpha) return slot.value;
        tt_best = slot.best_action;
    }

    // Quick Tricks 剪枝暂时禁用。
    // 问题：Spades 的计分规则中 overtrick 和 nil 使得"确定赢墩数"无法简单
    // 转换为得分上下界（取决于墩在队伍内部的分配方式）。需要一个基于队伍墩数
    // 总和的专用计分函数，而非复用 per-player 的 evaluate_score_diff。
    // 代码保留在文件上方的 quick_tricks_bound / score_bounds_from_tricks 中。

    int32_t current_player = s->turn;
    std::vector<int32_t> actions;
    legal_actions(s, current_player, actions);
    if (actions.empty()) return evaluate_score_diff(s);

    // 动作排序：黑桃和高点优先（基础排序）
    std::sort(actions.begin(), actions.end(), [](int32_t a, int32_t b) {
        int32_t sa = card_suit(a) == 0 ? 100 : 0;
        int32_t sb = card_suit(b) == 0 ? 100 : 0;
        int32_t ra = card_rank(a);
        int32_t rb = card_rank(b);
        return (sa + ra) > (sb + rb);
    });

    // 优先级：TT best move > killer move > 静态排序
    int32_t depth = s->tricks_played * 4 + s->table_count;
    promote_action(actions, g_ctx.killer[depth]);  // killer 先于静态排序
    promote_action(actions, tt_best);          // TT best 优先级最高（最后 promote = 最前）

    int32_t current_team = s->teams[current_player];
    bool pruned = false;
    double value;
    int32_t best_action_here = actions[0];
    bool first_child = true;

    if (current_team == 0) {
        value = -std::numeric_limits<double>::infinity();
        for (int32_t action : actions) {
            NativeState child = *s;
            UndoInfo undo = make_move(&child, current_player, action);
            double child_value;
            if (first_child) {
                child_value = minimax(&child, alpha, beta);
                first_child = false;
            } else {
                child_value = minimax(&child, alpha, alpha + 1.0);
                if (child_value > alpha && child_value < beta) {
                    NativeState re = *s;
                    UndoInfo ru = make_move(&re, current_player, action);
                    child_value = minimax(&re, alpha, beta);
                }
            }
            if (child_value > value) {
                value = child_value;
                best_action_here = action;
            }
            alpha = std::max(alpha, value);
            if (value >= beta) {
                pruned = true;
                g_ctx.killer[depth] = action;
                break;
            }
        }
    } else {
        value = std::numeric_limits<double>::infinity();
        for (int32_t action : actions) {
            NativeState child = *s;
            UndoInfo undo = make_move(&child, current_player, action);
            double child_value;
            if (first_child) {
                child_value = minimax(&child, alpha, beta);
                first_child = false;
            } else {
                child_value = minimax(&child, beta - 1.0, beta);
                if (child_value < beta && child_value > alpha) {
                    NativeState re = *s;
                    UndoInfo ru = make_move(&re, current_player, action);
                    child_value = minimax(&re, alpha, beta);
                }
            }
            if (child_value < value) {
                value = child_value;
                best_action_here = action;
            }
            beta = std::min(beta, value);
            if (value <= alpha) {
                pruned = true;
                g_ctx.killer[depth] = action;
                break;
            }
        }
    }

    slot.flag = pruned ? (current_team == 0 ? TT_LOWER_BOUND : TT_UPPER_BOUND) : TT_EXACT;
    slot.key = key;
    slot.verify = verify_state(s);
    slot.value = value;
    slot.best_action = best_action_here;
    slot.used = true;
    return value;
}

// ============================================================
// 上下文参数化版 minimax（用于多线程，每线程有独立的 TT 和 killer）
// ============================================================
static double minimax_ctx(NativeState* s, double alpha, double beta, SolverCtx& ctx) {
    if (s->tricks_played >= 13) return evaluate_score_diff(s);

    uint64_t key = hash_state(s);
    TTEntry& slot = ctx.slot(key);
    int32_t tt_best = -1;
    if (slot.used && slot.key == key && slot.verify == verify_state(s)) {
        if (slot.flag == TT_EXACT) return slot.value;
        if (slot.flag == TT_LOWER_BOUND && slot.value >= beta) return slot.value;
        if (slot.flag == TT_UPPER_BOUND && slot.value <= alpha) return slot.value;
        tt_best = slot.best_action;
    }

    int32_t current_player = s->turn;
    std::vector<int32_t> actions;
    legal_actions(s, current_player, actions);
    if (actions.empty()) return evaluate_score_diff(s);

    std::sort(actions.begin(), actions.end(), [](int32_t a, int32_t b) {
        return (card_suit(a) == 0 ? 100 : 0) + card_rank(a) > (card_suit(b) == 0 ? 100 : 0) + card_rank(b);
    });

    int32_t depth = s->tricks_played * 4 + s->table_count;
    promote_action(actions, ctx.killer[depth]);
    promote_action(actions, tt_best);

    int32_t current_team = s->teams[current_player];
    bool pruned = false;
    double value;
    int32_t best_action_here = actions[0];
    bool first_child = true;

    if (current_team == 0) {
        value = -std::numeric_limits<double>::infinity();
        for (int32_t action : actions) {
            NativeState child = *s;
            UndoInfo undo = make_move(&child, current_player, action);
            double cv;
            if (first_child) { cv = minimax_ctx(&child, alpha, beta, ctx); first_child = false; }
            else {
                cv = minimax_ctx(&child, alpha, alpha + 1.0, ctx);
                if (cv > alpha && cv < beta) {
                    NativeState re = *s; UndoInfo ru = make_move(&re, current_player, action);
                    cv = minimax_ctx(&re, alpha, beta, ctx);
                }
            }
            if (cv > value) { value = cv; best_action_here = action; }
            alpha = std::max(alpha, value);
            if (value >= beta) { pruned = true; ctx.killer[depth] = action; break; }
        }
    } else {
        value = std::numeric_limits<double>::infinity();
        for (int32_t action : actions) {
            NativeState child = *s;
            UndoInfo undo = make_move(&child, current_player, action);
            double cv;
            if (first_child) { cv = minimax_ctx(&child, alpha, beta, ctx); first_child = false; }
            else {
                cv = minimax_ctx(&child, beta - 1.0, beta, ctx);
                if (cv < beta && cv > alpha) {
                    NativeState re = *s; UndoInfo ru = make_move(&re, current_player, action);
                    cv = minimax_ctx(&re, alpha, beta, ctx);
                }
            }
            if (cv < value) { value = cv; best_action_here = action; }
            beta = std::min(beta, value);
            if (value <= alpha) { pruned = true; ctx.killer[depth] = action; break; }
        }
    }

    slot.flag = pruned ? (current_team == 0 ? TT_LOWER_BOUND : TT_UPPER_BOUND) : TT_EXACT;
    slot.key = key; slot.verify = verify_state(s);
    slot.value = value; slot.best_action = best_action_here; slot.used = true;
    return value;
}

// 入口：只求当前状态的最优值。
// 返回值：当前状态最优得分差（队伍0 - 队伍1）。
double solve_native(const NativeState* input) {
    clear_tt();
    clear_killers();
    NativeState s = *input;
    return minimax(&s, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
}

// 入口：计算根节点每个合法动作的Q值，并返回最优动作。
// 多线程：每个根动作在独立线程中用独立的 TT/killer 求解。
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
        return (card_suit(a) == 0 ? 100 : 0) + card_rank(a) > (card_suit(b) == 0 ? 100 : 0) + card_rank(b);
    });

    int32_t n = static_cast<int32_t>(actions.size());
    bool maximize = (out_result->optimize_for_team == 0);

    // 每个根动作的结果
    std::vector<double> q_values(n);

    if (n == 1) {
        // 单动作：直接求解
        clear_tt();
        clear_killers();
        NativeState child = s;
        UndoInfo undo = make_move(&child, child.turn, actions[0]);
        q_values[0] = minimax(&child, -std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity());
    } else if (n <= 2 || popcount64(s.hand_bits[0] | s.hand_bits[1] | s.hand_bits[2] | s.hand_bits[3]) <= 28) {
        // 少量动作或小局面：单线程顺序求解（避免线程创建开销）
        clear_tt();
        clear_killers();
        double best_val = maximize ? -std::numeric_limits<double>::infinity()
                                   : std::numeric_limits<double>::infinity();
        for (int32_t i = 0; i < n; ++i) {
            NativeState child = s;
            UndoInfo undo = make_move(&child, child.turn, actions[i]);
            if (maximize) {
                q_values[i] = minimax(&child, best_val, std::numeric_limits<double>::infinity());
                if (q_values[i] > best_val) best_val = q_values[i];
            } else {
                q_values[i] = minimax(&child, -std::numeric_limits<double>::infinity(), best_val);
                if (q_values[i] < best_val) best_val = q_values[i];
            }
        }
    } else {
        // 多动作大局面：先串行求解第一个动作获取 baseline，
        // 然后用 baseline 做窗口并行求解其余动作。
        // 第1步：串行求第一个动作
        {
            SolverCtx ctx0;
            ctx0.init();
            NativeState child = s;
            UndoInfo undo = make_move(&child, child.turn, actions[0]);
            q_values[0] = minimax_ctx(&child,
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(), ctx0);
        }
        double baseline = q_values[0];

        // 第2步：并行求其余动作（用 baseline 做窗口）
        if (n > 1) {
            std::vector<SolverCtx> ctxs(n - 1);
            std::vector<std::thread> threads;
            for (int32_t i = 1; i < n; ++i) {
                ctxs[i-1].init();
                threads.emplace_back([&, i]() {
                    NativeState child = s;
                    UndoInfo undo = make_move(&child, child.turn, actions[i]);
                    if (maximize) {
                        q_values[i] = minimax_ctx(&child, baseline,
                            std::numeric_limits<double>::infinity(), ctxs[i-1]);
                    } else {
                        q_values[i] = minimax_ctx(&child,
                            -std::numeric_limits<double>::infinity(), baseline, ctxs[i-1]);
                    }
                });
            }
            for (auto& t : threads) t.join();
        }
    }

    // 汇总结果
    double best_value = maximize ? -std::numeric_limits<double>::infinity()
                                 : std::numeric_limits<double>::infinity();
    int32_t best_action = actions[0];

    for (int32_t i = 0; i < n; ++i) {
        out_result->actions[i] = actions[i];
        out_result->q_values[i] = q_values[i];
        if (maximize) {
            if (q_values[i] > best_value) { best_value = q_values[i]; best_action = actions[i]; }
        } else {
            if (q_values[i] < best_value) { best_value = q_values[i]; best_action = actions[i]; }
        }
    }

    out_result->best_action = best_action;
    out_result->value = best_value;
}

}
