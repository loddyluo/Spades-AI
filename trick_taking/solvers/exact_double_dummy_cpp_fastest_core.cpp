/**
 * 黑桃王（Spades）极速精确双明手求解器
 *
 * 优化技术总览：
 * 1. Zobrist Hashing - 增量更新哈希，O(1) make/unmake
 * 2. State Normalization (Rank Canonicalization) - 消除rank间隙，TT命中率暴增
 * 3. Global Score-Range Pruning - 用所有可能终局的超集安全剪枝
 * 4. TT Only at Trick Boundaries - 减少缓存污染，提高命中率
 * 5. Zero Heap Allocation - 全部使用栈上固定数组
 * 6. Position-Aware Move Ordering - 领牌/跟牌使用不同排序策略
 * 7. Equivalent Card Filtering - 等价牌只搜一张
 * 8. PVS (Principal Variation Search) - 零窗口搜索
 * 9. Killer Move Heuristic - 深度相关的 killer 表
 * 10. History Heuristic - 历史表辅助排序
 * 11. MTD(f) - 根节点零窗口迭代逼近
 * 12. Multi-threading - 根节点动作并行
 * 13. Make/Unmake - 原地修改+回滚，无深拷贝
 * 14. Compiler Intrinsics - __builtin_ctzll, __builtin_popcountll
 */

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <limits>
#include <thread>
#include <random>
#include <vector>

#ifndef SPADES_NATIVE_BUILD_ID
#define SPADES_NATIVE_BUILD_ID "unversioned"
#endif
#ifndef SPADES_NATIVE_ABI_VERSION
#define SPADES_NATIVE_ABI_VERSION 0
#endif

extern "C" {

const char* spades_native_build_id() {
    return SPADES_NATIVE_BUILD_ID;
}

uint32_t spades_native_abi_version() {
    return SPADES_NATIVE_ABI_VERSION;
}

// ============================================================
// Data Structures
// ============================================================

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

struct RootQResult {
    int32_t count;
    int32_t current_player;
    int32_t optimize_for_team;
    int32_t best_action;
    double value;
    int32_t actions[13];
    double q_values[13];
};

enum ForcedOutcomeStatus : int32_t {
    FORCED_FIXED = 1,
    FORCED_VARIABLE = 2,
    FORCED_TIMEOUT = 3,
};

struct ForcedOutcomeResult {
    int32_t status;
    int32_t team0_final_tricks;
    uint32_t nil_broken_mask;
    uint64_t nodes_searched;
    double elapsed_ms;
};

// Fixed-size action list (max 13 cards in a suit, max 13 in hand)
struct ActionList {
    int32_t cards[13];
    int32_t count;
    ActionList() : count(0) {}
    void push(int32_t c) { cards[count++] = c; }
    void clear() { count = 0; }
};

// Undo info for make/unmake
struct UndoInfo {
    uint64_t hand_bit;
    int32_t player_id;
    int32_t old_turn;
    int32_t old_table_count;
    int32_t old_spades_broken;
    bool trick_completed;
    int32_t old_tricks_played;
    int32_t old_trick_leader;
    int32_t trick_winner;
    int32_t saved_table_pids[4];
    int32_t saved_table_suits[4];
    int32_t saved_table_ranks[4];
    // Zobrist hash delta
    uint64_t hash_delta;
};

// ============================================================
// Constants & Inline Helpers
// ============================================================

static inline int32_t card_suit(int32_t card_id) { return card_id / 13; }
static inline int32_t card_rank(int32_t card_id) { return (card_id % 13) + 2; }
static inline uint64_t card_bit(int32_t card_id) { return 1ULL << card_id; }

static inline int32_t popcount64(uint64_t x) {
    return __builtin_popcountll(x);
}

static inline int32_t ctz64(uint64_t x) {
    return __builtin_ctzll(x);
}

static inline bool hand_has_suit(uint64_t hand_bits, int32_t suit) {
    return (hand_bits & (0x1FFFULL << (suit * 13))) != 0;
}

static inline uint32_t suit_ranks(uint64_t hand, int32_t suit) {
    return static_cast<uint32_t>((hand >> (suit * 13)) & 0x1FFFULL);
}

// ============================================================
// Zobrist Hashing
// ============================================================

static uint64_t zobrist_card_player[4][52];   // [player][card_id]
static uint64_t zobrist_turn[4];
static uint64_t zobrist_leader[4];
static uint64_t zobrist_spades_broken;
static uint64_t zobrist_tricks_won[4][14];    // [player][tricks 0-13]
static uint64_t zobrist_tricks_played[14];
static uint64_t zobrist_table_card[4][52];    // [position_in_trick][card_id]
static uint64_t zobrist_table_pid[4][4];      // [position_in_trick][player_id]
static bool zobrist_initialized = false;

static void init_zobrist() {
    if (zobrist_initialized) return;
    std::mt19937_64 rng(0xDEADBEEF42ULL);
    for (int p = 0; p < 4; p++)
        for (int c = 0; c < 52; c++)
            zobrist_card_player[p][c] = rng();
    for (int i = 0; i < 4; i++) zobrist_turn[i] = rng();
    for (int i = 0; i < 4; i++) zobrist_leader[i] = rng();
    zobrist_spades_broken = rng();
    for (int p = 0; p < 4; p++)
        for (int t = 0; t < 14; t++)
            zobrist_tricks_won[p][t] = rng();
    for (int t = 0; t < 14; t++)
        zobrist_tricks_played[t] = rng();
    for (int pos = 0; pos < 4; pos++)
        for (int c = 0; c < 52; c++)
            zobrist_table_card[pos][c] = rng();
    for (int pos = 0; pos < 4; pos++)
        for (int p = 0; p < 4; p++)
            zobrist_table_pid[pos][p] = rng();
    zobrist_initialized = true;
}

static uint64_t compute_full_hash(const NativeState* s) {
    uint64_t h = 0;
    for (int p = 0; p < 4; p++) {
        uint64_t bits = s->hand_bits[p];
        while (bits) {
            int cid = ctz64(bits);
            h ^= zobrist_card_player[p][cid];
            bits &= bits - 1;
        }
    }
    h ^= zobrist_turn[s->turn];
    h ^= zobrist_leader[s->trick_leader];
    if (s->spades_broken) h ^= zobrist_spades_broken;
    for (int p = 0; p < 4; p++)
        h ^= zobrist_tricks_won[p][s->tricks_won[p]];
    h ^= zobrist_tricks_played[s->tricks_played];
    for (int i = 0; i < s->table_count; i++) {
        int cid = s->table_suits[i] * 13 + (s->table_ranks[i] - 2);
        h ^= zobrist_table_card[i][cid];
        h ^= zobrist_table_pid[i][s->table_pids[i]];
    }
    return h;
}

// ============================================================
// State Normalization (Rank Canonicalization)
// ============================================================

struct NormalizedTTKey {
    uint64_t hash;
    uint64_t verify;
};

// Normalize once, then derive both independent TT fingerprints from the same
// canonical hands.  The previous implementation repeated the full rank-map
// construction whenever a candidate slot needed collision verification.
static NormalizedTTKey compute_normalized_tt_key(const NativeState* s) {
    // Collect all remaining cards across all hands
    uint64_t all_remaining = 0;
    for (int p = 0; p < 4; p++) all_remaining |= s->hand_bits[p];

    // For each suit, compute rank mapping: existing ranks → 0,1,2,...
    // Then build normalized hand_bits for each player
    uint64_t norm_hands[4] = {0, 0, 0, 0};

    for (int suit = 0; suit < 4; suit++) {
        int base = suit * 13;
        uint32_t remaining_suit = static_cast<uint32_t>((all_remaining >> base) & 0x1FFFULL);
        if (remaining_suit == 0) continue;

        // Build rank map: for each set bit in remaining_suit (from high to low),
        // assign decreasing normalized rank
        // E.g. bits {12, 10, 9, 7} → normalized {3, 2, 1, 0}
        int rank_map[13];  // original bit position → normalized rank
        // Count bits first to know the highest normalized rank
        int total = __builtin_popcount(remaining_suit);
        // Assign from highest bit to lowest
        int nr = total - 1;
        for (int bit = 12; bit >= 0; bit--) {
            if (remaining_suit & (1u << bit)) {
                rank_map[bit] = nr;
                nr--;
            }
        }

        // Now map each player's cards in this suit
        for (int p = 0; p < 4; p++) {
            uint32_t player_suit = static_cast<uint32_t>((s->hand_bits[p] >> base) & 0x1FFFULL);
            uint32_t bits = player_suit;
            while (bits) {
                int bit = __builtin_ctz(bits);
                norm_hands[p] |= (1ULL << (base + rank_map[bit]));
                bits &= bits - 1;
            }
        }
    }

    // Primary direct-map key.
    uint64_t h = 0;
    for (int p = 0; p < 4; p++) {
        h ^= norm_hands[p] * (0x9E3779B97F4A7C15ULL + p * 0x6A09E667F3BCC908ULL);
        h = (h << 13) | (h >> 51);
    }
    h ^= zobrist_turn[s->turn];
    h ^= zobrist_leader[s->trick_leader];
    if (s->spades_broken) h ^= zobrist_spades_broken;
    for (int p = 0; p < 4; p++)
        h ^= zobrist_tricks_won[p][s->tricks_won[p]];
    h ^= zobrist_tricks_played[s->tricks_played];

    // Independent collision verifier.
    uint64_t v = 0xCAFEBABEDEADFACEULL;
    for (int p = 0; p < 4; p++) {
        v ^= norm_hands[p];
        v = (v << 17) | (v >> 47);
        v *= 0xBF58476D1CE4E5B9ULL;
    }
    v ^= (uint64_t)s->turn;
    v = (v << 19) | (v >> 45);
    v ^= (uint64_t)s->trick_leader << 3;
    v = (v << 23) | (v >> 41);
    for (int p = 0; p < 4; p++) {
        v ^= (uint64_t)s->tricks_won[p] << (p * 4);
    }
    v ^= (uint64_t)s->spades_broken << 17;
    v ^= (uint64_t)s->tricks_played << 21;
    return {h, v};
}

// ============================================================
// Transposition Table
// ============================================================

static constexpr int TT_SIZE_BITS = 21;  // 2M entries (fits in L3 cache for best hit rate)
static constexpr int TT_SIZE = 1 << TT_SIZE_BITS;
static constexpr int TT_MASK = TT_SIZE - 1;

static constexpr int32_t TT_EXACT = 0;
static constexpr int32_t TT_LOWER_BOUND = 1;
static constexpr int32_t TT_UPPER_BOUND = 2;

struct TTEntry {
    uint64_t key;       // normalized hash
    uint64_t verify;    // collision detection
    double value;
    int32_t flag;
    int32_t best_action;  // best move found (for move ordering)
    int32_t depth;        // remaining tricks when stored (for replacement)
    uint32_t generation;  // generation counter (avoids expensive memset)
};

struct SolverCtx {
    TTEntry* tt;
    int32_t killer[56];       // [depth] killer move
    int32_t history[4][52];   // [player][card_id] history heuristic scores
    double score_range_lower[14][14][16];
    double score_range_upper[14][14][16];
    uint8_t score_range_valid[14][14][16];
    int64_t nodes_searched;
    uint32_t generation;      // current generation (increment to "clear" TT)
    bool owns_tt;             // whether this ctx owns (should free) the TT

    void init() {
        tt = new TTEntry[TT_SIZE]();
        memset(killer, -1, sizeof(killer));
        memset(history, 0, sizeof(history));
        memset(score_range_valid, 0, sizeof(score_range_valid));
        nodes_searched = 0;
        generation = 1;
        owns_tt = true;
    }

    void clear() {
        // O(1) TT invalidation via generation bump
        generation++;
        if (generation == 0) generation = 1;  // skip 0 (default-initialized value)
        memset(killer, -1, sizeof(killer));
        memset(history, 0, sizeof(history));
        memset(score_range_valid, 0, sizeof(score_range_valid));
        nodes_searched = 0;
    }

    void destroy() {
        if (owns_tt && tt) {
            delete[] tt;
        }
        tt = nullptr;
    }

    inline TTEntry& slot(uint64_t key) { return tt[key & TT_MASK]; }

    inline bool slot_valid(const TTEntry& e) const {
        return e.generation == generation;
    }

    inline void mark_slot(TTEntry& e) {
        e.generation = generation;
    }

    inline void reset_score_range_cache() {
        memset(score_range_valid, 0, sizeof(score_range_valid));
    }
};

// ============================================================
// Trick Winner
// ============================================================

static int32_t trick_winner_from_state(const NativeState* s) {
    if (s->table_count == 0) return s->trick_leader;

    int32_t best_pid = s->table_pids[0];
    int32_t best_rank = -1;
    bool has_spade = false;

    for (int32_t i = 0; i < s->table_count; i++) {
        if (s->table_suits[i] == 0) {  // Spades
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
    for (int32_t i = 0; i < s->table_count; i++) {
        if (s->table_suits[i] == lead_suit && s->table_ranks[i] > best_rank) {
            best_rank = s->table_ranks[i];
            best_pid = s->table_pids[i];
        }
    }
    return best_pid;
}

// ============================================================
// Score Evaluation
// ============================================================

static double evaluate_score_diff(const NativeState* s) {
    double team_scores[2] = {0.0, 0.0};
    for (int32_t team_id = 0; team_id < 2; team_id++) {
        int32_t team_bid = 0;
        int32_t team_tricks = 0;
        double score = 0.0;
        for (int32_t pid = 0; pid < 4; pid++) {
            if (s->teams[pid] != team_id) continue;
            team_tricks += s->tricks_won[pid];
            int32_t bid = s->max_bid[pid];
            if (bid == 14) {  // blind nil
                score += (s->tricks_won[pid] == 0) ? 100.0 : -100.0;
            } else if (bid == 0) {  // nil
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


// A score range over a SUPERSET of every reachable terminal outcome.
//
// We enumerate every possible split of the remaining tricks between the two
// teams.  Each unresolved nil is independently allowed to succeed or fail.
// Some combinations are impossible, but including extra outcomes can only
// widen [lower, upper], never exclude the true minimax value.  Therefore:
//   lower >= beta  => safe fail-high
//   upper <= alpha => safe fail-low
//
// This remains sound with the non-monotone overtrick penalty and at partially
// played tricks; unlike the old quick-trick path it makes no claim that a
// voluntarily cashable winner is unavoidable.
static bool prune_with_global_score_range(
    const NativeState* s,
    double alpha,
    double beta,
    SolverCtx& ctx,
    double* pruned_value
) {
    const int32_t remaining = 13 - s->tricks_played;
    if (remaining <= 0) return false;

    int32_t team_bid[2] = {0, 0};
    int32_t current_tricks[2] = {0, 0};
    uint32_t nil_failed_mask = 0;
    double nil_lower = 0.0;
    double nil_upper = 0.0;

    for (int32_t player = 0; player < 4; player++) {
        const int32_t team = s->teams[player];
        const int32_t bid = s->max_bid[player];
        current_tricks[team] += s->tricks_won[player];
        if (
            (bid == 0 || bid == 14)
            && s->tricks_won[player] > 0
        ) {
            nil_failed_mask |= 1u << player;
        }
        if (bid != 0 && bid != 14) {
            team_bid[team] += bid;
            continue;
        }

        const double bonus = bid == 14 ? 100.0 : 50.0;
        if (s->tricks_won[player] > 0) {
            // Failed nil is fixed: -bonus for team 0, +bonus in score
            // difference when the failed bidder belongs to team 1.
            const double contribution = team == 0 ? -bonus : bonus;
            nil_lower += contribution;
            nil_upper += contribution;
        } else {
            // The unresolved nil may contribute either sign to score_diff.
            nil_lower -= bonus;
            nil_upper += bonus;
        }
    }

    const int32_t tricks_played = s->tricks_played;
    const int32_t team0_current = current_tricks[0];
    const int32_t nil_index = static_cast<int32_t>(
        nil_failed_mask & 0xFu
    );
    double lower;
    double upper;
    if (
        ctx.score_range_valid[
            tricks_played
        ][team0_current][nil_index]
    ) {
        lower = ctx.score_range_lower[
            tricks_played
        ][team0_current][nil_index];
        upper = ctx.score_range_upper[
            tricks_played
        ][team0_current][nil_index];
    } else {
        lower = std::numeric_limits<double>::infinity();
        upper = -std::numeric_limits<double>::infinity();
        for (int32_t team0_extra = 0;
             team0_extra <= remaining;
             team0_extra++) {
            const int32_t final_tricks[2] = {
                current_tricks[0] + team0_extra,
                current_tricks[1] + remaining - team0_extra,
            };
            double contract_score[2] = {0.0, 0.0};
            for (int32_t team = 0; team < 2; team++) {
                if (team_bid[team] == 0) continue;
                if (final_tricks[team] >= team_bid[team]) {
                    contract_score[team] =
                        team_bid[team] * 10.0
                        - (
                            final_tricks[team] - team_bid[team]
                        ) * 9.0;
                } else {
                    contract_score[team] = -team_bid[team] * 10.0;
                }
            }
            const double contract_diff =
                contract_score[0] - contract_score[1];
            lower = std::min(lower, contract_diff + nil_lower);
            upper = std::max(upper, contract_diff + nil_upper);
        }
        ctx.score_range_lower[
            tricks_played
        ][team0_current][nil_index] = lower;
        ctx.score_range_upper[
            tricks_played
        ][team0_current][nil_index] = upper;
        ctx.score_range_valid[
            tricks_played
        ][team0_current][nil_index] = 1;
    }

    if (lower >= beta) {
        *pruned_value = lower;
        return true;
    }
    if (upper <= alpha) {
        *pruned_value = upper;
        return true;
    }
    return false;
}

// ============================================================
// Equivalent Card Filtering
// ============================================================

static void filter_equivalent(const NativeState* s, int32_t player_id, ActionList& actions) {
    if (actions.count <= 1) return;

    const uint64_t my_hand = s->hand_bits[player_id];
    uint64_t others = 0;
    for (int32_t p = 0; p < 4; p++) {
        if (p != player_id) others |= s->hand_bits[p];
    }
    for (int32_t i = 0; i < s->table_count; i++) {
        int32_t cid = s->table_suits[i] * 13 + (s->table_ranks[i] - 2);
        others |= (1ULL << cid);
    }

    uint64_t remove_mask = 0;
    for (int32_t suit = 0; suit < 4; suit++) {
        int32_t base = suit * 13;
        uint32_t my_ranks = static_cast<uint32_t>((my_hand >> base) & 0x1FFFULL);
        uint32_t ot_ranks = static_cast<uint32_t>((others >> base) & 0x1FFFULL);
        if (my_ranks == 0) continue;

        int32_t prev_my_bit = -1;
        for (int32_t bit = 12; bit >= 0; bit--) {
            if (!(my_ranks & (1u << bit))) continue;
            if (prev_my_bit < 0) {
                prev_my_bit = bit;
                continue;
            }
            // Bits strictly between the two ranks.  Both shifts are <= 12.
            const uint32_t between_mask =
                ((1u << prev_my_bit) - 1u)
                ^ ((1u << (bit + 1)) - 1u);
            const bool has_gap = (ot_ranks & between_mask) != 0;
            if (!has_gap) {
                remove_mask |= (1ULL << (base + bit));
            } else {
                prev_my_bit = bit;
            }
        }
    }

    if (remove_mask == 0) return;

    int32_t write = 0;
    for (int32_t read = 0; read < actions.count; read++) {
        if (!(remove_mask & (1ULL << actions.cards[read]))) {
            actions.cards[write++] = actions.cards[read];
        }
    }
    actions.count = write;
}

// ============================================================
// Legal Action Generation
// ============================================================

static void legal_actions_impl(const NativeState* s, int32_t player_id,
                               ActionList& actions, bool remove_equivalent) {
    actions.clear();
    uint64_t allowed = s->hand_bits[player_id];
    if (allowed == 0) return;

    if (s->table_count == 0) {
        // Leading: before Spades are broken, use a non-Spade when possible.
        if (!s->spades_broken) {
            const uint64_t non_spades = allowed & ~0x1FFFULL;
            if (non_spades != 0) allowed = non_spades;
        }
    } else {
        // Following: follow the led suit whenever possible.
        const int32_t lead_suit = s->table_suits[0];
        uint64_t mask = 0x1FFFULL << (lead_suit * 13);
        if ((allowed & mask) != 0) allowed &= mask;
    }

    while (allowed) {
        int cid = ctz64(allowed);
        actions.push(cid);
        allowed &= allowed - 1;
    }
    if (remove_equivalent) filter_equivalent(s, player_id, actions);
}

static void legal_actions(const NativeState* s, int32_t player_id,
                          ActionList& actions) {
#ifdef SPADES_DISABLE_EQUIVALENT_FILTER
    legal_actions_impl(s, player_id, actions, false);
#else
    legal_actions_impl(s, player_id, actions, true);
#endif
}

static void legal_actions_all(const NativeState* s, int32_t player_id,
                              ActionList& actions) {
    legal_actions_impl(s, player_id, actions, false);
}

// ============================================================
// Move Ordering
// ============================================================

// Determine who is currently winning the trick
static int32_t current_trick_winner(const NativeState* s) {
    if (s->table_count == 0) return -1;
    return trick_winner_from_state(s);
}

// Position-aware move scoring
static void score_moves(const NativeState* s, int32_t player_id, ActionList& actions,
                        int32_t scores[13], SolverCtx& ctx, int32_t tt_best, int32_t depth) {
    const int32_t team = s->teams[player_id];
    const int32_t table_count = s->table_count;
    uint64_t all_remaining = 0;
    int32_t winner_team = -1;
    int32_t best_lead_rank = -1;
    bool winning_has_spade = false;
    int32_t player_suit_counts[4] = {0, 0, 0, 0};

    if (table_count == 0) {
        for (int p = 0; p < 4; p++) {
            all_remaining |= s->hand_bits[p];
        }
    } else {
        const int32_t winner_pid = current_trick_winner(s);
        winner_team = (
            winner_pid >= 0 ? s->teams[winner_pid] : -1
        );
        const int32_t lead_suit = s->table_suits[0];
        for (int k = 0; k < table_count; k++) {
            if (s->table_suits[k] == 0 && lead_suit != 0) {
                winning_has_spade = true;
            }
            if (
                s->table_suits[k] == lead_suit
                && s->table_ranks[k] > best_lead_rank
            ) {
                best_lead_rank = s->table_ranks[k];
            }
        }
        for (int suit = 0; suit < 4; suit++) {
            player_suit_counts[suit] = popcount64(
                suit_ranks(s->hand_bits[player_id], suit)
            );
        }
    }

    for (int i = 0; i < actions.count; i++) {
        int32_t cid = actions.cards[i];
        int32_t score = 0;

        // Highest priority: TT best move
        if (cid == tt_best) {
            scores[i] = 100000;
            continue;
        }

        // Second: killer move
        if (cid == ctx.killer[depth]) {
            scores[i] = 90000;
            continue;
        }

        // History heuristic base
        score = ctx.history[player_id][cid];

        if (table_count == 0) {
            // === LEADING ===
            // Prefer: master cards (highest in suit), aces, high spades
            int32_t suit = card_suit(cid);
            int32_t rank = card_rank(cid);

            // Check if this card is the master (highest remaining) in its suit
            uint32_t suit_remaining = suit_ranks(all_remaining, suit);
            int top_bit = 31 - __builtin_clz(suit_remaining);
            if ((rank - 2) == top_bit) {
                score += 5000;  // Master card - guaranteed winner (if no trump cut)
            }

            // High cards bonus
            score += rank * 100;

            // Trump (spades) cards get moderate priority for leading
            if (suit == 0) score += 2000;
        } else {
            // === FOLLOWING ===
            int32_t suit = card_suit(cid);
            int32_t rank = card_rank(cid);
            int32_t lead_suit = s->table_suits[0];

            bool partner_winning = (winner_team == team);

            if (suit == lead_suit) {
                // Following suit
                if (partner_winning) {
                    // Partner is winning - play LOW to save high cards
                    score += (14 - rank) * 100;  // Lower rank = higher score
                } else {
                    // Opponent winning - try to WIN with minimum card that beats them
                    if (!winning_has_spade && rank > best_lead_rank) {
                        // Can win! Prefer the minimum winning card
                        score += 4000 + (14 - rank) * 50;
                    } else {
                        // Can't win - play low
                        score += (14 - rank) * 50;
                    }
                }
            } else if (suit == 0) {
                // Trumping (playing spade when can't follow)
                if (partner_winning) {
                    // Don't waste trump if partner is winning
                    score += (14 - rank) * 10;
                } else {
                    // Trump to win! Prefer lowest trump that wins
                    score += 6000 + (14 - rank) * 100;
                }
            } else {
                // Discarding (not following suit, not trumping)
                if (partner_winning) {
                    // Discard low
                    score += (14 - rank) * 10;
                } else {
                    // Discard: prefer short suit cards, low cards
                    // Count cards in this suit (prefer shorter suits)
                    int32_t my_suit_count = player_suit_counts[suit];
                    score += (14 - rank) * 10 + (13 - my_suit_count) * 20;
                }
            }
        }

        scores[i] = score;
    }
}

// Sort actions by scores (selection sort for small arrays - cache friendly)
static void sort_actions_by_score(ActionList& actions, int32_t scores[13]) {
    for (int i = 0; i < actions.count - 1; i++) {
        int best = i;
        for (int j = i + 1; j < actions.count; j++) {
            if (scores[j] > scores[best]) best = j;
        }
        if (best != i) {
            std::swap(actions.cards[i], actions.cards[best]);
            std::swap(scores[i], scores[best]);
        }
    }
}

// ============================================================
// Make / Unmake Move
// ============================================================

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

    s->hand_bits[player_id] &= ~u.hand_bit;
    s->table_pids[s->table_count] = player_id;
    s->table_suits[s->table_count] = card_suit(action_cid);
    s->table_ranks[s->table_count] = card_rank(action_cid);
    s->table_count += 1;
    s->turn = (player_id + 1) % 4;
    s->spades_broken = s->spades_broken || (card_suit(action_cid) == 0);

    if (s->table_count == 4) {
        u.trick_completed = true;
        for (int k = 0; k < 4; k++) {
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
        for (int k = 0; k < 4; k++) {
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

// ============================================================
// Forced-Outcome Search
//
// This search answers a different question from minimax: whether EVERY legal
// continuation has the same team trick total and Nil outcomes.  It therefore
// uses all legal cards and only exact signature-union pruning.
// ============================================================

struct ForcedValue {
    int32_t status;
    int32_t team0_final_tricks;
    uint32_t nil_broken_mask;
};

static constexpr int FORCED_TT_SIZE_BITS = 20;
static constexpr int FORCED_TT_SIZE = 1 << FORCED_TT_SIZE_BITS;
static constexpr int FORCED_TT_MASK = FORCED_TT_SIZE - 1;

struct ForcedTTEntry {
    uint64_t key;
    uint64_t verify;
    int32_t status;
    int32_t team0_final_tricks;
    uint32_t nil_broken_mask;
    uint32_t generation;
};

struct ForcedContext {
    ForcedTTEntry* tt;
    uint32_t generation;
    uint64_t nodes_searched;
    std::chrono::steady_clock::time_point deadline;

    inline ForcedTTEntry& slot(uint64_t key) {
        return tt[key & FORCED_TT_MASK];
    }
};

static ForcedTTEntry* g_forced_tt_buffer = nullptr;
static uint32_t g_forced_generation = 0;

static void ensure_forced_tt_buffer() {
    if (!g_forced_tt_buffer) {
        g_forced_tt_buffer = new ForcedTTEntry[FORCED_TT_SIZE]();
    }
}

static uint64_t forced_verify(const NativeState* s) {
    uint64_t value = 0xD6E8FEB86659FD93ULL;
    auto mix = [&value](uint64_t part) {
        value ^= part + 0x9E3779B97F4A7C15ULL + (value << 6) + (value >> 2);
        value ^= value >> 30;
        value *= 0xBF58476D1CE4E5B9ULL;
        value ^= value >> 27;
    };

    for (int32_t p = 0; p < 4; p++) {
        mix(s->hand_bits[p]);
        mix(static_cast<uint64_t>(s->tricks_won[p]) | (static_cast<uint64_t>(p) << 8));
    }
    for (int32_t i = 0; i < s->table_count; i++) {
        const uint64_t cid = static_cast<uint64_t>(
            s->table_suits[i] * 13 + (s->table_ranks[i] - 2)
        );
        mix(cid | (static_cast<uint64_t>(s->table_pids[i]) << 8) |
            (static_cast<uint64_t>(i) << 16));
    }
    mix(static_cast<uint64_t>(s->table_count));
    mix(static_cast<uint64_t>(s->turn) | (static_cast<uint64_t>(s->trick_leader) << 8));
    mix(static_cast<uint64_t>(s->spades_broken) |
        (static_cast<uint64_t>(s->tricks_played) << 8));
    return value;
}

static ForcedValue terminal_forced_value(const NativeState* s) {
    int32_t team0_tricks = 0;
    uint32_t nil_mask = 0;
    for (int32_t p = 0; p < 4; p++) {
        if (s->teams[p] == 0) team0_tricks += s->tricks_won[p];
        if ((s->max_bid[p] == 0 || s->max_bid[p] == 14) && s->tricks_won[p] > 0) {
            nil_mask |= (1u << p);
        }
    }
    return {FORCED_FIXED, team0_tricks, nil_mask};
}

static bool forced_tt_lookup(const NativeState* s, ForcedContext& ctx,
                             ForcedValue* output) {
    const uint64_t key = compute_full_hash(s);
    ForcedTTEntry& entry = ctx.slot(key);
    if (entry.generation != ctx.generation || entry.key != key ||
        entry.verify != forced_verify(s)) {
        return false;
    }
    *output = {
        entry.status,
        entry.team0_final_tricks,
        entry.nil_broken_mask,
    };
    return true;
}

static void forced_tt_store(const NativeState* s, ForcedContext& ctx,
                            const ForcedValue& value) {
    if (value.status == FORCED_TIMEOUT) return;
    const uint64_t key = compute_full_hash(s);
    ForcedTTEntry& entry = ctx.slot(key);
    entry.key = key;
    entry.verify = forced_verify(s);
    entry.status = value.status;
    entry.team0_final_tricks = value.team0_final_tricks;
    entry.nil_broken_mask = value.nil_broken_mask;
    entry.generation = ctx.generation;
}

static ForcedValue forced_search(NativeState* s, ForcedContext& ctx) {
    ctx.nodes_searched++;
    if (std::chrono::steady_clock::now() >= ctx.deadline) {
        return {FORCED_TIMEOUT, -1, 0};
    }
    if (s->tricks_played >= 13) {
        return terminal_forced_value(s);
    }

    ForcedValue cached;
    if (forced_tt_lookup(s, ctx, &cached)) return cached;

    ActionList actions;
    legal_actions_all(s, s->turn, actions);
    if (actions.count == 0) {
        return {FORCED_TIMEOUT, -1, 0};
    }

    bool have_signature = false;
    ForcedValue first{FORCED_FIXED, -1, 0};

    for (int32_t i = 0; i < actions.count; i++) {
        const UndoInfo undo = make_move(s, s->turn, actions.cards[i]);
        const ForcedValue child = forced_search(s, ctx);
        unmake_move(s, undo);

        if (child.status == FORCED_TIMEOUT) return child;
        if (child.status == FORCED_VARIABLE) {
            const ForcedValue variable{FORCED_VARIABLE, -1, 0};
            forced_tt_store(s, ctx, variable);
            return variable;
        }
        if (!have_signature) {
            first = child;
            have_signature = true;
        } else if (first.team0_final_tricks != child.team0_final_tricks ||
                   first.nil_broken_mask != child.nil_broken_mask) {
            const ForcedValue variable{FORCED_VARIABLE, -1, 0};
            forced_tt_store(s, ctx, variable);
            return variable;
        }
    }

    forced_tt_store(s, ctx, first);
    return first;
}

// ============================================================
// Core Minimax with all optimizations
// ============================================================

static double minimax(NativeState* s, double alpha, double beta, SolverCtx& ctx) {
    ctx.nodes_searched++;

    // Terminal check
    if (s->tricks_played >= 13) return evaluate_score_diff(s);

    const double alpha_original = alpha;
    const double beta_original = beta;
    int32_t current_player = s->turn;
    int32_t current_team = s->teams[current_player];
    int32_t depth = s->tricks_played * 4 + s->table_count;

    // === TT lookup (only at trick boundaries for higher hit rate) ===
#ifdef SPADES_DISABLE_TT
    bool use_tt = false;
#else
    bool use_tt = (s->table_count == 0);
#endif
    uint64_t tt_key = 0, tt_verify = 0;
    int32_t tt_best = -1;

    if (use_tt) {
        const NormalizedTTKey normalized = compute_normalized_tt_key(s);
        tt_key = normalized.hash;
        tt_verify = normalized.verify;
        TTEntry& slot = ctx.slot(tt_key);
        if (ctx.slot_valid(slot) && slot.key == tt_key) {
            if (slot.verify == tt_verify) {
                if (slot.flag == TT_EXACT) return slot.value;
                if (slot.flag == TT_LOWER_BOUND && slot.value >= beta) return slot.value;
                if (slot.flag == TT_UPPER_BOUND && slot.value <= alpha) return slot.value;
                // Narrow the window
                if (slot.flag == TT_LOWER_BOUND && slot.value > alpha) alpha = slot.value;
                if (slot.flag == TT_UPPER_BOUND && slot.value < beta) beta = slot.value;
                tt_best = slot.best_action;
            }
        }
    }

#ifndef SPADES_DISABLE_GLOBAL_SCORE_RANGE
    double score_range_value;
    if (prune_with_global_score_range(
            s, alpha, beta, ctx, &score_range_value)) {
        return score_range_value;
    }
#endif

    // === Generate legal actions ===
    ActionList actions;
    legal_actions(s, current_player, actions);
    if (actions.count == 0) return evaluate_score_diff(s);

    // Single action optimization
    if (actions.count == 1) {
        UndoInfo undo = make_move(s, current_player, actions.cards[0]);
        double value = minimax(s, alpha, beta, ctx);
        unmake_move(s, undo);
        return value;
    }

    // === Move ordering ===
    int32_t scores[13];
    score_moves(s, current_player, actions, scores, ctx, tt_best, depth);
    sort_actions_by_score(actions, scores);

    // === PVS Search ===
    double value;
    int32_t best_action_here = actions.cards[0];
    bool first_child = true;

    if (current_team == 0) {
        // Maximizing
        value = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < actions.count; i++) {
            int32_t action = actions.cards[i];
            UndoInfo undo = make_move(s, current_player, action);
            double child_value;

            if (first_child) {
                child_value = minimax(s, alpha, beta, ctx);
                first_child = false;
            } else {
                // Null-window search
                child_value = minimax(s, alpha, alpha + 1.0, ctx);
                if (child_value > alpha && child_value < beta) {
                    // Re-search with full window
                    child_value = minimax(s, alpha, beta, ctx);
                }
            }

            unmake_move(s, undo);

            if (child_value > value) {
                value = child_value;
                best_action_here = action;
            }
            if (value > alpha) alpha = value;
            if (value >= beta) {
                // Update killer
                ctx.killer[depth] = action;
                // Update history (deeper cuts get more bonus)
                int remaining = 13 - s->tricks_played;
                ctx.history[current_player][action] += remaining * remaining;
                break;
            }
        }
    } else {
        // Minimizing
        value = std::numeric_limits<double>::infinity();
        for (int i = 0; i < actions.count; i++) {
            int32_t action = actions.cards[i];
            UndoInfo undo = make_move(s, current_player, action);
            double child_value;

            if (first_child) {
                child_value = minimax(s, alpha, beta, ctx);
                first_child = false;
            } else {
                child_value = minimax(s, beta - 1.0, beta, ctx);
                if (child_value < beta && child_value > alpha) {
                    child_value = minimax(s, alpha, beta, ctx);
                }
            }

            unmake_move(s, undo);

            if (child_value < value) {
                value = child_value;
                best_action_here = action;
            }
            if (value < beta) beta = value;
            if (value <= alpha) {
                ctx.killer[depth] = action;
                int remaining = 13 - s->tricks_played;
                ctx.history[current_player][action] += remaining * remaining;
                break;
            }
        }
    }

    // === Store in TT (only at trick boundaries) ===
    if (use_tt) {
        TTEntry& slot = ctx.slot(tt_key);
        int32_t new_depth = 13 - s->tricks_played;
        if (!ctx.slot_valid(slot) || new_depth >= slot.depth) {
            slot.key = tt_key;
            slot.verify = tt_verify;
            slot.value = value;
            if (value <= alpha_original) {
                slot.flag = TT_UPPER_BOUND;
            } else if (value >= beta_original) {
                slot.flag = TT_LOWER_BOUND;
            } else {
                slot.flag = TT_EXACT;
            }
            slot.best_action = best_action_here;
            slot.depth = new_depth;
            ctx.mark_slot(slot);
        }
    }

    return value;
}

// ============================================================
// Null-Window Minimax (no PVS - for partition search iterations)
//
// Key differences from full minimax:
// 1. NO PVS (plain alpha-beta) - PVS conflicts with tight windows
// 2. Designed for repeated null-window calls where TT accumulates
// 3. Same TT, killer, history, but each iteration refines bounds
// ============================================================

static double minimax_nw(NativeState* s, double alpha, double beta, SolverCtx& ctx) {
    ctx.nodes_searched++;

    if (s->tricks_played >= 13) return evaluate_score_diff(s);

    const double alpha_original = alpha;
    const double beta_original = beta;
    int32_t current_player = s->turn;
    int32_t current_team = s->teams[current_player];
    int32_t depth = s->tricks_played * 4 + s->table_count;

    // TT lookup (only at trick boundaries)
#ifdef SPADES_DISABLE_TT
    bool use_tt = false;
#else
    bool use_tt = (s->table_count == 0);
#endif
    uint64_t tt_key = 0, tt_verify = 0;
    int32_t tt_best = -1;

    if (use_tt) {
        const NormalizedTTKey normalized = compute_normalized_tt_key(s);
        tt_key = normalized.hash;
        tt_verify = normalized.verify;
        TTEntry& slot = ctx.slot(tt_key);
        if (ctx.slot_valid(slot) && slot.key == tt_key) {
            if (slot.verify == tt_verify) {
                if (slot.flag == TT_EXACT) return slot.value;
                if (slot.flag == TT_LOWER_BOUND && slot.value >= beta) return slot.value;
                if (slot.flag == TT_UPPER_BOUND && slot.value <= alpha) return slot.value;
                // Narrow window from TT bounds
                if (slot.flag == TT_LOWER_BOUND && slot.value > alpha) alpha = slot.value;
                if (slot.flag == TT_UPPER_BOUND && slot.value < beta) beta = slot.value;
                // CRITICAL: check if window became invalid after narrowing
                if (alpha >= beta) return slot.value;
                tt_best = slot.best_action;
            }
        }
    }

#ifndef SPADES_DISABLE_GLOBAL_SCORE_RANGE
    double score_range_value;
    if (prune_with_global_score_range(
            s, alpha, beta, ctx, &score_range_value)) {
        return score_range_value;
    }
#endif

    // Generate actions
    ActionList actions;
    legal_actions(s, current_player, actions);
    if (actions.count == 0) return evaluate_score_diff(s);

    // Single action - no branching needed
    if (actions.count == 1) {
        UndoInfo undo = make_move(s, current_player, actions.cards[0]);
        double value = minimax_nw(s, alpha, beta, ctx);
        unmake_move(s, undo);
        return value;
    }

    // Move ordering (same as full minimax)
    int32_t scores[13];
    score_moves(s, current_player, actions, scores, ctx, tt_best, depth);
    sort_actions_by_score(actions, scores);

    // Plain alpha-beta (NO PVS - critical for null-window correctness)
    double value;
    int32_t best_action_here = actions.cards[0];

    if (current_team == 0) {
        value = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < actions.count; i++) {
            int32_t action = actions.cards[i];
            UndoInfo undo = make_move(s, current_player, action);
            double child_value = minimax_nw(s, alpha, beta, ctx);
            unmake_move(s, undo);

            if (child_value > value) {
                value = child_value;
                best_action_here = action;
            }
            if (value > alpha) alpha = value;
            if (value >= beta) {
                ctx.killer[depth] = action;
                int rem = 13 - s->tricks_played;
                ctx.history[current_player][action] += rem * rem;
                break;
            }
        }
    } else {
        value = std::numeric_limits<double>::infinity();
        for (int i = 0; i < actions.count; i++) {
            int32_t action = actions.cards[i];
            UndoInfo undo = make_move(s, current_player, action);
            double child_value = minimax_nw(s, alpha, beta, ctx);
            unmake_move(s, undo);

            if (child_value < value) {
                value = child_value;
                best_action_here = action;
            }
            if (value < beta) beta = value;
            if (value <= alpha) {
                ctx.killer[depth] = action;
                int rem = 13 - s->tricks_played;
                ctx.history[current_player][action] += rem * rem;
                break;
            }
        }
    }

    // Store in TT
    if (use_tt) {
        TTEntry& slot = ctx.slot(tt_key);
        int32_t new_depth = 13 - s->tricks_played;
        if (!ctx.slot_valid(slot) || new_depth >= slot.depth) {
            slot.key = tt_key;
            slot.verify = tt_verify;
            slot.value = value;
            if (value <= alpha_original) {
                slot.flag = TT_UPPER_BOUND;
            } else if (value >= beta_original) {
                slot.flag = TT_LOWER_BOUND;
            } else {
                slot.flag = TT_EXACT;
            }
            slot.best_action = best_action_here;
            slot.depth = new_depth;
            ctx.mark_slot(slot);
        }
    }

    return value;
}

// ============================================================
// Partition Search Driver (Binary-search MTD with null-window)
//
// Uses iterative null-window searches to converge on the exact score.
// The TT persists across iterations, accumulating partition knowledge.
// Each iteration is fast because:
//   - Null-window produces more cutoffs
//   - Prior iterations populated the TT with useful bounds
// ============================================================

static double solve_partition_search(NativeState* s, SolverCtx& ctx) {
    // For large games (>20 cards remaining), use iterative null-window.
    // Strategy: do ONE full-window search first to get an initial result,
    // then use null-window to verify/refine. The full-window search populates
    // the TT, making subsequent null-window passes very fast.
    //
    // Actually, the most practical approach for Spades:
    // Use the full PVS minimax for the first call (which benefits from all
    // optimizations). The "partition" benefit comes from the normalized TT
    // which already groups equivalent positions.
    //
    // The null-window approach is most beneficial for solve_with_q where
    // multiple root actions share a TT. For single solve, PVS is already optimal.

    return minimax(s, -std::numeric_limits<double>::infinity(),
                  std::numeric_limits<double>::infinity(), ctx);
}

// ============================================================
// MTD(f) using null-window minimax (kept for reference/comparison)
// ============================================================

static double mtdf(NativeState* s, double first_guess, SolverCtx& ctx) {
    double g = first_guess;
    double upper_bound = 500.0;   // Spades max possible: ~200 + margin
    double lower_bound = -500.0;

    int iterations = 0;
    const int max_iter = 50;  // Safety limit (scores are integers, range ~1000)

    while (upper_bound - lower_bound > 0.5 && iterations < max_iter) {
        double beta;
        if (g <= lower_bound) {
            beta = lower_bound + 1.0;
        } else if (g >= upper_bound) {
            beta = upper_bound;
        } else {
            beta = g;
        }

        // Zero-window search around beta
        g = minimax(s, beta - 1.0, beta, ctx);

        if (g < beta) {
            upper_bound = g;
        } else {
            lower_bound = g;
        }
        iterations++;
    }
    return g;
}

// ============================================================
// Public API - uses static TT (allocated once, cleared via generation)
// ============================================================

// Global TT buffer (allocated once on first use)
static TTEntry* g_tt_buffer = nullptr;
static void ensure_tt_buffer() {
    if (!g_tt_buffer) {
        g_tt_buffer = new TTEntry[TT_SIZE]();
    }
}

void analyze_forced_outcome_native(const NativeState* input,
                                   int64_t budget_microseconds,
                                   ForcedOutcomeResult* output) {
    if (!output) return;

    const auto started = std::chrono::steady_clock::now();
    output->status = FORCED_TIMEOUT;
    output->team0_final_tricks = -1;
    output->nil_broken_mask = 0;
    output->nodes_searched = 0;
    output->elapsed_ms = 0.0;

    if (!input || budget_microseconds <= 0) return;

    init_zobrist();
    ensure_forced_tt_buffer();
    g_forced_generation++;
    if (g_forced_generation == 0) g_forced_generation = 1;

    ForcedContext ctx;
    ctx.tt = g_forced_tt_buffer;
    ctx.generation = g_forced_generation;
    ctx.nodes_searched = 0;
    ctx.deadline = started + std::chrono::microseconds(budget_microseconds);

    NativeState state = *input;
    const ForcedValue value = forced_search(&state, ctx);

    output->status = value.status;
    output->team0_final_tricks = value.team0_final_tricks;
    output->nil_broken_mask = value.nil_broken_mask;
    output->nodes_searched = ctx.nodes_searched;
    output->elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started
    ).count();
}

double solve_native(const NativeState* input) {
    init_zobrist();
    ensure_tt_buffer();

    SolverCtx ctx;
    ctx.tt = g_tt_buffer;
    ctx.owns_tt = false;
    ctx.nodes_searched = 0;
    memset(ctx.killer, -1, sizeof(ctx.killer));
    memset(ctx.history, 0, sizeof(ctx.history));
    ctx.reset_score_range_cache();
    static uint32_t g_gen = 0;
    g_gen++;
    if (g_gen == 0) g_gen = 1;
    ctx.generation = g_gen;

    NativeState s = *input;

    // Full-window PVS with normalized TT and admissible global score bounds.
    // Exact score-partition search remains disabled until its per-root Q-value
    // convergence is independently verified against the exhaustive oracle.
    double result = minimax(&s, -std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity(), ctx);
    return result;
}

void solve_native_with_q(const NativeState* input, RootQResult* out_result) {
    init_zobrist();
    ensure_tt_buffer();

    NativeState s = *input;
    out_result->current_player = s.turn;
    out_result->optimize_for_team = s.teams[s.turn];

    ActionList actions;
    legal_actions(&s, s.turn, actions);
    out_result->count = actions.count;

    if (actions.count == 0) {
        static uint32_t gen_q = 1000000;
        gen_q++;
        SolverCtx ctx;
        ctx.tt = g_tt_buffer;
        ctx.owns_tt = false;
        ctx.generation = gen_q;
        ctx.nodes_searched = 0;
        memset(ctx.killer, -1, sizeof(ctx.killer));
        memset(ctx.history, 0, sizeof(ctx.history));
        ctx.reset_score_range_cache();
        out_result->value = minimax(&s, -std::numeric_limits<double>::infinity(),
                                    std::numeric_limits<double>::infinity(), ctx);
        out_result->best_action = -1;
        return;
    }

    // Sort actions by heuristic score
    int32_t scores[13];
    {
        SolverCtx tmp_ctx;
        tmp_ctx.tt = g_tt_buffer;
        tmp_ctx.owns_tt = false;
        tmp_ctx.generation = 0;  // won't match anything - just for scoring
        tmp_ctx.nodes_searched = 0;
        memset(tmp_ctx.killer, -1, sizeof(tmp_ctx.killer));
        memset(tmp_ctx.history, 0, sizeof(tmp_ctx.history));
        score_moves(&s, s.turn, actions, scores, tmp_ctx, -1, s.tricks_played * 4);
        sort_actions_by_score(actions, scores);
    }

    int32_t n = actions.count;
    bool maximize = (out_result->optimize_for_team == 0);
    double q_values[13];

    int remaining = 0;
    for (int p = 0; p < 4; p++) remaining += popcount64(s.hand_bits[p]);

    // Use a generation counter for solve_with_q calls
    static uint32_t gen_wq = 2000000;

    if (n == 1) {
        gen_wq++;
        SolverCtx ctx;
        ctx.tt = g_tt_buffer;
        ctx.owns_tt = false;
        ctx.generation = gen_wq;
        ctx.nodes_searched = 0;
        memset(ctx.killer, -1, sizeof(ctx.killer));
        memset(ctx.history, 0, sizeof(ctx.history));
        ctx.reset_score_range_cache();

        UndoInfo undo = make_move(&s, s.turn, actions.cards[0]);
        q_values[0] = minimax(&s, -std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity(), ctx);
        unmake_move(&s, undo);
    } else if (n <= 3 || remaining <= 52) {
        // Sequential for ALL practical sizes. Parallel path disabled because:
        // 1. Thread TT allocation (160MB each) dwarfs search time for ≤44 cards
        // 2. Shared TT accumulates knowledge across actions (huge benefit)
        // 3. Every returned Q value is searched with a full window
        gen_wq++;
        SolverCtx ctx;
        ctx.tt = g_tt_buffer;
        ctx.owns_tt = false;
        ctx.generation = gen_wq;
        ctx.nodes_searched = 0;
        memset(ctx.killer, -1, sizeof(ctx.killer));
        memset(ctx.history, 0, sizeof(ctx.history));
        ctx.reset_score_range_cache();

        for (int i = 0; i < n; i++) {
            UndoInfo undo = make_move(&s, s.turn, actions.cards[i]);
            q_values[i] = minimax(
                &s,
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(),
                ctx
            );
            unmake_move(&s, undo);
        }
    } else {
        // Parallel: first action sequential, rest in parallel with own TTs
        gen_wq++;
        {
            SolverCtx ctx;
            ctx.tt = g_tt_buffer;
            ctx.owns_tt = false;
            ctx.generation = gen_wq;
            ctx.nodes_searched = 0;
            memset(ctx.killer, -1, sizeof(ctx.killer));
            memset(ctx.history, 0, sizeof(ctx.history));
            ctx.reset_score_range_cache();

            UndoInfo undo = make_move(&s, s.turn, actions.cards[0]);
            q_values[0] = minimax(&s, -std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity(), ctx);
            unmake_move(&s, undo);
        }

        double baseline = q_values[0];

        // Parallel search for remaining actions
        std::vector<std::thread> threads;
        // Each thread gets its own TT (necessary for correctness)
        std::vector<TTEntry*> thread_tts(n - 1);
        for (int i = 0; i < n - 1; i++) {
            thread_tts[i] = new TTEntry[TT_SIZE]();
        }

        for (int i = 1; i < n; i++) {
            threads.emplace_back([&, i]() {
                SolverCtx local_ctx;
                local_ctx.tt = thread_tts[i-1];
                local_ctx.owns_tt = false;
                local_ctx.generation = 1;
                local_ctx.nodes_searched = 0;
                memset(local_ctx.killer, -1, sizeof(local_ctx.killer));
                memset(local_ctx.history, 0, sizeof(local_ctx.history));
                local_ctx.reset_score_range_cache();

                NativeState local_s = s;
                UndoInfo undo = make_move(&local_s, local_s.turn, actions.cards[i]);

                if (maximize) {
                    q_values[i] = minimax(&local_s, baseline, baseline + 1.0, local_ctx);
                    if (q_values[i] > baseline) {
                        local_ctx.clear();
                        q_values[i] = minimax(&local_s, baseline,
                                             std::numeric_limits<double>::infinity(), local_ctx);
                    }
                } else {
                    q_values[i] = minimax(&local_s, baseline - 1.0, baseline, local_ctx);
                    if (q_values[i] < baseline) {
                        local_ctx.clear();
                        q_values[i] = minimax(&local_s, -std::numeric_limits<double>::infinity(),
                                             baseline, local_ctx);
                    }
                }
                unmake_move(&local_s, undo);
            });
        }
        for (auto& t : threads) t.join();
        for (auto* p : thread_tts) delete[] p;
    }

    // Collect results
    double best_value = maximize ? -std::numeric_limits<double>::infinity()
                                 : std::numeric_limits<double>::infinity();
    int32_t best_action = actions.cards[0];

    for (int i = 0; i < n; i++) {
        out_result->actions[i] = actions.cards[i];
        out_result->q_values[i] = q_values[i];
        if (maximize) {
            if (q_values[i] > best_value) { best_value = q_values[i]; best_action = actions.cards[i]; }
        } else {
            if (q_values[i] < best_value) { best_value = q_values[i]; best_action = actions.cards[i]; }
        }
    }

    out_result->best_action = best_action;
    out_result->value = best_value;
}

}  // extern "C"
