use crate::evaluator::evaluate_7;
use gto_core::{Card, CardSet};
use rand::seq::SliceRandom;

/// Equity result for a player.
#[derive(Clone, Debug)]
pub struct EquityResult {
    pub wins: u64,
    pub ties: u64,
    pub losses: u64,
    pub total: u64,
}

impl EquityResult {
    pub fn equity(&self) -> f64 {
        (self.wins as f64 + self.ties as f64 * 0.5) / self.total as f64
    }
}

/// Compute exact head-to-head equity by enumerating all possible boards.
///
/// `board` can be empty (preflop), 3 cards (flop), 4 cards (turn), or 5 cards (river).
pub fn equity_exact(
    hand1: [Card; 2],
    hand2: [Card; 2],
    board: &[Card],
) -> EquityResult {
    let mut dead = CardSet::empty();
    for &c in &hand1 {
        dead.insert(c);
    }
    for &c in &hand2 {
        dead.insert(c);
    }
    for &c in board {
        dead.insert(c);
    }

    let remaining: Vec<Card> = (0..52)
        .map(Card)
        .filter(|c| !dead.contains(*c))
        .collect();

    let cards_needed = 5 - board.len();
    let mut result = EquityResult {
        wins: 0,
        ties: 0,
        losses: 0,
        total: 0,
    };

    match cards_needed {
        0 => {
            // River: just evaluate
            eval_showdown(&hand1, &hand2, board, &mut result);
        }
        1 => {
            // Turn: 1 card to come
            for &c in &remaining {
                let mut full_board: Vec<Card> = board.to_vec();
                full_board.push(c);
                eval_showdown(&hand1, &hand2, &full_board, &mut result);
            }
        }
        2 => {
            // Flop: 2 cards to come
            let n = remaining.len();
            for i in 0..n {
                for j in (i + 1)..n {
                    let mut full_board: Vec<Card> = board.to_vec();
                    full_board.push(remaining[i]);
                    full_board.push(remaining[j]);
                    eval_showdown(&hand1, &hand2, &full_board, &mut result);
                }
            }
        }
        5 => {
            // Preflop: 5 cards to come
            let n = remaining.len();
            for i in 0..n {
                for j in (i + 1)..n {
                    for k in (j + 1)..n {
                        for l in (k + 1)..n {
                            for m in (l + 1)..n {
                                let full_board = [
                                    remaining[i],
                                    remaining[j],
                                    remaining[k],
                                    remaining[l],
                                    remaining[m],
                                ];
                                eval_showdown(
                                    &hand1,
                                    &hand2,
                                    &full_board,
                                    &mut result,
                                );
                            }
                        }
                    }
                }
            }
        }
        3 => {
            // Pre-flop with no board: 3 cards to come... wait, 5-0=5 or 5-3=2.
            // This case is "no board, 3 cards to deal" which shouldn't happen in Hold'em.
            // 5 - board.len() where board.len() == 2 means we have 2 board cards
            // which doesn't happen in Hold'em. Handle generically:
            let n = remaining.len();
            for i in 0..n {
                for j in (i + 1)..n {
                    for k in (j + 1)..n {
                        let mut full_board: Vec<Card> = board.to_vec();
                        full_board.push(remaining[i]);
                        full_board.push(remaining[j]);
                        full_board.push(remaining[k]);
                        eval_showdown(&hand1, &hand2, &full_board, &mut result);
                    }
                }
            }
        }
        _ => unreachable!("Invalid board length"),
    }

    result
}

fn eval_showdown(
    hand1: &[Card; 2],
    hand2: &[Card; 2],
    board: &[Card],
    result: &mut EquityResult,
) {
    let cards1 = [
        hand1[0], hand1[1], board[0], board[1], board[2], board[3], board[4],
    ];
    let cards2 = [
        hand2[0], hand2[1], board[0], board[1], board[2], board[3], board[4],
    ];

    let s1 = evaluate_7(&cards1);
    let s2 = evaluate_7(&cards2);

    result.total += 1;
    if s1 > s2 {
        result.wins += 1;
    } else if s1 == s2 {
        result.ties += 1;
    } else {
        result.losses += 1;
    }
}

/// Compute equity via Monte Carlo simulation.
///
/// Randomly completes the board and optionally the opponent's hand.
pub fn equity_monte_carlo(
    hand: [Card; 2],
    board: &[Card],
    num_opponents: usize,
    num_sims: usize,
    rng: &mut impl rand::Rng,
) -> f64 {
    let mut dead = CardSet::empty();
    dead.insert(hand[0]);
    dead.insert(hand[1]);
    for &c in board {
        dead.insert(c);
    }

    let mut deck: Vec<Card> = (0..52)
        .map(Card)
        .filter(|c| !dead.contains(*c))
        .collect();

    let mut wins = 0u64;
    let mut ties = 0u64;
    let total = num_sims as u64;

    for _ in 0..num_sims {
        deck.shuffle(rng);

        // Deal remaining board
        let mut full_board = [Card(0); 5];
        for (i, &c) in board.iter().enumerate() {
            full_board[i] = c;
        }
        let mut deck_idx = 0;
        for i in board.len()..5 {
            full_board[i] = deck[deck_idx];
            deck_idx += 1;
        }

        // Evaluate hero's hand
        let hero_cards = [
            hand[0],
            hand[1],
            full_board[0],
            full_board[1],
            full_board[2],
            full_board[3],
            full_board[4],
        ];
        let hero_strength = evaluate_7(&hero_cards);

        // Evaluate opponents
        let mut hero_wins = true;
        let mut hero_ties = false;
        for opp in 0..num_opponents {
            let opp_hand = [deck[deck_idx + opp * 2], deck[deck_idx + opp * 2 + 1]];
            let opp_cards = [
                opp_hand[0],
                opp_hand[1],
                full_board[0],
                full_board[1],
                full_board[2],
                full_board[3],
                full_board[4],
            ];
            let opp_strength = evaluate_7(&opp_cards);

            if opp_strength > hero_strength {
                hero_wins = false;
                hero_ties = false;
                break;
            } else if opp_strength == hero_strength {
                hero_ties = true;
            }
        }

        if hero_wins && !hero_ties {
            wins += 1;
        } else if hero_wins && hero_ties {
            ties += 1;
        }
    }

    (wins as f64 + ties as f64 * 0.5) / total as f64
}

/// Canonical hand class index (0-168) for preflop hands.
///
/// 0-12: pairs (22..AA)
/// 13-90: suited hands (lower rank, higher rank)
/// 91-168: offsuit hands (lower rank, higher rank)
pub fn hand_class(card1: Card, card2: Card) -> usize {
    let r1 = card1.rank().max(card2.rank());
    let r2 = card1.rank().min(card2.rank());
    let suited = card1.suit() == card2.suit();

    if r1 == r2 {
        r1 as usize // 0-12: pairs
    } else if suited {
        13 + hand_class_index(r2, r1) // 13-90: suited
    } else {
        91 + hand_class_index(r2, r1) // 91-168: offsuit
    }
}

/// Index within suited/offsuit groups: C(r1, 2) + r2 mapped to 0-77.
fn hand_class_index(low: u8, high: u8) -> usize {
    // Triangle number: sum of (high-1) + low
    (high as usize) * (high as usize - 1) / 2 + low as usize
}

/// Get a human-readable hand class name like "AKs", "QQ", "T9o".
pub fn hand_class_name(card1: Card, card2: Card) -> String {
    const RANKS: &[u8] = b"23456789TJQKA";
    let r1 = card1.rank().max(card2.rank());
    let r2 = card1.rank().min(card2.rank());
    let suited = card1.suit() == card2.suit();

    let c1 = RANKS[r1 as usize] as char;
    let c2 = RANKS[r2 as usize] as char;

    if r1 == r2 {
        format!("{}{}", c1, c2)
    } else if suited {
        format!("{}{}s", c1, c2)
    } else {
        format!("{}{}o", c1, c2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn c(s: &str) -> Card {
        let bytes = s.as_bytes();
        let rank = match bytes[0] {
            b'2' => 0, b'3' => 1, b'4' => 2, b'5' => 3, b'6' => 4,
            b'7' => 5, b'8' => 6, b'9' => 7, b'T' => 8, b'J' => 9,
            b'Q' => 10, b'K' => 11, b'A' => 12,
            _ => panic!("bad rank"),
        };
        let suit = match bytes[1] {
            b'c' => 0, b'd' => 1, b'h' => 2, b's' => 3,
            _ => panic!("bad suit"),
        };
        Card::new(rank, suit)
    }

    #[test]
    fn aa_vs_kk_equity() {
        // AA vs KK: AA should win ~81%
        let result = equity_exact(
            [c("As"), c("Ah")],
            [c("Ks"), c("Kh")],
            &[],
        );
        let eq = result.equity();
        assert!(
            (eq - 0.8235).abs() < 0.01,
            "AA vs KK equity = {:.4}, expected ~0.8235",
            eq
        );
    }

    #[test]
    fn aks_vs_qq_equity() {
        // AKs vs QQ: AKs should win ~46%
        let result = equity_exact(
            [c("As"), c("Ks")],
            [c("Qh"), c("Qd")],
            &[],
        );
        let eq = result.equity();
        assert!(
            (eq - 0.46).abs() < 0.02,
            "AKs vs QQ equity = {:.4}, expected ~0.46",
            eq
        );
    }

    #[test]
    fn river_equity() {
        // On the river, equity should be 0 or 1 (or 0.5 for tie)
        let board = [c("2d"), c("5h"), c("9s"), c("Jc"), c("3d")];
        let result = equity_exact(
            [c("As"), c("Ah")], // pair of aces
            [c("Ks"), c("Kh")], // pair of kings
            &board,
        );
        assert_eq!(result.equity(), 1.0); // AA beats KK on this board
    }

    #[test]
    fn hand_class_naming() {
        assert_eq!(hand_class_name(c("As"), c("Ah")), "AA");
        assert_eq!(hand_class_name(c("As"), c("Ks")), "AKs");
        assert_eq!(hand_class_name(c("Kh"), c("As")), "AKo");
        assert_eq!(hand_class_name(c("Ts"), c("9s")), "T9s");
        assert_eq!(hand_class_name(c("2h"), c("7d")), "72o");
    }

    #[test]
    fn hand_class_indices() {
        // Pairs: 0-12
        assert_eq!(hand_class(c("2s"), c("2h")), 0);
        assert_eq!(hand_class(c("As"), c("Ah")), 12);
        // Suited: 13-90
        let aks = hand_class(c("As"), c("Ks"));
        assert!(aks >= 13 && aks <= 90);
        // Offsuit: 91-168
        let ako = hand_class(c("Ah"), c("Ks"));
        assert!(ako >= 91 && ako <= 168);
    }

    #[test]
    fn monte_carlo_roughly_correct() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        // AA vs random hand: should be ~85%
        let eq = equity_monte_carlo(
            [c("As"), c("Ah")],
            &[],
            1,
            10_000,
            &mut rng,
        );
        assert!(
            (eq - 0.85).abs() < 0.03,
            "AA vs random MC equity = {:.4}, expected ~0.85",
            eq
        );
    }
}
