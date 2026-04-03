use crate::action::Action;

/// Tracks pot, stacks, and action history during a hand.
#[derive(Clone, Debug)]
pub struct GameState {
    /// Current pot size.
    pub pot: u32,
    /// Stack sizes for each player.
    pub stacks: Vec<u32>,
    /// How much each player has invested in the current round.
    pub invested: Vec<u32>,
    /// History of actions taken.
    pub history: Vec<Action>,
    /// Number of players.
    pub num_players: usize,
}

impl GameState {
    pub fn new(num_players: usize, starting_stack: u32) -> Self {
        GameState {
            pot: 0,
            stacks: vec![starting_stack; num_players],
            invested: vec![0; num_players],
            history: Vec::new(),
            num_players,
        }
    }

    pub fn add_action(&mut self, action: Action) {
        self.history.push(action);
    }
}
