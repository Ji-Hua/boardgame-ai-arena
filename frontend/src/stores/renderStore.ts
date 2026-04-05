// renderStore.ts
// Responsibilities:
// - Single source of truth for RenderState
// - Observable store for UI components to subscribe to
// - Updated only by Controllers via StateMapper
// Forbidden:
// - No direct state mutation from UI components
// - No backend communication
// - No game logic

import { RenderState } from "../core/RenderState";

type Subscriber = (state: RenderState) => void;

class RenderStore {
  private state: RenderState;
  private subscribers: Set<Subscriber> = new Set();

  constructor() {
    // Initial empty state (seat-based, v0.2.0)
    this.state = {
      boardSize: 9,
      pawns: {
        1: [0, 4], // Seat 1 starts at row 0, col 4
        2: [8, 4], // Seat 2 starts at row 8, col 4
      },
      walls: [],
      wallsRemaining: {
        1: 10,
        2: 10,
      },
      currentSeat: 1, // Seat 1 goes first
      stepCount: 0,
      lastAction: null,
      actor: null,
      legalActions: undefined,
      isTerminal: false,
      result: null,
    };
  }

  getState(): RenderState {
    return this.state;
  }

  setState(newState: RenderState): void {
    this.state = newState;
    this.notifySubscribers();
  }

  subscribe(subscriber: Subscriber): () => void {
    this.subscribers.add(subscriber);
    // Immediately call with current state
    subscriber(this.state);
    // Return unsubscribe function
    return () => {
      this.subscribers.delete(subscriber);
    };
  }

  private notifySubscribers(): void {
    this.subscribers.forEach((subscriber) => {
      subscriber(this.state);
    });
  }
}

export const renderStore = new RenderStore();
