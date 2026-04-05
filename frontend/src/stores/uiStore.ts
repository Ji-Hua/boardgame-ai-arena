// uiStore.ts
// Responsibilities:
// - Stores purely UI-level states (hover, selection, panels, toggles)
// - Observable store for UI interactions
// Forbidden:
// - No game data storage
// - No backend communication
// - No game logic

import { RenderWall } from "../core/RenderState";

interface UIState {
  hoverPosition: [number, number] | null;
  selectedPosition: [number, number] | null;
  selectedWall: RenderWall | null;
  // Future: panel states, toggles, etc.
}

type Subscriber = (state: UIState) => void;

class UIStore {
  private state: UIState = {
    hoverPosition: null,
    selectedPosition: null,
    selectedWall: null,
  };
  private subscribers: Set<Subscriber> = new Set();

  getState(): UIState {
    return { ...this.state };
  }

  setHoverPosition(position: [number, number] | null): void {
    this.state.hoverPosition = position;
    this.notifySubscribers();
  }

  setSelectedPosition(position: [number, number] | null): void {
    this.state.selectedPosition = position;
    this.notifySubscribers();
  }

  setSelectedWall(wall: RenderWall | null): void {
    this.state.selectedWall = wall;
    this.notifySubscribers();
  }

  reset(): void {
    this.state = {
      hoverPosition: null,
      selectedPosition: null,
      selectedWall: null,
    };
    this.notifySubscribers();
  }

  subscribe(subscriber: Subscriber): () => void {
    this.subscribers.add(subscriber);
    // Immediately call with current state
    subscriber(this.getState());
    // Return unsubscribe function
    return () => {
      this.subscribers.delete(subscriber);
    };
  }

  private notifySubscribers(): void {
    const state = this.getState();
    this.subscribers.forEach((subscriber) => {
      subscriber(state);
    });
  }
}

export const uiStore = new UIStore();
