// ReplayController.ts
//
// Backend-path replay: sends actions through LiveController → WebSocket → Backend.
// State updates come back via state_update → StateMapper → renderStore.
// Replay is ONLY an action scheduler — no frontend simulation.
//
// The same pipeline as live game. Accept/reject determined by backend only.

import type { ReplayData, ReplayStep } from "../../types/Replay";
import type { ActionWire } from "../../api/websocket";
import { LiveController } from "../live/LiveController";

export class ReplayController {
  private data: ReplayData;
  private currentIndex: number = -1; // -1 = initial state
  private controller: LiveController;
  private busy: boolean = false;

  // Simple observer for UI
  private listeners: Set<() => void> = new Set();

  constructor(data: ReplayData) {
    this.data = data;
    this.controller = new LiveController();
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notify(): void {
    this.listeners.forEach((l) => l());
  }

  /** Total number of steps. */
  get totalSteps(): number {
    return this.data.steps.length;
  }

  /** Current step index (-1 = initial state). */
  get stepIndex(): number {
    return this.currentIndex;
  }

  /** Current step data (null if at initial state). */
  get currentStep(): ReplayStep | null {
    if (this.currentIndex < 0 || this.currentIndex >= this.data.steps.length) {
      return null;
    }
    return this.data.steps[this.currentIndex];
  }

  get isAtEnd(): boolean {
    return this.currentIndex >= this.data.steps.length - 1;
  }

  get isAtStart(): boolean {
    return this.currentIndex <= -1;
  }

  get isBusy(): boolean {
    return this.busy;
  }

  /** Whether the LiveController reports game ended. */
  get gameEnded(): boolean {
    return this.controller.gameEnded;
  }

  /** Bootstrap a backend game. Call once before navigation. */
  async init(): Promise<void> {
    await this.controller.bootstrap();
    this.currentIndex = -1;
    this.notify();
  }

  /** Clean up resources. */
  disconnect(): void {
    this.controller.disconnect();
  }

  // ── Navigation ───────────────────────────────────────────────

  /** Render the initial state (already done by bootstrap). */
  async renderInitial(): Promise<void> {
    if (this.currentIndex !== -1) {
      await this.jumpToStep(-1);
    }
  }

  /** Advance to the next step: send action through backend. */
  async stepForward(): Promise<void> {
    if (this.busy || this.currentIndex >= this.data.steps.length - 1) return;
    this.busy = true;

    this.currentIndex++;
    const step = this.data.steps[this.currentIndex];
    await this.sendAction(step);

    this.busy = false;
    this.notify();
  }

  /** Go back: restart game and replay up to previous step. */
  async stepBackward(): Promise<void> {
    if (this.busy || this.currentIndex <= -1) return;
    await this.jumpToStep(this.currentIndex - 1);
  }

  /** Jump to a specific step: restart game and replay up to target. */
  async jumpToStep(index: number): Promise<void> {
    if (this.busy) return;
    this.busy = true;

    const target = Math.max(-1, Math.min(index, this.data.steps.length - 1));

    // Restart: disconnect old game, bootstrap fresh one
    this.controller.disconnect();
    this.controller = new LiveController();
    await this.controller.bootstrap();
    this.currentIndex = -1;

    // Rapid-send all actions up to target step
    for (let i = 0; i <= target && i < this.data.steps.length; i++) {
      this.currentIndex = i;
      await this.sendAction(this.data.steps[i]);
    }

    this.busy = false;
    this.notify();
  }

  async jumpToStart(): Promise<void> {
    await this.jumpToStep(-1);
  }

  async jumpToEnd(): Promise<void> {
    await this.jumpToStep(this.data.steps.length - 1);
  }

  // ── Action sending ───────────────────────────────────────────

  /**
   * Convert a ReplayStep action to ActionWire and send through backend.
   * Waits for action_result before returning.
   */
  private async sendAction(step: ReplayStep): Promise<void> {
    const action = step.action;

    // action.position is [engine_y, engine_x]
    // ActionWire.target is [engine_x, engine_y] (backend convention)
    const wire: ActionWire = {
      player: action.seat,
      type: action.type,
      target: [action.position[1], action.position[0]],
    };

    await this.controller.takeActionAsync(wire);
    // Backend processes: accept → state_update → renderStore → UI
    //                    reject → action_result only, no state change
  }
}
