"""
hysteresis.py - Multi-operator Prandtl‑Ishlinskii hysteresis model
X and Y axes have independent states to eliminate coupling.
"""

import csv
import time
import numpy as np
import matplotlib.pyplot as plt

def play_operator(u_k, y_prev, r):
    return max(u_k - r, min(u_k + r, y_prev))

class NanoPositioner:
    def __init__(self, r_list=None, w_list=None, log_file="movement_log.csv"):
        # Default thresholds: 0 to 32, step 4, total 9 operators
        if r_list is None:
            self.r_list = np.linspace(0, 32, 9)
        else:
            self.r_list = np.array(r_list)

        # Default weights: exponential decay and normalization
        if w_list is None:
            w = np.exp(-self.r_list / 6.0)
            self.w_list = w / np.sum(w)
        else:
            self.w_list = np.array(w_list)

        self.n_ops = len(self.r_list)
        # Maintain separate states for X and Y
        self.states_x = np.zeros(self.n_ops)
        self.states_y = np.zeros(self.n_ops)

        self.x = 0.0
        self.y = 0.0
        self.cmd_x = 0.0
        self.cmd_y = 0.0

        self.log_file = log_file
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "cmd_x", "cmd_y", "actual_x", "actual_y"])
        self.start_time = time.time()

        self.history_cmd = []
        self.history_actual = []
        self.history_time = []

        print(f"Multi-operator PI model initialized, number of operators: {self.n_ops}")
        print(f"Threshold range: [{self.r_list[0]:.1f}, {self.r_list[-1]:.1f}]")
        print(f"Weight sum: {np.sum(self.w_list):.4f}")

    def _compute_output(self, u, states):
        """Given input u and corresponding state array, compute output"""
        total = 0.0
        for i in range(self.n_ops):
            r = self.r_list[i]
            prev = states[i]
            new_state = play_operator(u, prev, r)
            states[i] = new_state
            total += self.w_list[i] * new_state
        return total

    def _update_position(self):
        self.x = self._compute_output(self.cmd_x, self.states_x)
        self.y = self._compute_output(self.cmd_y, self.states_y)
        self._log_motion()
        self._record_history()
        return self.x, self.y

    def _log_motion(self):
        t = time.time() - self.start_time
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"{t:.6f}", f"{self.cmd_x:.3f}", f"{self.cmd_y:.3f}",
                             f"{self.x:.3f}", f"{self.y:.3f}"])

    def _record_history(self):
        self.history_cmd.append(self.cmd_x)
        self.history_actual.append(self.x)
        self.history_time.append(time.time() - self.start_time)
        if len(self.history_cmd) > 2000:
            self.history_cmd.pop(0)
            self.history_actual.pop(0)
            self.history_time.pop(0)

    def move_to(self, target_x, target_y):
        self.cmd_x = target_x
        self.cmd_y = target_y
        return self._update_position()

    def move(self, dx, dy):
        self.cmd_x += dx
        self.cmd_y += dy
        return self._update_position()

    def reset(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
        self.cmd_x = x
        self.cmd_y = y
        self.states_x[:] = x
        self.states_y[:] = y
        self._log_motion()

    def plot_hysteresis(self, title="Multi‑Operator PI Hysteresis"):
        if len(self.history_cmd) == 0:
            print("No data to plot")
            return
        plt.figure(figsize=(8, 6))
        plt.plot(self.history_cmd, self.history_actual, 'b-', linewidth=2, label='Actual')
        plt.plot(self.history_cmd, self.history_cmd, 'r--', linewidth=1.5, label='Ideal')
        plt.fill_between(self.history_cmd, self.history_actual, self.history_cmd,
                         alpha=0.2, color='blue')
        plt.xlabel('Command Position (μm)')
        plt.ylabel('Actual Position (μm)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    stage = NanoPositioner()
    u_up = np.linspace(0, 100, 200)
    u_down = np.linspace(100, 0, 200)
    u_seq = np.concatenate([u_up, u_down])
    actual = []
    for u in u_seq:
        a, _ = stage.move_to(u, 0)
        actual.append(a)
        time.sleep(0.005)
    plt.plot(u_seq, actual, 'b-', label='Actual')
    plt.plot(u_seq, u_seq, 'r--', label='Ideal')
    plt.xlabel('Command (μm)')
    plt.ylabel('Actual (μm)')
    plt.title('Multi‑Operator PI Model – Hysteresis Loop')
    plt.grid(True)
    plt.legend()
    plt.show()